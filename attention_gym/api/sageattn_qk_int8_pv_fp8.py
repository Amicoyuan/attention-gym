from typing import Any, Optional

import torch

from ..kernel import (
    per_block_int8_triton_kernel,
    per_channel_fp8_triton_kernel,
    qk_int8_pv_fp8_forward_kernel,
)


@torch.compiler.disable
def sageattn_qk_int8_pv_fp8_triton(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    tensor_layout: str = "HND",
    is_causal: bool = False,
    sm_scale: Optional[float] = None,
    smooth_k: bool = True,
    return_lse: bool = False,
    **kwargs: Any,
) -> torch.Tensor:
    """
    SageAttention 2 with per-block INT8 quantization for Q and K, and per-block FP8 quantization V.

    Parameters
    ----------
    q : torch.Tensor
        The query tensor. Shape:
        - If `tensor_layout` is "HND": ``[batch_size, num_qo_heads, qo_len, head_dim]``.
        - If `tensor_layout` is "NHD": ``[batch_size, qo_len, num_qo_heads, head_dim]``.

    k : torch.Tensor
        The key tensor. Shape:
        - If `tensor_layout` is "HND": ``[batch_size, num_kv_heads, kv_len, head_dim]``.
        - If `tensor_layout` is "NHD": ``[batch_size, kv_len, num_kv_heads, head_dim]``.

    v : torch.Tensor
        The value tensor. Shape:
        - If `tensor_layout` is "HND": ``[batch_size, num_kv_heads, kv_len, head_dim]``.
        - If `tensor_layout` is "NHD": ``[batch_size, kv_len, num_kv_heads, head_dim]``.

    tensor_layout : str
        The tensor layout, either "HND" or "NHD".
        Default: "HND".

    is_causal : bool
        Whether to apply causal mask to the attention matrix. Only applicable when qo_len == kv_len (MHA).
        Default: False.

    sm_scale : Optional[float]
        The scale used in softmax, if not provided, will be set to ``1.0 / sqrt(head_dim)``.

    smooth_k : bool
        Whether to smooth the key tensor by subtracting the mean along the sequence dimension.
        Default: True.

    return_lse : bool
        Whether to return the log sum of the exponentiated attention weights. Used for cases like Ring Attention.
        Default: False.

    Returns
    -------
    torch.Tensor
        The output tensor. Shape:
        - If `tensor_layout` is "HND": ``[batch_size, num_qo_heads, qo_len, head_dim]``.
        - If `tensor_layout` is "NHD": ``[batch_size, qo_len, num_qo_heads, head_dim]``.

    torch.Tensor
        The logsumexp of each row of the matrix QK^T * scaling (e.g., log of the softmax normalization factor).
        Shape: ``[batch_size, num_qo_heads, qo_len]``.
        Only returned if `return_lse` is True.

    Note
    ----
    - ``num_qo_heads`` must be divisible by ``num_kv_heads``.
    - The tensors `q`, `k`, and `v` must have the dtype ``torch.float16``, ``torch.bfloat16``
    - All tensors must be on the same cuda device.
    - `smooth_k` will introduce slight overhead but will improve the accuracy under most circumstances.
    """

    dtype = q.dtype
    assert q.is_cuda, "Input tensors must be on cuda."
    assert dtype in [
        torch.float16,
        torch.bfloat16,
    ], "Input tensors must be in dtype of torch.float16 or torch.bfloat16"
    assert q.device == k.device == v.device, "All tensors must be on the same device."
    assert q.dtype == k.dtype == v.dtype, "All tensors must have the same dtype."

    torch.cuda.set_device(v.device)

    head_dim_og = q.size(-1)

    if head_dim_og < 64:
        q = torch.nn.functional.pad(q, (0, 64 - head_dim_og))
        k = torch.nn.functional.pad(k, (0, 64 - head_dim_og))
        v = torch.nn.functional.pad(v, (0, 64 - head_dim_og))
    elif head_dim_og > 64 and head_dim_og < 128:
        q = torch.nn.functional.pad(q, (0, 128 - head_dim_og))
        k = torch.nn.functional.pad(k, (0, 128 - head_dim_og))
        v = torch.nn.functional.pad(v, (0, 128 - head_dim_og))
    elif head_dim_og > 128:
        raise ValueError(f"Unsupported head_dim: {head_dim_og}")

    # assert last dim is contiguous
    assert (
        q.stride(-1) == 1 and k.stride(-1) == 1 and v.stride(-1) == 1
    ), "Last dim of qkv must be contiguous."

    seq_dim = 1 if tensor_layout == "NHD" else 2

    if smooth_k:
        km = k.mean(dim=seq_dim, keepdim=True)
        if return_lse:
            if tensor_layout == "NHD":
                lse_correction = (
                    torch.matmul(q.transpose(1, 2), km.transpose(1, 2).transpose(2, 3))
                    .squeeze(-1)
                    .to(torch.float32)
                )
            else:
                lse_correction = (
                    torch.matmul(q, km.transpose(2, 3)).squeeze(-1).to(torch.float32)
                )
    else:
        km = None

    kv_len = k.size(seq_dim)
    v_pad_len = 128 - (kv_len % 128) if kv_len % 128 != 0 else 0
    if v_pad_len > 0:
        if tensor_layout == "HND":
            v = torch.cat(
                [
                    v,
                    torch.zeros(
                        v.size(0),
                        v.size(1),
                        v_pad_len,
                        v.size(3),
                        dtype=v.dtype,
                        device=v.device,
                    ),
                ],
                dim=2,
            )
        else:
            v = torch.cat(
                [
                    v,
                    torch.zeros(
                        v.size(0),
                        v_pad_len,
                        v.size(2),
                        v.size(3),
                        dtype=v.dtype,
                        device=v.device,
                    ),
                ],
                dim=1,
            )

    if sm_scale is None:
        sm_scale = 1.0 / (head_dim_og**0.5)

    q_int8, q_scale, k_int8, k_scale = per_block_int8_triton_kernel(
        q, k, km=km, sm_scale=sm_scale, tensor_layout=tensor_layout
    )

    """
        when v dtype is fp8 transpose v and v_fp8 layout
        from [seq_dim, head_dim ] row-major ----->  [seq_dim, head_dim] col-major
    """
    v = v.transpose(-1, -2).contiguous().transpose(-1, -2)
    v_fp8, v_scale = per_channel_fp8_triton_kernel(
        v, tensor_layout=tensor_layout, smooth_v=False
    )

    o, lse = qk_int8_pv_fp8_forward_kernel(
        q_int8,
        k_int8,
        v_fp8,
        q_scale,
        k_scale,
        v_scale,
        sm_scale,
        is_causal=is_causal,
        tensor_layout=tensor_layout,
        output_dtype=dtype,
        return_lse=return_lse,
    )

    o = o[..., :head_dim_og]
    # cast o to origin dtype
    o = o.to(dtype)
    # 1.44269504 = 1/ln2

    if return_lse:
        return o, (
            lse / 1.44269504 + lse_correction * sm_scale
            if smooth_k
            else lse / 1.44269504
        )
    else:
        return o
