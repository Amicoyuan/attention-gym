from typing import Any, Optional

import torch

from ..kernel import flash_attention2_forward_kernel


@torch.compiler.disable
def flash_attention2_triton(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    tensor_layout: str = "HND",
    is_causal: bool = False,
    sm_scale: Optional[float] = None,
    return_lse: bool = False,
    **kwargs: Any,
) -> torch.Tensor:
    """
    Flash Attention 2 implementation using triton.

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

    o, lse = flash_attention2_forward_kernel(
        q,
        k,
        v,
        sm_scale,
        is_causal=is_causal,
        tensor_layout=tensor_layout,
        output_dtype=dtype,
        return_lse=return_lse,
    )

    o = o[..., :head_dim_og]
    # cast o to origin dtype
    o = o.to(dtype)

    if return_lse:
        return o, lse / 1.44269504
    else:
        return o
