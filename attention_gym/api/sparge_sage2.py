import torch

from ..kernel import (
    get_block_map_meansim,
    hyperparameter_check,
    per_block_int8_triton_kernel,
    per_channel_fp8_triton_kernel,
    sparge_sage2_forward_triton_kernel,
)


@torch.compiler.disable
def sparge_sage2_triton(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attn_mask=None,
    dropout_p=0.0,
    is_causal: bool = False,
    scale=None,
    smooth_k: bool = True,
    simthreshd1=0.3,
    cdfthreshd=0.96,
    pvthreshd=20,
    attention_sink=False,
    tensor_layout: str = "HND",
    output_dtype=torch.float16,
    return_sparsity=False,
) -> torch.Tensor:
    """
    Sparge_sage2  with per-block INT8 quantization for Q and K, per channel FP8 quantization for V

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

    scale : float
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

    torch.cuda.set_device(q.device)
    origin_tensor_layout = tensor_layout
    if tensor_layout == "NHD":
        tensor_layout = "HND"
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

    dtype = q.dtype
    assert q.is_cuda, "Input tensors must be on cuda."
    assert dtype in [
        torch.float16,
        torch.bfloat16,
    ], "Input tensors must be in dtype of torch.float16 or torch.bfloat16"
    assert q.device == k.device == v.device, "All tensors must be on the same device."
    assert q.dtype == k.dtype == v.dtype, "All tensors must have the same dtype."

    # assert last dim is contiguous
    assert (
        q.stride(-1) == 1 and k.stride(-1) == 1 and v.stride(-1) == 1
    ), "Last dim of qkv must be contiguous."

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
        k = k - k.mean(dim=seq_dim, keepdim=True)

    k_block_indices = get_block_map_meansim(
        q,
        k,
        is_causal=is_causal,
        simthreshd1=simthreshd1,
        cdfthreshd=cdfthreshd,
        attention_sink=attention_sink,
    )

    if scale is None:
        sm_scale = 1.0 / (head_dim_og**0.5)

    q_int8, q_scale, k_int8, k_scale = per_block_int8_triton_kernel(
        q, k, sm_scale=sm_scale, tensor_layout=tensor_layout
    )
    # q_int8, q_scale, k_int8, k_scale = per_block_int8_sparge(q, k)
    pvthreshd = hyperparameter_check(pvthreshd, q.size(-3), q.device)

    """
        when v dtype is fp8 transpose v and v_fp8 layout
        from [seq_dim, head_dim ] row-major ----->  [seq_dim, head_dim] col-major
    """
    v = v.transpose(-1, -2).contiguous().transpose(-1, -2)
    v_fp8, v_scale = per_channel_fp8_triton_kernel(
        v, tensor_layout=tensor_layout, smooth_v=False
    )

    # when sparse fp8 need relayout because now have bug in inner kernel
    v_fp8 = v_fp8.to(torch.float8_e4m3fn).contiguous()

    o = sparge_sage2_forward_triton_kernel(
        q_int8,
        k_int8,
        k_block_indices,
        v_fp8,
        q_scale,
        k_scale,
        v_scale,
        sm_scale,
        pvthreshd,
        is_causal=is_causal,
        tensor_layout=tensor_layout,
        output_dtype=dtype,
    )

    o = o[..., :head_dim_og]
    # cast o to origin dtype
    o = o.to(output_dtype)

    if origin_tensor_layout == "NHD":
        o = o.transpose(1, 2)
    if return_sparsity:
        b, qo_len, n, d = q.shape
        kv_len = k.shape[seq_dim]
        lut_3 = d
        lut_2 = kv_len // 64
        lut_1 = qo_len // 128
        lut_0 = b
        if is_causal is False:
            qk_sparsity = 1 - (k_block_indices.sum().item()) / (
                lut_3 * lut_2 * lut_0 * lut_1
            )
        else:
            qk_sparsity = 1 - (k_block_indices.sum().item()) / (
                (lut_3 + 2) // 2 * lut_2 * lut_0 * lut_1
            )
        return o, qk_sparsity.item()
    else:
        return o
