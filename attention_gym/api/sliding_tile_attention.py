import torch

from ..kernel import (
    get_sliding_tile_attention_mask_triton,
    sta_triton_forward_triton_kernel,
)


@torch.compiler.disable
def sliding_tile_attention_triton(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    window_sizes: list = None,
    tile_size_t: int = 2,
    tile_size_h: int = 8,
    tile_size_w: int = 8,
    t_dim: int = 24,
    h_dim: int = 32,
    w_dim: int = 80,
    scale=None,
    tensor_layout: str = "HND",
    output_dtype=torch.float16,
) -> torch.Tensor:
    """
    sliding tile attention with triton kernel

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

    scale : float
        The scale used in softmax, if not provided, will be set to ``1.0 / sqrt(head_dim)``.

    Returns
    -------
    torch.Tensor
        The output tensor. Shape:
        - If `tensor_layout` is "HND": ``[batch_size, num_qo_heads, qo_len, head_dim]``.
        - If `tensor_layout` is "NHD": ``[batch_size, qo_len, num_qo_heads, head_dim]``.

    Note
    ----
    - ``num_qo_heads`` must be divisible by ``num_kv_heads``.
    - The tensors `q`, `k`, and `v` must have the dtype ``torch.float16``, ``torch.bfloat16``
    - All tensors must be on the same cuda device.
    """

    torch.cuda.set_device(q.device)
    origin_tensor_layout = tensor_layout
    if tensor_layout == "NHD":
        tensor_layout = "HND"
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

    dtype = q.dtype
    head_num = q.shape[1]
    device = q.device
    assert q.shape[0] == 1, "Only batch_size=1 is supported."
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

    len_window_sizes = len(window_sizes)
    assert (
        len_window_sizes == 1 or len_window_sizes == head_num
    ), f"Error: len_window_sizes must be 1 or equal to headnum ({head_num}), but got {len_window_sizes}"

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

    smooth_k = False
    if smooth_k:
        k = k - k.mean(dim=seq_dim, keepdim=True)

    all_head_block_mask = []

    if len_window_sizes == 1:
        window_size_t, window_size_h, window_size_w = window_sizes[0]
        k_block_indices = get_sliding_tile_attention_mask_triton(
            q.shape[seq_dim],
            k.shape[seq_dim],
            (window_size_t, window_size_h, window_size_w),
            (tile_size_t, tile_size_h, tile_size_w),
            (t_dim, h_dim, w_dim),
            0,
            device,
            0,
        )
        k_block_indices = k_block_indices.repeat(1, head_num, 1, 1)
    else:
        for head in range(head_num):
            window_size_t, window_size_h, window_size_w = window_sizes[head]
            block_mask = get_sliding_tile_attention_mask_triton(
                q.shape[seq_dim],
                k.shape[seq_dim],
                (window_size_t, window_size_h, window_size_w),
                (tile_size_t, tile_size_h, tile_size_w),
                (t_dim, h_dim, w_dim),
                0,
                device,
                0,
            )
            all_head_block_mask.append(block_mask)

    if len_window_sizes == head_num:
        k_block_indices = torch.cat(all_head_block_mask, dim=1)

    if scale is None:
        sm_scale = 1.0 / (head_dim_og**0.5)

    o = sta_triton_forward_triton_kernel(
        q,
        k,
        k_block_indices,
        v,
        sm_scale,
        is_causal=False,
        tensor_layout=tensor_layout,
        output_dtype=dtype,
    )

    o = o[..., :head_dim_og]
    # cast o to origin dtype
    o = o.to(output_dtype)

    if origin_tensor_layout == "NHD":
        o = o.transpose(1, 2)
    return o
