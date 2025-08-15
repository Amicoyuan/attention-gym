import torch
import triton
import triton.language as tl


@triton.jit
def quant_per_channel_fp8_kernel(
    Input,
    Output,
    Scale,
    L,
    stride_iz,
    stride_ih,
    stride_in,
    stride_ik,
    stride_oz,
    stride_oh,
    stride_on,
    stride_ok,
    stride_sz,
    stride_sh,
    stride_sk,
    BLOCK_SIZE: tl.constexpr,
):
    block_xid = tl.program_id(0)
    block_yid = tl.program_id(1)
    block_zid = tl.program_id(2)

    input_ptrs = (
        Input + block_zid * stride_iz + block_yid * stride_ih + block_xid * stride_ik
    )
    output_ptrs = (
        Output + block_zid * stride_oz + block_yid * stride_oh + block_xid * stride_ok
    )
    scale_ptrs = (
        Scale + block_zid * stride_sz + block_yid * stride_sh + block_xid * stride_sk
    )

    scale = tl.load(scale_ptrs)

    col_offsets = tl.arange(0, BLOCK_SIZE)
    absmax_val = -1000000.0
    for n_idx in range(0, tl.cdiv(L, BLOCK_SIZE)):
        curr_off = col_offsets + n_idx * BLOCK_SIZE
        mask = curr_off < L
        x = tl.load(input_ptrs + curr_off, mask=mask).to(tl.float32)
        curr_max = tl.max(tl.abs(x))
        absmax_val = curr_max if curr_max > absmax_val else absmax_val

    scale = absmax_val / 448.0

    for n_idx in range(0, tl.cdiv(L, BLOCK_SIZE)):
        curr_off = col_offsets + n_idx * BLOCK_SIZE
        mask = curr_off < L
        x_fp32 = tl.load(input_ptrs + curr_off, mask=mask).to(tl.float32)
        fp8_y = tl.maximum(tl.minimum((x_fp32 / scale), 448.0), -448.0).to(
            tl.float8e4nv
        )
        tl.store(output_ptrs + curr_off, fp8_y, mask=mask)

    tl.store(scale_ptrs, scale)


def per_channel_fp8_triton_kernel(
    v, tensor_layout="HND", scale_max=448.0, smooth_v=False
):

    v_fp8 = torch.empty(v.shape, dtype=torch.float8_e4m3fn, device=v.device)
    v_fp8 = (
        v_fp8.to(torch.float8_e4m3fn).transpose(-1, -2).contiguous().transpose(-1, -2)
    )

    if tensor_layout == "HND":
        b, h_kv, kv_len, head_dim = v.shape

        stride_bz_v, stride_h_v, stride_seq_v, stride_v_k = (
            v.stride(0),
            v.stride(1),
            v.stride(2),
            v.stride(3),
        )
        stride_bz_vo, stride_h_vo, stride_seq_vo, stride_ko = (
            v_fp8.stride(0),
            v_fp8.stride(1),
            v_fp8.stride(2),
            v_fp8.stride(3),
        )
    elif tensor_layout == "NHD":
        b, kv_len, h_kv, head_dim = v.shape

        stride_bz_v, stride_h_v, stride_seq_v, stride_v_k = (
            v.stride(0),
            v.stride(2),
            v.stride(1),
            v.stride(3),
        )
        stride_bz_vo, stride_h_vo, stride_seq_vo, stride_ko = (
            v_fp8.stride(0),
            v_fp8.stride(2),
            v_fp8.stride(1),
            v_fp8.stride(3),
        )
    else:
        raise ValueError(f"Unknown tensor layout: {tensor_layout}")

    v_scale = torch.empty((b, h_kv, head_dim), device=v.device, dtype=torch.float32)

    stride_bz_vs, stride_h_vs, stride_vk_vs = (
        v_scale.stride(0),
        v_scale.stride(1),
        v_scale.stride(2),
    )

    grid = (head_dim, h_kv, b)
    BLOCK_SIZE = 256
    quant_per_channel_fp8_kernel[grid](
        v,
        v_fp8,
        v_scale,
        kv_len,
        stride_bz_v,
        stride_h_v,
        stride_seq_v,
        stride_v_k,
        stride_bz_vo,
        stride_h_vo,
        stride_seq_vo,
        stride_ko,
        stride_bz_vs,
        stride_h_vs,
        stride_vk_vs,
        BLOCK_SIZE,
    )
    return v_fp8, v_scale
