import pytest
import torch
from sta_flex_attention import sta_flex_attention

import attention_gym


@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize("q_len", [24576])
@pytest.mark.parametrize("kv_len", [24576])
@pytest.mark.parametrize("hidden_size", [1536])
@pytest.mark.parametrize("head_num", [12])
@pytest.mark.parametrize("image_size", [(24, 32, 32)])
@pytest.mark.parametrize("tile_size", [(6, 8, 8)])
@pytest.mark.parametrize("windouw_size", [[(6, 8, 8)]])
@pytest.mark.parametrize("device", ["cuda:0"])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("tensor_layout", ["HND"])
def test_attention_triton_01(
    batch_size,
    q_len,
    kv_len,
    hidden_size,
    head_num,
    image_size,
    tile_size,
    windouw_size,
    device,
    dtype,
    tensor_layout,
):
    torch.manual_seed(20)
    head_dim = hidden_size // head_num

    if tensor_layout == "HND":
        q = torch.empty(
            (batch_size, head_num, q_len, head_dim), dtype=dtype, device=device
        ).normal_(mean=0.0, std=0.5)
        k = torch.empty(
            (batch_size, head_num, kv_len, head_dim), dtype=dtype, device=device
        ).normal_(mean=0.0, std=0.5)
        v = torch.empty(
            (batch_size, head_num, kv_len, head_dim), dtype=dtype, device=device
        ).normal_(mean=0.0, std=0.5)
    else:
        q = torch.empty(
            (batch_size, q_len, head_num, head_dim), dtype=dtype, device=device
        ).normal_(mean=0.0, std=0.5)
        k = torch.empty(
            (batch_size, kv_len, head_num, head_dim), dtype=dtype, device=device
        ).normal_(mean=0.0, std=0.5)
        v = torch.empty(
            (batch_size, kv_len, head_num, head_dim), dtype=dtype, device=device
        ).normal_(mean=0.0, std=0.5)

    q_ref = q.clone()
    k_ref = k.clone()
    v_ref = v.clone()
    window_size_t, window_size_h, window_size_w = windouw_size[0]

    print(f"window_size: {window_size_t, window_size_h, window_size_w}")

    tile_size_t, tile_size_h, tile_size_w = tile_size
    t_dim, h_dim, w_dim = image_size

    with torch.cuda.device(q.device.index):
        tri_out_ref = sta_flex_attention(
            q_ref,
            k_ref,
            v_ref,
            window_size_t,
            window_size_h,
            window_size_w,
            tile_size_t,
            tile_size_h,
            tile_size_w,
            t_dim,
            h_dim,
            w_dim,
        )
    windouw_sizes = windouw_size * head_num
    with torch.cuda.device(q.device.index):
        tri_out = attention_gym.sliding_tile_attention_triton(
            q,
            k,
            v,
            windouw_sizes,
            tile_size_t,
            tile_size_h,
            tile_size_w,
            t_dim,
            h_dim,
            w_dim,
            tensor_layout=tensor_layout,
            output_dtype=q_ref.dtype,
        )
    has_inf = torch.isinf(tri_out).any()
    print("triton out elements have inf:", has_inf.item())

    has_nan = torch.isnan(tri_out).any()
    print("triton out elements have nan:", has_nan.item())

    diff = tri_out - tri_out_ref
    abs_diff = torch.abs(diff)
    max_diff = torch.max(abs_diff).item()
    abs_diff = torch.sum(abs_diff).item()
    abs_diff = abs_diff / (batch_size * q_len * head_num * head_dim)
    print(f"sta triton vs sta flex attention abs_diff : {abs_diff}")
    print(f"sta triton vs sta flex attention max_diff : {max_diff}")
    print("show tensor")
    print("sta triton")
    print(tri_out[0][0][0][:10])
    print("sta flex attention")
    print(tri_out_ref[0][0][0][:10])

    print(tri_out_ref[0][0][0][-10:])
    print(tri_out[0][0][0][-10:])


if __name__ == "__main__":
    pytest.main(["test_sliding_tile_attention.py", "-s"])
