import pytest
import torch
import torch.nn.functional as F

import attention_gym


@pytest.mark.parametrize("batch_size", [2])
@pytest.mark.parametrize("q_len", [69120])
@pytest.mark.parametrize("kv_len", [69120])
@pytest.mark.parametrize("hidden_size", [1536])
@pytest.mark.parametrize("head_num", [24])
@pytest.mark.parametrize("causal", [False])
@pytest.mark.parametrize("device", ["cuda:0"])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("tensor_layout", ["HND"])
@pytest.mark.parametrize("return_lse", [False])
def test_attention_suit_01(
    batch_size,
    q_len,
    kv_len,
    hidden_size,
    head_num,
    causal,
    device,
    dtype,
    tensor_layout,
    return_lse,
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

    with torch.cuda.device(q.device.index):
        out_ref = F.scaled_dot_product_attention(q_ref, k_ref, v_ref, is_causal=causal)
        out = attention_gym.flash_attention2_triton(
            q,
            k,
            v,
            tensor_layout=tensor_layout,
            is_causal=causal,
            return_lse=return_lse,
        )

    diff = out - out_ref
    abs_diff = torch.abs(diff)
    max_diff = torch.max(abs_diff).item()
    abs_diff = torch.sum(abs_diff).item()
    abs_diff = abs_diff / (batch_size * q_len * head_num * head_dim)
    print(f"flash attention2 triton  vs torch FSDP abs_diff : {abs_diff}")
    print(f"flash attention2 triton vs torch FSDP max_diff : {max_diff}")
    # compare
    assert torch.allclose(out, out_ref, atol=1e-2, rtol=0)
    print("show tensor")
    print("torch FSDP")
    print(out_ref[0][0][0][:10])
    print("flash attention2 triton")
    print(out[0][0][0][:10])


if __name__ == "__main__":
    pytest.main(["test_flash_attention2.py", "-s"])
