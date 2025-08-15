import pytest
import torch

import attention_gym

try:
    from spas_sage_attn import spas_sage2_attn_meansim_cuda
except ModuleNotFoundError:
    raise Exception(
        "SpargeAttn is not installed. To use SpargeAttn, please compile from source."
    )


@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize("seq_len", [8192])
@pytest.mark.parametrize("hidden_size", [3072])
@pytest.mark.parametrize("head_num", [24])
@pytest.mark.parametrize("causal", [False])
@pytest.mark.parametrize("device", ["cuda:0"])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
def test_attention_triton_01(
    batch_size, seq_len, hidden_size, head_num, causal, device, dtype
):
    torch.manual_seed(20)
    head_dim = hidden_size // head_num
    q = torch.empty(
        (batch_size, head_num, seq_len, head_dim), dtype=dtype, device=device
    ).normal_(mean=0.0, std=0.5)
    k = torch.empty(
        (batch_size, head_num, seq_len, head_dim), dtype=dtype, device=device
    ).normal_(mean=0.0, std=0.5)
    v = torch.empty(
        (batch_size, head_num, seq_len, head_dim), dtype=dtype, device=device
    ).normal_(mean=0.0, std=0.5)

    with torch.cuda.device(q.device.index):
        out = attention_gym.sparge_sage2_triton(
            q, k, v, tensor_layout="HND", is_causal=causal
        )
        out_ref = spas_sage2_attn_meansim_cuda(
            q, k, v, tensor_layout="HND", is_causal=causal
        )

    diff = out_ref - out
    abs_diff = torch.abs(diff)
    max_diff = torch.max(abs_diff).item()
    abs_diff = torch.sum(abs_diff).item()
    abs_diff = abs_diff / (batch_size * seq_len * head_num * head_dim)
    print(f"sparge_sage2_triton vs sparge_sage2_cuda abs_diff : {abs_diff}")
    print(f"sparge_sage2_triton vs sparge_sage2_cuda max_diff : {max_diff}")

    print("show tensor")
    print("sparge_sage2_triton")
    print(out_ref[0][0][0][:10])
    print("sparge_sage2_cuda")
    print(out_ref[0][0][0][:10])


if __name__ == "__main__":
    pytest.main(["test_sparge_sage2.py", "-s"])
