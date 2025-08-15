import torch
from benchmark_common_utils import Benchmark
from matplotlib_func import pack_show_plot_data_volume_vs_bandwidth

import attention_gym

try:
    from sageattention import sageattn_qk_int8_pv_fp8_cuda_sm90
except ModuleNotFoundError:
    raise Exception(
        "SageAttention is not installed. To use SageAttention 2.1.1, please compile from source."
    )


class AttentionBenchmark(Benchmark):
    def __init__(
        self,
        op_name,
        func,
        config,
        device=None,
    ):
        super().__init__(op_name, func, config, device=device)

    def prepare_input(self, func, config):
        q = torch.randn(
            config["batch_size"],
            config["q_len"],
            config["head_num"],
            config["head_dim"],
            device=self.device,
            dtype=config["dtype"],
        )
        k = torch.randn(
            config["batch_size"],
            config["k_len"],
            config["head_num"],
            config["head_dim"],
            device=self.device,
            dtype=config["dtype"],
        )
        v = torch.randn(
            config["batch_size"],
            config["v_len"],
            config["head_num"],
            config["head_dim"],
            device=self.device,
            dtype=config["dtype"],
        )
        if config["input_suit_type"] == 0:
            q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
            return func, (q, k, v), {"tensor_layout": "HND", "is_causal": False}

    def get_bandwidth(self, op, index, config, *args, **kwargs):
        batch_size = config["batch_size"]
        q_len = config["q_len"]
        k_len = config["k_len"]
        v_len = config["v_len"]
        head_num = config["head_num"]
        head_dim = config["head_dim"]
        data_read = (
            q_len * batch_size * head_num * head_dim
            + k_len * batch_size * head_num * head_dim
            + v_len * batch_size * head_num * head_dim
        )
        data_written = q_len * batch_size * head_num * head_dim
        if config["dtype"] == torch.bfloat16:
            dtype_size = 2
        total_data = data_read * dtype_size + data_written * dtype_size
        self.total_data.append(total_data)
        bw = total_data / (self.average_time[index] * 1e9)
        return bw

    def get_compute_performance(self, op, index, config, *args, **kwargs):

        batch_size = config["batch_size"]
        q_len = config["q_len"]
        head_num = config["head_num"]
        head_dim = config["head_dim"]
        hidden_dim = head_num * head_dim

        flops = (4 * batch_size * q_len * q_len * hidden_dim) + (
            3 * batch_size * q_len * q_len - batch_size * q_len
        )
        cp = flops / (self.average_time[index] * 1e12)
        return cp

    def record_shape_info(self, op, index, config, *args, **kwargs):
        fn = lambda: op(*args, **kwargs)
        output = fn()
        if index not in self.shapes:
            self.shapes[index] = []
        if config["input_suit_type"] == 0:
            # q
            self.shapes[index].append(args[0].transpose(1, 2).shape)
            # k
            self.shapes[index].append(args[1].transpose(1, 2).shape)
            # v
            self.shapes[index].append(args[2].transpose(1, 2).shape)
        else:
            # q
            self.shapes[index].append(args[0].shape)
            # k
            self.shapes[index].append(args[1].shape)
            # v
            self.shapes[index].append(args[2].shape)

    def show_shape_info(self, index):
        print(f"query tensor: {self.shapes[index][0]}")
        print(f"key tensor: {self.shapes[index][1]}")
        print(f"value tensor: {self.shapes[index][2]}")


attention_configs = [
    {
        "batch_size": 1,
        "q_len": 1024,
        "k_len": 1024,
        "v_len": 1024,
        "head_num": 5,
        "head_dim": 128,
        "attention_type": "MHA",
        "dtype": torch.bfloat16,
        "quantizetion": 1,
        "input_suit_type": 0,
    },
    {
        "batch_size": 1,
        "q_len": 8192,
        "k_len": 8192,
        "v_len": 8192,
        "head_num": 5,
        "head_dim": 128,
        "attention_type": "MHA",
        "dtype": torch.bfloat16,
        "quantizetion": 1,
        "input_suit_type": 0,
    },
    {
        "batch_size": 1,
        "q_len": 75600,
        "k_len": 75600,
        "v_len": 75600,
        "head_num": 5,
        "head_dim": 128,
        "attention_type": "MHA",
        "dtype": torch.bfloat16,
        "quantizetion": 1,
        "input_suit_type": 0,
    },
    {
        "batch_size": 1,
        "q_len": 28080,
        "k_len": 28080,
        "v_len": 28080,
        "head_num": 12,
        "head_dim": 128,
        "attention_type": "MHA",
        "dtype": torch.bfloat16,
        "quantizetion": 1,
        "input_suit_type": 0,
    },
]


def benchmark_suit_attention():
    device = torch.device("cuda:3")
    bench1 = AttentionBenchmark(
        "sageattn_qk_int8_pv_fp8_cuda_sm90",
        sageattn_qk_int8_pv_fp8_cuda_sm90,
        config=attention_configs,
        device=device,
    )
    bench2 = AttentionBenchmark(
        "attention_gym_qk_int8_pv_fp8_triton",
        attention_gym.sageattn_qk_int8_pv_fp8_triton,
        config=attention_configs,
        device=device,
    )
    bench1.run()
    bench2.run()


if __name__ == "__main__":
    benchmark_suit_attention()
