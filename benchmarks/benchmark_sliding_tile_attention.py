import torch
from benchmark_common_utils import Benchmark

import attention_gym
from tests.sta_flex_attention import sta_flex_attention


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
        if config["tensor_layout"] == "NHD":
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
                config["kv_len"],
                config["head_num"],
                config["head_dim"],
                device=self.device,
                dtype=config["dtype"],
            )
            v = torch.randn(
                config["batch_size"],
                config["kv_len"],
                config["head_num"],
                config["head_dim"],
                device=self.device,
                dtype=config["dtype"],
            )
        else:
            q = torch.randn(
                config["batch_size"],
                config["head_num"],
                config["q_len"],
                config["head_dim"],
                device=self.device,
                dtype=config["dtype"],
            )
            k = torch.randn(
                config["batch_size"],
                config["head_num"],
                config["kv_len"],
                config["head_dim"],
                device=self.device,
                dtype=config["dtype"],
            )
            v = torch.randn(
                config["batch_size"],
                config["head_num"],
                config["kv_len"],
                config["head_dim"],
                device=self.device,
                dtype=config["dtype"],
            )
        window_sizes = config["window_size"] * config["head_num"]
        tile_size_t, tile_size_h, tile_size_w = config["tile_size"]
        t_dim, h_dim, w_dim = config["image_size"]
        return (
            func,
            (
                q,
                k,
                v,
                window_sizes,
                tile_size_t,
                tile_size_h,
                tile_size_w,
                t_dim,
                h_dim,
                w_dim,
            ),
            {},
        )

    def get_bandwidth(self, op, index, config, *args, **kwargs):
        batch_size = config["batch_size"]
        q_len = config["q_len"]
        k_len = config["kv_len"]
        v_len = config["kv_len"]
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
        # q
        self.shapes[index].append(args[0].shape)
        # k
        self.shapes[index].append(args[1].shape)
        # v
        self.shapes[index].append(args[2].shape)
        # window_sizes
        self.shapes[index].append(args[3])

    def show_shape_info(self, index):
        print(f"query tensor: {self.shapes[index][0]}")
        print(f"key tensor: {self.shapes[index][1]}")
        print(f"value tensor: {self.shapes[index][2]}")
        print(f"window_sizes : {self.shapes[index][3]}")


attention_configs = [
    {
        "batch_size": 1,
        "q_len": 24576,
        "kv_len": 24576,
        "head_num": 12,
        "head_dim": 128,
        "attention_type": "MHA",
        "dtype": torch.bfloat16,
        "window_size": [(3, 1, 8)],
        "image_size": (24, 32, 32),
        "tile_size": (2, 8, 8),
        "tensor_layout": "HND",
    },
    {
        "batch_size": 1,
        "q_len": 24576,
        "kv_len": 24576,
        "head_num": 12,
        "head_dim": 128,
        "attention_type": "MHA",
        "dtype": torch.bfloat16,
        "window_size": [(3, 1, 10)],
        "image_size": (24, 32, 32),
        "tile_size": (6, 8, 8),
        "tensor_layout": "HND",
    },
    {
        "batch_size": 1,
        "q_len": 24576,
        "kv_len": 24576,
        "head_num": 12,
        "head_dim": 128,
        "attention_type": "MHA",
        "dtype": torch.bfloat16,
        "window_size": [(6, 4, 4)],
        "image_size": (24, 32, 32),
        "tile_size": (6, 8, 8),
        "tensor_layout": "HND",
    },
]


def benchmark_suit_attention():
    device = torch.device("cuda:0")
    bench1 = AttentionBenchmark(
        "attention_gym.sliding_tile_attention_triton",
        attention_gym.sliding_tile_attention_triton,
        config=attention_configs,
        device=device,
    )
    bench1.run()


if __name__ == "__main__":
    benchmark_suit_attention()
