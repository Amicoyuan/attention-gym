import time

import torch


class Benchmark:
    def __init__(
        self,
        op_name,
        func,
        config,
        device=None,
        warm_up_times: int = 10,
        num_runs: int = 1000,
    ):
        assert op_name is not None
        assert func is not None
        self.op_name = op_name
        self.func = func
        self.config = config
        self.device = device
        self.total_data = []
        self.average_time = []
        self.bandwidth = []
        self.qo_len = []
        self.shapes = {}
        self.compute_performance = []
        self.warm_up_times = warm_up_times
        self.num_runs = num_runs

    def get_latency(self, op, *args, **kwargs):
        fn = lambda: op(*args, **kwargs)
        # Warm up
        torch.cuda.set_device(self.device)

        for _ in range(self.warm_up_times):
            _ = fn()

        torch.cuda.synchronize()
        start_time = time.time()

        # Benchmark
        with torch.no_grad():
            for _ in range(self.num_runs):
                output = fn()

        torch.cuda.synchronize()
        end_time = time.time()

        # Calculate average time
        average_time = (end_time - start_time) / self.num_runs
        self.average_time.append(average_time)

    def print_suit01(self):
        print("=" * 40)
        print("\033[1;34m" + "Op: " + "\033[0m" + f"{self.op_name}")
        print("\033[1;34m" + "Device: " + "\033[0m" + f"{self.device}")
        print("\033[1;34m" + "Warm-up times: " + "\033[0m" + f"{self.warm_up_times}")
        print("\033[1;34m" + "Num runs: " + "\033[0m" + f"{self.num_runs}")

    def record_shape_info(self, op, index, config, *args, **kwargs):
        pass

    def record(self, op, index, config, *args, **kwargs):
        self.record_shape_info(op, index, config, *args, **kwargs)
        bw = self.get_bandwidth(op, index, config, *args, **kwargs)
        cp = self.get_compute_performance(op, index, config, *args, **kwargs)
        self.bandwidth.append(bw)
        self.compute_performance.append(cp)

    def run(self):
        self.print_suit01()
        for index, config in enumerate(self.config):
            op, args, kwargs = self.prepare_input(self.func, config)
            self.get_latency(op, *args, **kwargs)  # Unpack args and kwargs correctly
            self.record(op, index, config, *args, **kwargs)
            self.show_result(index, config)
        print("=" * 40)

    def single_run(self):
        self.warm_up_times = 0
        self.num_runs = 1
        self.print_suit01()
        for index, config in enumerate(self.config):
            op, args, kwargs = self.prepare_input(self.func, config)
            self.get_latency(op, *args, **kwargs)  # Unpack args and kwargs correctly
            self.record(op, index, config, *args, **kwargs)
            self.show_result(index, config)
        print("=" * 40)

    def get_bandwidth(self, op, index, config, *args, **kwargs):
        pass

    def get_compute_performance(self, op, index, config, *args, **kwargs):
        pass

    def prepare_input(self, func, config):
        return

    def show_shape_info(self, index):
        pass

    def show_result(self, index, config):
        # 长度足够长的分隔线
        separator = "=" * 40
        index_line = f"Index: {index}".center(40)
        print(separator)
        print("\033[1;36m" + index_line + "\033[0m")
        print(separator)
        self.show_shape_info(index)
        print(
            "\033[1;34m"
            + "Average time: "
            + "\033[0m"
            + f"{self.average_time[index] * 1000:.3f} ms"
        )
        print(
            "\033[1;34m"
            + "Bandwidth: "
            + "\033[0m"
            + f"{self.bandwidth[index]:.3f} GB/s"
        )
        if "quantizetion" in config:
            if config["quantizetion"] == 1:
                print(
                    "\033[1;34m"
                    + "Compute performance: "
                    + "\033[0m"
                    + f"{self.compute_performance[index]:.3f} TOPS"
                )
        else:
            print(
                "\033[1;34m"
                + "Compute performance: "
                + "\033[0m"
                + f"{self.compute_performance[index]:.3f} TFLOPS"
            )
        print(separator)

    def show_plot_data_volume_vs_bandwidth(self):
        import matplotlib.pyplot as plt

        data_volume = self.total_data
        bandwidth = self.bandwidth
        print(data_volume)

        # 对数据进行排序
        sorted_data = sorted(zip(data_volume, bandwidth))
        data_volume_sorted, bandwidth_sorted = zip(*sorted_data)

        plt.figure(figsize=(10, 6))

        # 使用range进行均匀分布的 x 轴
        plt.plot(
            range(len(data_volume_sorted)),
            bandwidth_sorted,
            marker="o",
            linestyle="-",
            color="blue",
            label="Data Volume vs Bandwidth",
        )

        plt.title(str(self.op_name), fontsize=16)
        plt.xlabel("Data Volume (B)", fontsize=14)  # 注意x轴标签已经改为Index
        plt.xticks(
            ticks=range(len(data_volume_sorted)), labels=data_volume_sorted, rotation=45
        )
        plt.ylabel("Bandwidth (GB/s)", fontsize=14)

        plt.grid(True)

        plt.legend(loc="upper left", fontsize=12)

        plt.tight_layout()

        plt.savefig("data_volume_vs_bandwidth_plot.png", dpi=300)

        plt.show()
