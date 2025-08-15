import matplotlib.pyplot as plt


def pack_show_plot_data_volume_vs_bandwidth(benchmarks):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 6))

    colors = ["blue", "green", "red", "cyan", "magenta", "yellow", "black"]

    for i, bench in enumerate(benchmarks):
        data_volume = bench.total_data
        bandwidth = bench.bandwidth

        sorted_data = sorted(zip(data_volume, bandwidth))
        data_volume_sorted, bandwidth_sorted = zip(*sorted_data)
        plt.plot(
            range(len(data_volume_sorted)),
            bandwidth_sorted,
            marker="o",
            linestyle="-",
            color=colors[i % len(colors)],
            label=bench.op_name,
        )

    plt.title("Data Volume vs Bandwidth Comparison", fontsize=16)
    plt.xlabel("Data Volume (B)", fontsize=14)
    plt.xticks(
        ticks=range(len(data_volume_sorted)), labels=data_volume_sorted, rotation=45
    )
    plt.ylabel("Bandwidth (GB/s)", fontsize=14)

    plt.grid(True)
    plt.legend(loc="upper left", fontsize=12)

    plt.tight_layout()
    plt.savefig("data_volume_vs_bandwidth_comparison.png", dpi=300)
    plt.show()
