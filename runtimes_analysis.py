import numpy as np
import os
import json
import matplotlib.pyplot as plt

if __name__ == "__main__":
    RESULTS_FOLDER = "results"

    # LOAD JSON FILES
    FILE_NAMES = [
        "runtimes_1.json",
        "runtimes_4.json",
        "runtimes_16.json",
        "runtimes_64.json",
    ]

    n_procs = []
    runtimes_gaussian = []
    runtimes_SHRT = []

    for file_name in FILE_NAMES:
        with open(os.path.join(RESULTS_FOLDER, file_name), "r") as file:
            data = json.load(file)
            for entry in data:
                n_procs.append(entry["n_proc"])
                runtimes_gaussian.append(entry["runtimes_gaussian"])
                runtimes_SHRT.append(entry["runtimes_SHRT"])

    labels = ["Sketching", "Cholesky", "Z with substitution", "QR", "Rank-k truncation"]
    colors = ["#0b3954", "#087e8b", "#bfd7ea", "#ff5a5f", "#c81d25"]

    # PLOT RUNTIMES_GAUSSIAN
    fig, ax = plt.subplots()
    x = range(len(n_procs))
    bottom = [0] * len(n_procs)

    for i in range(len(labels)):
        values = [r[i] for r in runtimes_gaussian]
        ax.bar(x, values, bottom=bottom, label=labels[i], color=colors[i])
        bottom = [b + v for b, v in zip(bottom, values)]

    ax.set_xlabel("Number of Processors")
    ax.set_ylabel("Runtime [s]")
    ax.set_title("Runtimes with gaussian sketch matrix")
    ax.set_xticks(x)
    ax.set_xticklabels(n_procs)
    ax.legend()
    plt.tight_layout()
    plt.savefig("results/runtimes/runtimes_gaussian.png", bbox_inches="tight")
    plt.savefig(
        "results/runtimes/runtimes_gaussian.svg", format="svg", bbox_inches="tight"
    )

    # PLOT RUNTIME_SHRT
    fig, ax = plt.subplots()
    bottom = [0] * len(n_procs)

    for i in range(len(labels)):
        values = [r[i] for r in runtimes_SHRT]
        ax.bar(x, values, bottom=bottom, label=labels[i], color=colors[i])
        bottom = [b + v for b, v in zip(bottom, values)]

    ax.set_xlabel("Number of Processors")
    ax.set_ylabel("Runtime [s]")
    ax.set_title("Runtimes with SHRT sketch matrix")
    ax.set_xticks(x)
    ax.set_xticklabels(n_procs)
    ax.legend()
    plt.tight_layout()
    plt.savefig("results/runtimes/runtimes_SHRT.png", bbox_inches="tight")
    plt.savefig("results/runtimes/runtimes_SHRT.svg", format="svg", bbox_inches="tight")

    # PLOT EFFICIENCY + SPEED UP FOR RUNTIMES_GAUSSIAN
    total_runtimes_gaussian = [sum(r) for r in runtimes_gaussian]
    T_1_gaussian = total_runtimes_gaussian[0]
    speed_up_gaussian = [T_1_gaussian / T_p for T_p in total_runtimes_gaussian]
    efficiency_gaussian = [
        T_1_gaussian / (p * T_p) for p, T_p in zip(n_procs, total_runtimes_gaussian)
    ]

    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()

    ax1.plot(
        n_procs, speed_up_gaussian, "--", marker="D", label="Speed-up", color=colors[0]
    )
    ax2.plot(
        n_procs,
        efficiency_gaussian,
        "--",
        marker="D",
        label="Efficiency",
        color=colors[1],
    )

    ax1.tick_params(axis="y", labelcolor=colors[0])
    ax2.tick_params(axis="y", labelcolor=colors[1])

    ax1.set_xlabel("Number of Processors (n_proc)")
    ax1.set_ylabel("Speed-up", color=colors[0])
    ax2.set_ylabel("Efficiency", color=colors[1])
    ax1.set_title("Speed-up and Efficiency for Gaussian Runtimes")
    ax1.grid()

    fig.tight_layout()
    plt.savefig("results/runtimes/efficiency_gaussian.png", bbox_inches="tight")
    plt.savefig(
        "results/runtimes/efficiency_gaussian.svg", format="svg", bbox_inches="tight"
    )

    # PLOT EFFICIENCY + SPEEDUP FOR RUNTIMES_SHRT
    total_runtimes_SHRT = [sum(r) for r in runtimes_SHRT]
    T_1_SHRT = total_runtimes_SHRT[0]
    speed_up_SHRT = [T_1_SHRT / T_p for T_p in total_runtimes_SHRT]
    efficiency_SHRT = [
        T_1_SHRT / (p * T_p) for p, T_p in zip(n_procs, total_runtimes_SHRT)
    ]

    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()

    ax1.plot(
        n_procs, speed_up_SHRT, "--", marker="D", label="Speed-up", color=colors[0]
    )
    ax2.plot(
        n_procs, efficiency_SHRT, "--", marker="D", label="Efficiency", color=colors[1]
    )

    ax1.tick_params(axis="y", labelcolor=colors[0])
    ax2.tick_params(axis="y", labelcolor=colors[1])

    ax1.set_xlabel("Number of Processors (n_proc)")
    ax1.set_ylabel("Speed-up", color=colors[0])
    ax2.set_ylabel("Efficiency", color=colors[1])
    ax1.set_title("Speed-up and Efficiency for SHRT Runtimes")
    ax1.grid()

    fig.tight_layout()
    plt.savefig("results/runtimes/efficiency_SHRT.png", bbox_inches="tight")
    plt.savefig(
        "results/runtimes/efficiency_SHRT.svg", format="svg", bbox_inches="tight"
    )
