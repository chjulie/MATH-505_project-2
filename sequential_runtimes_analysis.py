import numpy as np
import os
import json
import matplotlib.pyplot as plt
# plt.style.use("ggplot")

if __name__ == "__main__":
    RESULTS_FOLDER = 'results'

    # LOAD JSON FILES
    FILE_NAME = 'runtimes_sequential_l.json'

    n = []
    l = []
    runtimes_gaussian = []
    runtimes_SHRT = []

    with open(os.path.join(RESULTS_FOLDER, FILE_NAME), 'r') as file:
        data = json.load(file)
        for entry in data:
            n.append(entry["matrix_size"])
            l.append(entry["sketch_dim"])
            runtimes_gaussian.append(entry["runtimes_gaussian"])
            runtimes_SHRT.append(entry["runtimes_SHRT"])

    # FILE_NAMES = ['runtimes_sequential_1.json', 'runtimes_sequential_4.json', 'runtimes_sequential_16.json', 'runtimes_sequential_64.json']

    # for file_name in FILE_NAMES:
    #     with open(os.path.join(RESULTS_FOLDER, file_name), 'r') as file:
    #         data = json.load(file)
    #         for entry in data:
    #             n.append(entry["matrix_size"])
    #             runtimes_gaussian.append(entry["runtimes_gaussian"])
    #             runtimes_SHRT.append(entry["runtimes_SHRT"])


    labels = ['Sketching', 'Cholesky', 'Z with substitution', 'QR', 'Rank-k truncation']
    # colors = ['#621708', '#720026', '#ce4257', '#ff7f51', '#f5cb5c']
    colors = ["#0b3954","#087e8b","#bfd7ea","#ff5a5f","#c81d25"]
    
    # PLOT RUNTIMES_GAUSSIAN
    fig, ax = plt.subplots()
    ax.set_xscale('log', base=2)
    x = [10, 11, 12, 13]
    bottom = [0] * len(n)

    for i in range(len(labels)):
        values = [r[i] for r in runtimes_gaussian]
        ax.bar(x, values, bottom=bottom, label=labels[i], color=colors[i])
        bottom = [b + v for b, v in zip(bottom, values)]

    ll = 128
    n = np.array(n)
    theoretical_time = (n**2) * ll + (ll**3) + n*ll**2
    ax2 = ax.twinx()
    ax2.plot(x,theoretical_time, '--', color='#000000',label='Theoretical runtime')

    ax.set_xlabel("Matrix size n")
    ax.set_ylabel("Runtime [s]")
    ax2.set_ylabel("Theoretical complexity")
    ax.set_title("Runtimes vs Matrix size")
    ax.set_xticks(x)
    ax.set_xticklabels(n)
    ax.legend(frameon=False)
    ax2.legend(bbox_to_anchor = (0.4, 0.72), frameon=False)
    plt.tight_layout()
    plt.savefig('results/runtimes/runtimes_sequential_gaussian.png', bbox_inches='tight')
    plt.savefig('results/runtimes/runtimes_sequential_gaussian.svg', format='svg', bbox_inches='tight')

    # PLOT RUNTIME_SHRT
    fig, ax = plt.subplots()
    bottom = [0] * len(n)

    for i in range(len(labels)):
        values = [r[i] for r in runtimes_SHRT]
        ax.bar(x, values, bottom=bottom, label=labels[i], color=colors[i])
        bottom = [b + v for b, v in zip(bottom, values)]

    ax2 = ax.twinx()
    theoretical_time = (n**2) * np.log(n) + (ll**3) + n*ll**2
    ax2.plot(x,theoretical_time, '--', color='#000000',label='Theoretical runtime')

    ax.set_xlabel("Matrix size n")
    ax.set_ylabel("Runtime [s]")
    ax2.set_ylabel("Theoretical complexity")
    ax.set_title("Runtimes vs Matrix size")
    ax.set_xticks(x)
    ax.set_xticklabels(n)
    ax.legend(frameon=False)
    ax2.legend(bbox_to_anchor = (0.4, 0.72),frameon=False)
    plt.tight_layout()
    plt.savefig('results/runtimes/runtimes_sequential_SHRT.png', bbox_inches='tight')
    plt.savefig('results/runtimes/runtimes_sequential_SHRT.svg', format='svg', bbox_inches='tight')


    # PLOT RUNTIMES_GAUSSIAN
    fig, ax = plt.subplots()
    ax.set_xscale('log', base=2)
    x = [7, 8, 9, 10]
    bottom = [0] * len(l)

    for i in range(len(labels)):
        values = [r[i] for r in runtimes_gaussian]
        ax.bar(x, values, bottom=bottom, label=labels[i], color=colors[i])
        bottom = [b + v for b, v in zip(bottom, values)]

    n = 1024
    l = np.array(l)
    theoretical_time = (n**2) * l + (l**3) + n*l**2
    ax2 = ax.twinx()
    ax2.plot(x,theoretical_time, '--', color='#000000',label='Theoretical runtime')

    ax.set_xlabel("Sketching dimension l")
    ax.set_ylabel("Runtime [s]")
    ax2.set_ylabel("Theoretical complexity")
    ax.set_title("Runtimes vs Sketching dimension")
    ax.set_xticks(x)
    ax.set_xticklabels(l)
    ax.legend(frameon=False)
    ax2.legend(bbox_to_anchor = (0.4, 0.72), frameon=False)
    plt.tight_layout()
    plt.savefig('results/runtimes/runtimes_sequential_gaussian_l.png', bbox_inches='tight')
    plt.savefig('results/runtimes/runtimes_sequential_gaussian_l.svg', format='svg', bbox_inches='tight')

    # PLOT RUNTIME_SHRT
    fig, ax = plt.subplots()
    bottom = [0] * len(l)

    for i in range(len(labels)):
        values = [r[i] for r in runtimes_SHRT]
        ax.bar(x, values, bottom=bottom, label=labels[i], color=colors[i])
        bottom = [b + v for b, v in zip(bottom, values)]

    ax2 = ax.twinx()
    theoretical_time = (n**2) * np.log(n) + (l**3) + n*l**2
    ax2.plot(x,theoretical_time, '--', color='#000000',label='Theoretical runtime')

    ax.set_xlabel("Sketching dimension l")
    ax.set_ylabel("Runtime [s]")
    ax2.set_ylabel("Theoretical complexity")
    ax.set_title("Runtimes vs Sketching dimension")
    ax.set_xticks(x)
    ax.set_xticklabels(l)
    ax.legend(frameon=False)
    ax2.legend(bbox_to_anchor = (0.4, 0.72),frameon=False)
    plt.tight_layout()
    plt.savefig('results/runtimes/runtimes_sequential_SHRT_l.png', bbox_inches='tight')
    plt.savefig('results/runtimes/runtimes_sequential_SHRT_l.svg', format='svg', bbox_inches='tight')
    
    print('Algorithm finished successfully!')