import numpy as np
import matplotlib.pyplot as plt
import os
import json

from data_helpers import pol_decay, exp_decay
from functions import (
    is_power_of_two,
    rand_nystrom_parallel,
    nuclear_error_relative,
    nuclear_error,
    plot_errors,
)

if __name__ == "__main__":

    folder = "results/numerical_stability_data/"
    Ps = [1, 4, 16, 64]
    colors = ["#0b3954", "#087e8b", "#ff5a5f", "#c81d25"]  # "#bfd7ea"
    n = 4096
    l = 64
    ks = [10, 25, 32, 64]
    datas_gaussian = []
    datas_SHRT = []

    for P in Ps:
        file_name = folder + "P" + str(P) + "_n" + str(n) + ".json"
        with open(os.path.join(file_name), "r") as file:
            data = json.load(file)
            # Checks
            if data["info"]["l"] != l:
                exit(-1)
            for i, k in enumerate(data["info"]["k"]):
                if k != ks[i]:
                    exit(-1)
            # Add data
            datas_gaussian.append(data["results"]["gaussian"])
            datas_SHRT.append(data["results"]["SHRT"])

    # Plot for each method
    results_folder = "results/numerical_stability_parallel/"
    # Gaussian method
    plot_errors(
        datas_gaussian,
        "gaussian",
        results_folder,
        ks,
        Ps,
        0,
        colors,
        "",
        y_label="nuclear norm relative error",
        pre_string_legend=r"$P=$",
    )

    # SHRT method
    plot_errors(
        datas_SHRT,
        "SHRT",
        results_folder,
        ks,
        Ps,
        0,
        colors,
        "",
        y_label="nuclear norm relative error",
        pre_string_legend=r"$P=$",
    )
