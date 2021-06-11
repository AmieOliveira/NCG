#!/usr/bin/python3

import pandas as pd
import matplotlib.pyplot as plt

k_values = [10]  # [100, 20, 10, 5, 2, 1]
n_values = [4, 8, 12, 16]

lim_iter = int(1e3)
plotType = "basConnect"  # "complete", "neighbors", "basConnect"
basename = "nll_bas4"

fig, ax = plt.subplots(1)

if plotType == "complete":
    for k in k_values:
        filename = f"nll_progress_single_k{k}.csv"

        df = pd.read_csv(filename, comment="#")
        df = df.astype(float)
        df = df.iloc[0:lim_iter]
        df = df.rename(columns={"NLL": f"CD-{k}"})
        df.plot(ax=ax, linewidth=1)

    plt.title("NLL evolution through complete RBM training")

elif plotType == "neighbors":
    for k in k_values:
        for neighbors in n_values:
            filename = f"nll_progress_single_connect_neighbors{neighbors}_k{k}.csv"

            if neighbors == 16:
                filename = f"nll_progress_single_k{k}.csv"

            df = pd.read_csv(filename, comment="#")
            df = df.astype(float)
            df = df.iloc[0:lim_iter]
            df = df.rename(columns={"NLL": f"CD-{k}, {neighbors} neighbors"})
            df.plot(ax=ax, linewidth=1, alpha=0.8)

    plt.title("NLL evolution through RBM training")

elif plotType == "basConnect":
    k = 10
    files = [f"nll_progress_single_basConnect_k{k}.csv",
             f"nll_progress_single_basConnect2_k{k}.csv"]
    identifiers = ["v1", "v2"]
    for i in range(len(files)):
        filename = files[i]
        df = pd.read_csv(filename, comment="#")
        df = df.astype(float)
        df = df.iloc[0:lim_iter]
        df = df.rename(columns={"NLL": f"CD-{k}, {identifiers[i]}"})
        df.plot(ax=ax)  # alpha=0.7

    plt.title("NLL evolution through RBM training")

plt.xlabel("Iteration")
plt.ylabel("NLL")
plt.grid(color="gray", linestyle=":", linewidth=.2)

plt.savefig(f"{basename}_{plotType}.pdf", transparent=True)
plt.show()
