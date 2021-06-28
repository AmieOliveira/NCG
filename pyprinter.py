#!/usr/bin/python3

import pandas as pd
import matplotlib.pyplot as plt
from math import log

k_values = [10]  # [100, 20, 10, 5, 2, 1]
n_values = [4, 6, 8, 12, 16]
#H_values = [16, 9, 8, 6, 4]

size = "default"  # "default", "wide"
lim_iter = int(1e3)
plotType = "basConnect"  # "complete", "neighbors", "basConnect", "diffHidden_complete"

dataType = "bas"
dataSize = 4
basename = f"nll_{dataType}{dataSize}"

path = "Training Outputs/"

figSize = {"default": (6.7, 5), "wide": (13,5)}

fig, ax = plt.subplots(1, figsize=figSize[size])

if plotType == "complete":
    for k in k_values:
        filename = path + f"nll_progress_single_k{k}.csv"

        df = pd.read_csv(filename, comment="#")
        df = df.astype(float)
        df = df.iloc[0:lim_iter]
        df = df.rename(columns={"NLL": f"CD-{k}"})
        df.plot(ax=ax, linewidth=1)

    plt.title("NLL evolution through complete RBM training")

elif plotType == "neighbors":
    for k in k_values:
        for neighbors in n_values:
            filename = path + f"nll_progress_single_connect_neighbors{neighbors}_k{k}.csv"

            if neighbors == 16:
                filename = path + f"nll_progress_single_k{k}.csv"

            df = pd.read_csv(filename, comment="#")
            df = df.astype(float)
            df = df.iloc[0:lim_iter]
            df = df.rename(columns={"NLL": f"CD-{k}, {neighbors} neighbors"})
            df.plot(ax=ax, linewidth=1, alpha=0.8)

    plt.title("NLL evolution through RBM training")

elif plotType == "basConnect":

    for k in k_values:
        files = [f"nll_progress_single_basConnect_k{k}.csv",
                 f"nll_progress_single_basConnect2_k{k}.csv"]
        identifiers = ["v1", "v2"]
        for i in range(len(files)):
            filename = path + files[i]
            df = pd.read_csv(filename, comment="#")
            df = df.astype(float)
            df = df.iloc[0:lim_iter]
            df = df.rename(columns={"NLL": f"CD-{k}, {identifiers[i]}"})
            df.plot(ax=ax)  # alpha=0.7

    plt.title("NLL evolution through RBM training")

elif plotType == "diffHidden_complete":
    for k in k_values:
        baseFname = "nll_progress_complete_k{}_H{}-run-1.csv"

        for H in H_values:
            filename = path + baseFname.format(k, H)
            if H == 16:
                filename = path + f"nll_progress_single_k{k}.csv"

            df = pd.read_csv(filename, comment="#")
            df = df.astype(float)
            df = df.iloc[0:lim_iter]
            df = df.rename(columns={"NLL": f"CD-{k}, H = {H}"})
            df.plot(ax=ax, linewidth=1, alpha=0.8)

    plt.title("NLL evolution through RBM training")

plt.xlabel("Iteration")
plt.ylabel("Average NLL")
plt.grid(color="gray", linestyle=":", linewidth=.2)

# Lower limit of NLL
nSamples = 2**(dataSize+1)
limitante = - log(1.0/nSamples)
# print(f"NLL mínimo possível: {limitante}")
plt.plot([0, lim_iter], [limitante, limitante], "r--")

plt.savefig(f"Plots/{basename}_{plotType}.pdf", transparent=True)
plt.show()
