""" File used to quickly make any needed plot manually """

import pandas as pd
import matplotlib.pyplot as plt
from math import log

basSize = 4

comparison = ""
# "8neighbors-basConnect2-complete", "basConnectV1-completeH9-completeH16", "bSize&lRate"
# "8&12neighbors_basConnect2_complete", "13neighbors-specialist-complete", "8&12neighbors_basConnect2_complete"

k_vals = [100, 20, 10, 5, 2, 1]
# k = 1
neighbors = [16, 14, 12, 10, 8, 6, 4]
neighType = "line"      # "line", "spiral"
identifier = 2
lim_iter = 10000
errorType = "quartile"       # None, "std", "quartile"
repeat = 25

figSize = {"default": (6.7, 5), "wide": (13, 5)}
sizeNum = {1: 1, 2: 1, 5: 1, 10: 2, 20: 2, 100: 3}

# fig, ax = plt.subplots(1, figsize=figSize["default"])
#
# filename = f"Training Outputs/meanNll_bas{basSize}_complete-5rep.csv"
# df = pd.read_csv(filename, comment="#", index_col=0)
# df = df.astype(float)
# df = df.iloc[0:lim_iter]
# df = df.rename(columns={f"CD-{k}": f"CD-{k}, Complete"})
# df[f"CD-{k}, Complete"].plot(ax=ax, linewidth=1, alpha=0.8)
#
# if errorType == "std":
#     error = df[f"CD-{k} std"].to_numpy()
#     mean = df[f"CD-{k}, Complete"].to_numpy()
#     ax.fill_between(df.index, mean - error, mean + error, alpha=0.3)
#
#
# filename = f"Training Outputs/meanNll_bas{basSize}_BAScon-5rep.csv"
# df = pd.read_csv(filename, comment="#", index_col=0)
# df = df.astype(float)
# df = df.iloc[0:lim_iter]
# df[f"CD-{k}, Specialist v{identifier}"].plot(ax=ax, linewidth=1, alpha=0.8)
#
# if errorType == "std":
#     error = df[f"CD-{k}, Specialist v{identifier} - std"].to_numpy()
#     mean = df[f"CD-{k}, Specialist v{identifier}"].to_numpy()
#     ax.fill_between(df.index, mean - error, mean + error, alpha=0.3)
#
#
# filename = f"Training Outputs/meanNll_bas{basSize}_neighbors-5rep.csv"
# df = pd.read_csv(filename, comment="#", index_col=0)
# df = df.astype(float)
# df = df.iloc[0:lim_iter]
# df[f"CD-{k}, {neighbors} neighbors"].plot(ax=ax, linewidth=1, alpha=0.8)
#
# if errorType == "std":
#     error = df[f"CD-{k}, {neighbors} neighbors - std"].to_numpy()
#     mean = df[f"CD-{k}, {neighbors} neighbors"].to_numpy()
#     ax.fill_between(df.index, mean - error, mean + error, alpha=0.3)
#
# plt.title("NLL evolution through RBM training")

# # ---------
# filename = f"Training Outputs/nll_progress_complete_k{k}_H9-run-1.csv"
# df = pd.read_csv(filename, comment="#")
# df = df.astype(float)
# df = df.iloc[0:lim_iter]
# df = df.rename(columns={"NLL": "Complete pattern"})
# df.plot(ax=ax, linewidth=1, alpha=0.8)
# filename = f"Training Outputs/nll_progress_single_basConnect_k{k}.csv"
# df = pd.read_csv(filename, comment="#")
# df = df.astype(float)
# df = df.iloc[0:lim_iter]
# df = df.rename(columns={"NLL": "BAS pattern v1"})
# df.plot(ax=ax, linewidth=1, alpha=0.8)
#
# plt.title(f"NLL evolution through RBM training for CD-{k} and 9 hidden units")
# # ---------

# # LL plot ---------
# filename = f"Training Outputs/meanNll_bas4_complete-lRate01-25rep.csv"
# df = pd.read_csv(filename, comment="#", index_col=0)
# df = df.astype(float)
# df = df.iloc[0:lim_iter]
# nSamples = 2**(basSize+1)
# df = df.apply( lambda x: x * -nSamples )
#
# cols = [f"CD-{k}" for k in k_vals]
# df[cols].plot(ax=ax, linewidth=1, alpha=0.8)
#
# plt.title("LL evolution through RBM training")
# plt.xlabel("Iteration")
# plt.ylabel("Log-Likelihood")
# # plt.ylabel("Average NLL")
# plt.grid(color="gray", linestyle=":", linewidth=.2)
# # plt.xlim(-10, lim_iter+10)
# plt.ylim(-350, -100)
# plt.savefig("Plots/meanNll_25rep_bas4_complete-lRate01_LL-short.pdf", transparent=True)
# # ---------

# # Mean: CD-k solo ---------
# filename = f"Training Outputs/meanNll_bas4_BAScon-25iter.csv"
# # f"Training Outputs/meanNll_bas4_neighbors-25iter.csv"
#
# df = pd.read_csv(filename, comment="#", index_col=0)
# df = df.astype(float)
# cols = []
# lIdx = 3 + sizeNum[k] + 1
# for c in df.columns:
#     if c[:lIdx] == f"CD-{k},":
#         cols += [c]
# df[cols].plot(ax=ax, linewidth=1, alpha=0.8)
# plt.title(f"NLL evolution for CD-{k}")
# # ---------

# # Mean BAScon: Version solo solo, all CD-k ---------
# filename = f"Training Outputs/meanNll_bas4_BAScon-25iter.csv"
# # f"Training Outputs/meanNll_bas4_neighbors-25iter.csv"
#
# df = pd.read_csv(filename, comment="#", index_col=0)
# df = df.astype(float)
# cols = []
# lIdx = 2
# for c in df.columns:
#     if c[-lIdx:] == f"v{identifier}":
#         cols += [c]
# df[cols].plot(ax=ax, linewidth=1, alpha=0.8)
# plt.title(f"NLL evolution for Specialist v{identifier}")
# # ---------

# # Random neighbors test ---------
# filename = f"Training Outputs/nll_progress_mixing100_neighbors{neighbors}_k5_repeat25.csv"
# df = pd.read_csv(filename, comment="#", index_col=0)
# df = df.astype(float)
# df["4 random neighbors"] = df.mean(axis=1)
# df["4 random neighbors"].plot(ax=ax, linewidth=1, alpha=0.8)
#
# filename = "Training Outputs/meanNll_bas4_neighbors.csv"
# df = pd.read_csv(filename, comment="#", index_col=0)
# df = df.astype(float)
# df = df.iloc[0:lim_iter]
# df[f"{neighbors} neighbors"] = df[f"CD-{k}, {neighbors} neighbors"]
# df[f"{neighbors} neighbors"].plot(ax=ax, linewidth=1, alpha=0.8)
#
# plt.title(f"NLL evolution for CD-{k}")
# # ---------

# Get all CD-k for a comparison ---------
linwdth = 1  # .5
shadalph = .3

# filenameC = f"Training Outputs/meanNll_bas4_complete-25rep.csv"
# dfC = pd.read_csv(filenameC, comment="#", index_col=0)
# dfC = dfC.astype(float)
# dfC = dfC.iloc[0:lim_iter]

filenameNl = f"Training Outputs/meanNll_bas{basSize}_neighbors_line-{repeat}rep.csv"
dfNl = pd.read_csv(filenameNl, comment="#", index_col=0)
dfNl = dfNl.astype(float)
dfNl = dfNl.iloc[0:lim_iter]

# filename = f"Training Outputs/meanNll_bas4_BAScon-25rep.csv"
# dfS = pd.read_csv(filename, comment="#", index_col=0)
# dfS = dfS.astype(float)
# dfS = dfS.iloc[0:lim_iter]

for k in k_vals:
    fig, ax = plt.subplots(1, figsize=figSize["default"])

    # dfC = dfC.rename(columns={f"CD-{k}": f"CD-{k}, Complete"})
    # dfC[f"CD-{k}, Complete"].plot(ax=ax, linewidth=linwdth, alpha=0.8)

    # dfS[f"CD-{k}, Specialist v{identifier}"].plot(ax=ax, linewidth=linwdth, alpha=0.8)

    for v in neighbors:
        dfNl[f"CD-{k}, {v} neighbors in {neighType}"].plot(ax=ax, linewidth=linwdth, alpha=0.8)

    if errorType == "std":
        # ind = dfC.index
        ind = dfNl.index
        # errorC = dfC[f"CD-{k} std"].to_numpy()
        # meanC = dfC[f"CD-{k}, Complete"].to_numpy()
        # ax.fill_between(ind, meanC - errorC, meanC + errorC, alpha=shadalph)

        # errorS = dfS[f"CD-{k}, Specialist v{identifier} - std"].to_numpy()
        # meanS = dfS[f"CD-{k}, Specialist v{identifier}"].to_numpy()
        # ax.fill_between(ind, meanS - errorS, meanS + errorS, alpha=shadalph)

        for v in neighbors:
            errorV = dfNl[f"CD-{k}, {v} neighbors {neighType} - std"].to_numpy()
            meanV = dfNl[f"CD-{k}, {v} neighbors in {neighType}"].to_numpy()
            ax.fill_between(ind, meanV - errorV, meanV + errorV, alpha=shadalph)

    elif errorType == "quartile":
        # ind = dfC.index
        ind = dfNl.index
        # TODO: Others

        for v in neighbors:
            errorPlus = dfNl[f"CD-{k}, {v} neighbors {neighType} - q3"].to_numpy()
            errorMinus = dfNl[f"CD-{k}, {v} neighbors {neighType} - q1"].to_numpy()
            ax.fill_between(ind, errorMinus, errorPlus, alpha=0.3)

    plt.title(f"NLL evolution for CD-{k}")

    plt.xlabel("Iteration")
    plt.ylabel("Average NLL")
    plt.grid(color="gray", linestyle=":", linewidth=.2)
    plt.legend()

    # Lower limit of NLL
    nSamples = 2**(basSize+1)
    limitante = - log(1.0/nSamples)
    plt.plot([0, lim_iter], [limitante, limitante], "r--")

    # plt.ylim(3, 7)

    errorPrint = f"-{errorType}Err" if errorType else ""

    # plt.savefig(f"Plots/meanNLL_bas{basSize}_CD-{k}_comparison-{comparison}-{repeat}rep{errorPrint}.pdf", transparent=True)
    plt.savefig(f"Plots/meanNLL_bas{basSize}_CD-{k}_neighbors_line-{repeat}rep{errorPrint}.pdf", transparent=True)
# ---------


# plt.xlabel("Iteration")
# plt.ylabel("Average NLL")
# plt.grid(color="gray", linestyle=":", linewidth=.2)
# # plt.xlim(-10, lim_iter+10)
# plt.legend()
#
# # Lower limit of NLL
# nSamples = 2**(basSize+1)
# limitante = - log(1.0/nSamples)
# plt.plot([0, lim_iter], [limitante, limitante], "r--")
#
# plt.savefig(f"Plots/meanNLL_bas{basSize}_CD-{k}_comparison-{comparison}.pdf", transparent=True)
# # plt.savefig(f"Plots/meanNll_25rep_bas4_BASconV{identifier}.pdf", transparent=True)  # neighbors
