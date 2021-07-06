""" File used to quickly make any needed plot manually """

import pandas as pd
import matplotlib.pyplot as plt
from math import log

basSize = 5

comparison = "13neighbors-complete"
# "8neighbors-basConnect2-complete", "basConnectV1-completeH9-completeH16", "bSize&lRate"
# "8&12neighbors_basConnect2_complete"

# k_vals = [1, 2, 5, 10, 20, 100]
k = 1
neighbors = 13  # [8, 12]
# identifier = 2
lim_iter = 2000

figSize = {"default": (6.7, 5), "wide": (13, 5)}
sizeNum = {1: 1, 2: 1, 5: 1, 10: 2, 20: 2, 100: 3}

fig, ax = plt.subplots(1, figsize=figSize["default"])

filename = f"Training Outputs/meanNll_bas5_complete-5rep.csv"
df = pd.read_csv(filename, comment="#", index_col=0)
df = df.astype(float)
df = df.iloc[0:lim_iter]
df = df.rename(columns={f"CD-{k}": f"CD-{k}, Complete"})
df.plot(ax=ax, linewidth=1, alpha=0.8)

# filename = f"nll_progress_single_basConnect2_k{k}.csv"
# df = pd.read_csv(filename, comment="#")
# df = df.astype(float)
# df = df.iloc[0:lim_iter]
# df = df.rename(columns={"NLL": f"CD-{k}, BAS pattern v{identifier}"})
# df.plot(ax=ax, linewidth=1, alpha=0.8)

filename = f"Training Outputs/meanNll_bas5_neighbors-5rep.csv"
df = pd.read_csv(filename, comment="#", index_col=0)
df = df.astype(float)
df = df.iloc[0:lim_iter]
df.plot(ax=ax, linewidth=1, alpha=0.8)

plt.title("NLL evolution through RBM training")

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
# filename = f"Training Outputs/meanNll_bas4_complete-lRate05-25rep.csv"
# df = pd.read_csv(filename, comment="#", index_col=0)
# df = df.astype(float)
# df = df.iloc[0:lim_iter]
# nSamples = 2**(basSize+1)
# df = df.apply( lambda x: x * -nSamples )
# df.plot(ax=ax, linewidth=1, alpha=0.8)
#
# plt.title("NLL evolution through RBM training")
# plt.xlabel("Iteration")
# plt.ylabel("Log-Likelihood")
# # plt.ylabel("Average NLL")
# plt.grid(color="gray", linestyle=":", linewidth=.2)
# # plt.xlim(-10, lim_iter+10)
# # plt.ylim(-350, -100)
# plt.savefig("Plots/meanNll_25rep_bas4_complete-lRate05_LL-short.pdf", transparent=True)
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

# # Get all CD-k for a comparison ---------
# filenameC = f"Training Outputs/meanNll_bas4_complete-25iter.csv"
# dfC = pd.read_csv(filenameC, comment="#", index_col=0)
# dfC = dfC.astype(float)
# dfC = dfC.iloc[0:lim_iter]
#
# filenameN = f"Training Outputs/meanNll_bas4_neighbors-25iter.csv"
# dfN = pd.read_csv(filenameN, comment="#", index_col=0)
# dfN = dfN.astype(float)
# dfN = dfN.iloc[0:lim_iter]
#
# filename = f"Training Outputs/meanNll_bas4_BAScon-25iter.csv"
# dfS = pd.read_csv(filename, comment="#", index_col=0)
# dfS = dfS.astype(float)
# dfS = dfS.iloc[0:lim_iter]
#
# for k in k_vals:
#     fig, ax = plt.subplots(1, figsize=figSize["default"])
#
#     dfC = dfC.rename(columns={f"CD-{k}": f"CD-{k}, Complete"})
#     dfC[f"CD-{k}, Complete"].plot(ax=ax, linewidth=1, alpha=0.8)
#
#     dfS[f"CD-{k}, Specialist v{identifier}"].plot(ax=ax, linewidth=1, alpha=0.8)
#
#     for v in neighbors:
#         dfN[f"CD-{k}, {v} neighbors"].plot(ax=ax, linewidth=1, alpha=0.8)
#
#     plt.title(f"NLL evolution for CD-{k}")
#
#     plt.xlabel("Iteration")
#     plt.ylabel("Average NLL")
#     plt.grid(color="gray", linestyle=":", linewidth=.2)
#     plt.legend()
#
#     # Lower limit of NLL
#     nSamples = 2**(basSize+1)
#     limitante = - log(1.0/nSamples)
#     plt.plot([0, lim_iter], [limitante, limitante], "r--")
#
#     plt.savefig(f"Plots/nll_bas{basSize}_CD-{k}_comparison-{comparison}.pdf", transparent=True)
# # ---------


plt.xlabel("Iteration")
plt.ylabel("Average NLL")
plt.grid(color="gray", linestyle=":", linewidth=.2)
# plt.xlim(-10, lim_iter+10)
plt.legend()

# Lower limit of NLL
nSamples = 2**(basSize+1)
limitante = - log(1.0/nSamples)
plt.plot([0, lim_iter], [limitante, limitante], "r--")

plt.savefig(f"Plots/meanNLL_bas{basSize}_CD-{k}_comparison-{comparison}.pdf", transparent=True)
# plt.savefig(f"Plots/meanNll_25rep_bas4_BASconV{identifier}.pdf", transparent=True)  # neighbors
