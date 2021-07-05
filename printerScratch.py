""" File used to quickly make any needed plot manually """

import pandas as pd
import matplotlib.pyplot as plt
from math import log

basSize = 4

comparison = "bSize&lRate"
# "8neighbors-basConnect2-complete", "basConnectV1-completeH9-completeH16"

k = 5
neighbors = 4
# identifier = 2
lim_iter = 1000

figSize = {"default": (6.7, 5), "wide": (13, 5)}
sizeNum = {1: 1, 2: 1, 5: 1, 10: 2, 20: 2, 100: 3}

fig, ax = plt.subplots(1, figsize=figSize["default"])

# filename = f"Training Outputs/nll_progress_single_k{k}.csv"
# df = pd.read_csv(filename, comment="#")
# df = df.astype(float)
# df = df.iloc[0:lim_iter]
# df = df.rename(columns={"NLL": f"CD-{k}, Complete"})
# df.plot(ax=ax, linewidth=1, alpha=0.8)
#
# filename = f"nll_progress_single_basConnect2_k{k}.csv"
# df = pd.read_csv(filename, comment="#")
# df = df.astype(float)
# df = df.iloc[0:lim_iter]
# df = df.rename(columns={"NLL": f"CD-{k}, BAS pattern v{identifier}"})
# df.plot(ax=ax, linewidth=1, alpha=0.8)
#
# filename = f"nll_progress_single_connect_neighbors{neighbors}_k{k}.csv"
# df = pd.read_csv(filename, comment="#")
# df = df.astype(float)
# df = df.iloc[0:lim_iter]
# df = df.rename(columns={"NLL": f"CD-{k}, {neighbors} neighbors"})
# df.plot(ax=ax, linewidth=1, alpha=0.8)
#
# plt.title("NLL evolution through RBM training")

filename = f"Training Outputs/nll_progress_single_k{k}.csv"
df = pd.read_csv(filename, comment="#")
df = df.astype(float)
df = df.iloc[0:lim_iter]
df = df.rename(columns={"NLL": f"bSize=5, lRate=0.1"})
df.plot(ax=ax, linewidth=1, alpha=0.8)

filename = f"Training Outputs/Teste Parametros/nll_progress_complete_k1-bSize10.csv"
df = pd.read_csv(filename, comment="#", index_col=0)
df = df.astype(float)
df = df.iloc[0:lim_iter]
df = df.rename(columns={"NLL": f"bSize=10, lRate=0.1"})
df.plot(ax=ax, linewidth=1, alpha=0.8)

filename = f"Training Outputs/Teste Parametros/nll_progress_complete_k1lRate05.csv"
df = pd.read_csv(filename, comment="#", index_col=0)
df = df.astype(float)
df = df.iloc[0:lim_iter]
df = df.rename(columns={"NLL": f"bSize=5, lRate=0.05"})
df.plot(ax=ax, linewidth=1, alpha=0.8)

plt.title("Complete RBMs training")

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
# filename = f"Training Outputs/meanNll_bas4_complete.csv"
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
# plt.savefig("Plots/meanNll_25rep_bas4_complete_LL-short.pdf", transparent=True)
# # ---------

# # Neighbors Mean: CD-k solo ---------
# filename = f"Training Outputs/meanNll_bas4_neighbors.csv"
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

plt.xlabel("Iteration")
plt.ylabel("Average NLL")
plt.grid(color="gray", linestyle=":", linewidth=.2)
# plt.xlim(-10, lim_iter+10)
plt.legend()

# Lower limit of NLL
nSamples = 2**(basSize+1)
limitante = - log(1.0/nSamples)
plt.plot([0, lim_iter], [limitante, limitante], "r--")

plt.savefig(f"Plots/nll_bas{basSize}_CD-{k}_comparison-{comparison}.pdf", transparent=True)
# plt.savefig(f"Plots/meanNll_25rep_bas4_neighbors_CD-{k}.pdf", transparent=True)
