""" File used to quickly make any needed plot manually """

import pandas as pd
import matplotlib.pyplot as plt
from math import log

fig, ax = plt.subplots(1)

basSize = 4

comparison = "8neighbors-basConnect2-complete"
# "basConnectV1-completeH9-completeH16"

k = 10
neighbors = 8
identifier = 2
lim_iter = 1000

filename = f"nll_progress_single_k{k}.csv"
df = pd.read_csv(filename, comment="#")
df = df.astype(float)
df = df.iloc[0:lim_iter]
df = df.rename(columns={"NLL": f"CD-{k}, Complete"})
df.plot(ax=ax, linewidth=1, alpha=0.8)

filename = f"nll_progress_single_basConnect2_k{k}.csv"
df = pd.read_csv(filename, comment="#")
df = df.astype(float)
df = df.iloc[0:lim_iter]
df = df.rename(columns={"NLL": f"CD-{k}, BAS pattern v{identifier}"})
df.plot(ax=ax, linewidth=1, alpha=0.8)

filename = f"nll_progress_single_connect_neighbors{neighbors}_k{k}.csv"
df = pd.read_csv(filename, comment="#")
df = df.astype(float)
df = df.iloc[0:lim_iter]
df = df.rename(columns={"NLL": f"CD-{k}, {neighbors} neighbors"})
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


plt.xlabel("Iteration")
plt.ylabel("NLL")
plt.grid(color="gray", linestyle=":", linewidth=.2)

# Lower limit of NLL
nSamples = 2**(basSize+1)
limitante = - log(1.0/nSamples)
plt.plot([0, lim_iter], [limitante, limitante], "r--")

plt.savefig(f"Plots/nll_bas{basSize}_comparison-{comparison}.pdf", transparent=True)
