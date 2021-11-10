""" File used to quickly make any needed plot manually """

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from math import log

# basSize = 4

comparison = "connectivityScale-batch1k-limiar0.12"

# k_vals = [1, 10]  # [100, 20, 10, 5, 2, 1]
# neighbors = [14, 12, 10]   # 16 [14, 12, 10, 8, 6, 4]
# neighType = "spiral"      # "line", "spiral"
# versions = [2, 3, 4]
# identifier = 2
k = 10
lim_iter = 100
errorType = None        # None, "std", "quartile"
# repeat = 5
# p_vals = [1, 0.5]
p = 1
seed = 40
lRate = 0.01
bSize = 50
H = 15
addOthers = False
zoom = False

plotSize = "default"            # "default", "wide"

figSize = {"default": (6.7, 5), "wide": (13, 5)}
sizeNum = {1: 1, 2: 1, 5: 1, 10: 2, 20: 2, 100: 3}

fig, ax = plt.subplots(1, figsize=figSize[plotSize])


# filename = f"Training Outputs/Teste Gradiente/BatchSize50/nll_mnist_sgd-{p}_H{H}_CD-{k}_lr{lRate}_mBatch{bSize}_iter{lim_iter}_seed{seed}_run0.csv"
# df = pd.read_csv(filename, comment="#", index_col=0)
# df = df.astype(float)
# df = df.iloc[0:lim_iter+1]
#
# df = df.rename(columns={"NLL": f"Batch 50"})
# df[f"Batch 50"].plot(ax=ax, linewidth=1, alpha=0.8)

# filename = f"Training Outputs/Teste Gradiente/Batch1k_scale2/nll_mnist_sgd-{p}_H{H}_CD-{k}_lr{lRate}_mBatch{bSize}_iter{lim_iter}_seed{seed}_run0.csv"
# df = pd.read_csv(filename, comment="#", index_col=0)
# df = df.astype(float)
# df = df.iloc[0:lim_iter+1]
#
# df = df.rename(columns={"NLL": f"Time Scale 2"})
# df[f"Time Scale 2"].plot(ax=ax, linewidth=1, alpha=0.8)
#
# filename = f"Training Outputs/Teste Gradiente/BatchSize1k/nll_mnist_sgd-{p}_H{H}_CD-{k}_lr{lRate}_mBatch{bSize}_iter{lim_iter}_seed{seed}_run0.csv"
# df = pd.read_csv(filename, comment="#", index_col=0)
# df = df.astype(float)
# df = df.iloc[0:lim_iter+1]
#
# # df = df.rename(columns={"NLL": f"Batch 1k"})
# # df[f"Batch 1k"].plot(ax=ax, linewidth=1, alpha=0.8)
# df = df.rename(columns={"NLL": f"Time Scale 5"})
# df[f"Time Scale 5"].plot(ax=ax, linewidth=1, alpha=0.8)
#
# filename = f"Training Outputs/Teste Gradiente/Batch1k_scale10/nll_mnist_sgd-{p}_H{H}_CD-{k}_lr{lRate}_mBatch{bSize}_iter{lim_iter}_seed{seed}_run0.csv"
# df = pd.read_csv(filename, comment="#", index_col=0)
# df = df.astype(float)
# df = df.iloc[0:lim_iter+1]
#
# df = df.rename(columns={"NLL": f"Time Scale 10"})
# df[f"Time Scale 10"].plot(ax=ax, linewidth=1, alpha=0.8)

# filename = f"Training Outputs/Teste Gradiente/Batch1k_scale20/nll_mnist_sgd-{p}_H{H}_CD-{k}_lr{lRate}_mBatch{bSize}_iter{lim_iter}_seed{seed}_run0.csv"
# df = pd.read_csv(filename, comment="#", index_col=0)
# df = df.astype(float)
# df = df.iloc[0:lim_iter+1]
#
# # df = df.rename(columns={"NLL": f"Time Scale 20"})
# # df[f"Time Scale 20"].plot(ax=ax, linewidth=1, alpha=0.8)
# df = df.rename(columns={"NLL": f"Threshold 0,04"})
# df[f"Threshold 0,04"].plot(ax=ax, linewidth=1, alpha=0.8)
#
# filename = f"Training Outputs/Teste Gradiente/Batch1k_scale20_lim0.08/nll_mnist_sgd-{p}_H{H}_CD-{k}_lr{lRate}_mBatch{bSize}_iter{lim_iter}_seed{seed}_run0.csv"
# df = pd.read_csv(filename, comment="#", index_col=0)
# df = df.astype(float)
# df = df.iloc[0:lim_iter+1]
#
# df = df.rename(columns={"NLL": f"Threshold 0,08"})
# df[f"Threshold 0,08"].plot(ax=ax, linewidth=1, alpha=0.8)
#
filename = f"Training Outputs/Teste Gradiente/Batch1k_scale20_lim0.12/nll_mnist_sgd-{p}_H{H}_CD-{k}_lr{lRate}_mBatch{bSize}_iter{lim_iter}_seed{seed}_run0.csv"
df = pd.read_csv(filename, comment="#", index_col=0)
df = df.astype(float)
df = df.iloc[0:lim_iter+1]

# df = df.rename(columns={"NLL": f"Threshold 0,12"})
# df[f"Threshold 0,12"].plot(ax=ax, linewidth=1, alpha=0.8)
df = df.rename(columns={"NLL": f"Scale 20"})
df[f"Scale 20"].plot(ax=ax, linewidth=1, alpha=0.8)

filename = f"Training Outputs/Teste Gradiente/Batch1k_scale5_lim0.12/nll_mnist_sgd-{p}_H{H}_CD-{k}_lr{lRate}_mBatch{bSize}_iter{lim_iter}_seed{seed}_run0.csv"
df = pd.read_csv(filename, comment="#", index_col=0)
df = df.astype(float)
df = df.iloc[0:lim_iter+1]

df = df.rename(columns={"NLL": f"Scale 5"})
df[f"Scale 5"].plot(ax=ax, linewidth=1, alpha=0.8)

# filename = f"Training Outputs/Teste Gradiente/SingleBatch/nll_mnist_sgd-{p}_H{H}_CD-{k}_lr{lRate}_mBatch{bSize}_iter{lim_iter}_seed{seed}_run0.csv"
# df = pd.read_csv(filename, comment="#", index_col=0)
# df = df.astype(float)
# df = df.iloc[0:lim_iter+1]
#
# df = df.rename(columns={"NLL": f"Single Batch"})
# df[f"Single Batch"].plot(ax=ax, linewidth=1, alpha=0.8)

# filename = f"Training Outputs/Teste Gradiente/SingleBatch_scale20_lim0.04/nll_mnist_sgd-{p}_H{H}_CD-{k}_lr{lRate}_mBatch{bSize}_iter{lim_iter}_seed{seed}_run0.csv"
# df = pd.read_csv(filename, comment="#", index_col=0)
# df = df.astype(float)
# df = df.iloc[0:lim_iter+1]
#
# df = df.rename(columns={"NLL": f"Threshold 0,04"})
# df[f"Threshold 0,04"].plot(ax=ax, linewidth=1, alpha=0.8)
#
# filename = f"Training Outputs/Teste Gradiente/SingleBatch_scale20_lim0.005/nll_mnist_sgd-{p}_H{H}_CD-{k}_lr{lRate}_mBatch{bSize}_iter{lim_iter}_seed{seed}_run0.csv"
# df = pd.read_csv(filename, comment="#", index_col=0)
# df = df.astype(float)
# df = df.iloc[0:lim_iter+1]
#
# df = df.rename(columns={"NLL": f"Threshold 0,005"})
# df[f"Threshold 0,005"].plot(ax=ax, linewidth=1, alpha=0.8)
#
# filename = f"Training Outputs/Teste Gradiente/SingleBatch_scale20_lim0.001/nll_mnist_sgd-{p}_H{H}_CD-{k}_lr{lRate}_mBatch{bSize}_iter{lim_iter}_seed{seed}_run0.csv"
# df = pd.read_csv(filename, comment="#", index_col=0)
# df = df.astype(float)
# df = df.iloc[0:lim_iter+1]
#
# df = df.rename(columns={"NLL": f"Threshold 0,001"})
# df[f"Threshold 0,001"].plot(ax=ax, linewidth=1, alpha=0.8)

filename = f"Training Outputs/Teste Gradiente/nll_mnist_complete_H{H}_CD-{k}_lr{lRate}_mBatch{bSize}_iter{lim_iter}_seed{seed}_run0.csv"
df = pd.read_csv(filename, comment="#", index_col=0)
df = df.astype(float)
df = df.iloc[0:lim_iter+1]

df = df.rename(columns={"NLL": f"Traditional Training"})
df[f"Traditional Training"].plot(ax=ax, linewidth=1, alpha=0.8, color="gray")

plt.title(f"NLL evolution through RBM training")
# ---------



plt.xlabel("Epoch")
plt.ylabel("Average NLL")
plt.grid(color="gray", linestyle=":", linewidth=.2)
# plt.xlim(-10, lim_iter+10)
plt.ylim(150, 250)
plt.legend()

# # Lower limit of NLL
# nSamples = 2**(basSize+1)
# limitante = - log(1.0/nSamples)
# plt.plot([0, lim_iter], [limitante, limitante], "r--")

plt.savefig(f"Plots/Teste Gradiente/nll_mnist_{comparison}_H{H}_CD-{k}_lr{lRate}_mBatch{bSize}_iter{lim_iter}_seed{seed}.pdf", transparent=True)
