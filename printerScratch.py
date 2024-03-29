""" File used to quickly make any needed plot manually """

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from math import log

basSize = 4

comparison = "vNeighbors"
# "inicializacaoA_"
# "neighTypes", "neighbors10-14_BAScon2-4_complete"
# "8neighbors-basConnect2-complete", "basConnectV1-completeH9-completeH16", "bSize&lRate"
# "8&12neighbors_basConnect2_complete", "13neighbors-specialist-complete", "8&12neighbors_basConnect2_complete"

k_vals = [100, 20, 10, 5, 2, 1]
neighbors = [14, 12, 8, 4]   # 16 [14, 12, 10, 8, 6, 4]
# neighType = "spiral"      # "line", "spiral"
# versions = [2, 3, 4]
# identifier = 2
# k = 1
lim_iter = 10000
errorType = "quartile"        # None, "std", "quartile"
repeat = 25
# p_vals = [1, 0.5]
# p = 1
# seed = 1
lRate = 0.01
bSize = 5
H = 16
addOthers = False
zoom = False

plotSize = "default"            # "default", "wide"

figSize = {"default": (5.36, 4), "wide": (13, 5)}  # "default": (6.7, 5)
sizeNum = {1: 1, 2: 1, 5: 1, 10: 2, 20: 2, 100: 3}

fig, ax = plt.subplots(1, figsize=figSize[plotSize])

# # NLL truncation error plot ---------
# filename = f"Training Outputs/Teste MNIST/nll_mnist_complete_H16_CD-{k}_lr0.01_mBatch50_iter{lim_iter}_run{seed}.csv"
# df = pd.read_csv(filename, comment="#", index_col=0)
# df = df.astype(float)
# df = df.iloc[0:lim_iter]
#
# df = df.rename(columns={"NLL": f"Real Value"})
# df[f"Real Value"].plot(ax=ax, linewidth=1, alpha=0.8)
#
# filename = f"Training Outputs/Teste MNIST/nll_mnist_complete_H16_CD-{k}_lr0.01_mBatch50_iter{lim_iter}_run-truncated{seed}.csv"
# df = pd.read_csv(filename, comment="#", index_col=0)
# df = df.astype(float)
# df = df.iloc[0:lim_iter]
#
# df = df.rename(columns={"NLL": f"Truncated Estimate"})
# df[f"Truncated Estimate"].plot(ax=ax, linewidth=1, alpha=0.8)
#
# plt.title(f"NLL evolution through complete RBM training")
# # ---------

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

# Get all CD-k for a comparison ---------
linwdth = 1  # .5
shadalph = .3
colormap = 'tab20'
cmsize = 20.0

cmap = cm.get_cmap(colormap)

filenameC = f"Training Outputs/meanNll_bas{basSize}_complete-25rep.csv"
dfC = pd.read_csv(filenameC, comment="#", index_col=0)
dfC = dfC.astype(float)
dfC = dfC.iloc[0:lim_iter]

filenameNl = f"Training Outputs/meanNll_bas{basSize}_neighbors_line-{repeat}rep.csv"
dfNl = pd.read_csv(filenameNl, comment="#", index_col=0)
dfNl = dfNl.astype(float)
dfNl = dfNl.iloc[0:lim_iter]

filenameNs = f"Training Outputs/meanNll_bas{basSize}_neighbors_spiral-{repeat}rep.csv"
dfNs = pd.read_csv(filenameNs, comment="#", index_col=0)
dfNs = dfNs.astype(float)
dfNs = dfNs.iloc[0:lim_iter]

filename = f"Training Outputs/meanNll_bas{basSize}_BAScon-25rep.csv"
dfS = pd.read_csv(filename, comment="#", index_col=0)
dfS = dfS.astype(float)
dfS = dfS.iloc[0:lim_iter]

for k in k_vals:
    fig, ax = plt.subplots(1, figsize=figSize[plotSize])

    df = dfC.rename(columns={f"CD-{k}": "v = 16"})
    df["v = 16"].plot(ax=ax, linewidth=linwdth, alpha=0.8)

    i = 2*1
    for v in neighbors:
        # NOTE: argument 'cmap' does not work since I plot one at a time
        df = dfNl.rename(columns={f"CD-{k}, {v} neighbors in line": f"Line, v = {v}"})
        df[f"Line, v = {v}"].plot(ax=ax, linewidth=linwdth, alpha=0.8, color=cmap(i / cmsize))

        df = dfNs.rename(columns={f"CD-{k}, {v} neighbors in spiral": f"Spiral, v = {v}"})
        df[f"Spiral, v = {v}"].plot(ax=ax, linestyle='dashed', linewidth=linwdth, alpha=0.8, color=cmap((i+1)/cmsize))
        i = i+2

    # dfNltmp = dfNl.rename(columns={f"CD-{k}, 8 neighbors in line": f"Line (v = 8)"})
    # dfNltmp[f"Line (v = 8)"].plot(ax=ax, linewidth=linwdth, alpha=0.8)
    #
    # dfNsTmp = dfNs.rename(columns={f"CD-{k}, 8 neighbors in spiral": f"Spiral (v = 8)"})
    # dfNsTmp[f"Spiral (v = 8)"].plot(ax=ax, linewidth=linwdth, alpha=0.8)
    #
    # dfStmp = dfS.rename(columns={f"CD-{k}, Specialist v4": f"Stairs"})
    # dfStmp[f"Stairs"].plot(ax=ax, linewidth=linwdth, alpha=0.8)

    # dfStmp = dfS.rename(columns={f"CD-{k}, Specialist v2": f"Cross"})
    # dfStmp[f"Cross"].plot(ax=ax, linewidth=linwdth, alpha=0.8, color=cmap(i/cmsize))

    if errorType == "std":
        pass
    #    # ind = dfC.index
    #    # ind = dfNl.index
    #    ind = dfNs.index
    #    # ind = dfS.index
    #
    #    errorC = dfC[f"CD-{k} std"].to_numpy()
    #    meanC = dfC[f"CD-{k}, Complete"].to_numpy()
    #    ax.fill_between(ind, meanC - errorC, meanC + errorC, alpha=shadalph)
    #
    #    i = 2*1
    #    for v in neighbors:
    #        errorVl = dfNl[f"CD-{k}, {v} neighbors line - std"].to_numpy()
    #        meanVl = dfNl[f"CD-{k}, {v} neighbors in line"].to_numpy()
    #        ax.fill_between(ind, meanVl - errorVl, meanVl + errorVl, alpha=shadalph, color=cmap((i)/cmsize))
    #
    #        errorVs = dfNs[f"CD-{k}, {v} neighbors spiral - std"].to_numpy()
    #        meanVs = dfNs[f"CD-{k}, {v} neighbors in spiral"].to_numpy()
    #        ax.fill_between(ind, meanVs - errorVs, meanVs + errorVs, alpha=shadalph, color=cmap((i+1)/cmsize))
    #        i = i + 2
    #
    #    for identifier in versions:
    #        errorS = dfS[f"CD-{k}, Specialist v{identifier} - std"].to_numpy()
    #        meanS = dfS[f"CD-{k}, Specialist v{identifier}"].to_numpy()
    #        ax.fill_between(ind, meanS - errorS, meanS + errorS, alpha=shadalph, color=cmap(i/cmsize))
    #        i += 2

    elif errorType == "quartile":
        # ind = dfC.index
        # ind = dfNl.index
        ind = dfNs.index
        # ind = dfS.index

        # errorPlus = dfNl[f"CD-{k}, 8 neighbors line - q3"].to_numpy()
        # errorMinus = dfNl[f"CD-{k}, 8 neighbors line - q1"].to_numpy()
        # ax.fill_between(ind, errorMinus, errorPlus, alpha=shadalph)
        #
        # errorPlus = dfNs[f"CD-{k}, 8 neighbors spiral - q3"].to_numpy()
        # errorMinus = dfNs[f"CD-{k}, 8 neighbors spiral - q1"].to_numpy()
        # ax.fill_between(ind, errorMinus, errorPlus, alpha=shadalph)
        #
        # errorPlus = dfS[f"CD-{k}, Specialist v4 - q3"].to_numpy()
        # errorMinus = dfS[f"CD-{k}, Specialist v4 - q1"].to_numpy()
        # ax.fill_between(ind, errorMinus, errorPlus, alpha=shadalph)

        errorPlus = dfC[f"CD-{k} q3"].to_numpy()
        errorMinus = dfC[f"CD-{k} q1"].to_numpy()
        ax.fill_between(ind, errorMinus, errorPlus, alpha=shadalph)

        i = 2*1
        for v in neighbors:
            errorPlus = dfNl[f"CD-{k}, {v} neighbors line - q3"].to_numpy()
            errorMinus = dfNl[f"CD-{k}, {v} neighbors line - q1"].to_numpy()
            ax.fill_between(ind, errorMinus, errorPlus, alpha=shadalph, color=cmap((i)/cmsize))

            errorPlus = dfNs[f"CD-{k}, {v} neighbors spiral - q3"].to_numpy()
            errorMinus = dfNs[f"CD-{k}, {v} neighbors spiral - q1"].to_numpy()
            ax.fill_between(ind, errorMinus, errorPlus, alpha=shadalph, color=cmap((i+1)/cmsize))
            i = i + 2

        # errorPlus = dfS[f"CD-{k}, Specialist v2 - q3"].to_numpy()
        # errorMinus = dfS[f"CD-{k}, Specialist v2 - q1"].to_numpy()
        # ax.fill_between(ind, errorMinus, errorPlus, alpha=shadalph, color=cmap(i/cmsize))

        # for identifier in versions:
        #     errorPlus = dfS[f"CD-{k}, Specialist v{identifier} - q3"].to_numpy()
        #     errorMinus = dfS[f"CD-{k}, Specialist v{identifier} - q1"].to_numpy()
        #     ax.fill_between(ind, errorMinus, errorPlus, alpha=shadalph, color=cmap(i/cmsize))
        #     i += 2

    # plt.title(f"NLL evolution for CD-{k}")

    plt.xlabel("Epoch")
    plt.ylabel("Average NLL")
    plt.grid(color="gray", linestyle=":", linewidth=.2)
    plt.legend(ncol=2)  # prop={'size': 6})

    # Lower limit of NLL
    nSamples = 2**(basSize+1)
    limitante = - log(1.0/nSamples)
    plt.plot([0, lim_iter], [limitante, limitante], "k--")

    # plt.ylim(3.3, 15.3)
    # plt.xlim(2000, 10000)

    errorPrint = f"-{errorType}Err" if errorType else ""
    sizeAppend = f"-{plotSize}" if plotSize != "default" else ""

    plt.savefig(  # f"Plots/"
                f"/Users/amandaoliveira/Documents/Mestrado/Dissertação/imagens/searchSpace/"
                f"meanNLL_bas{basSize}_comparison-{comparison}_H{H}_CD-{k}_lr{lRate}_mBatch{bSize}_iter{lim_iter}"
                f"-{repeat}rep{errorPrint}{sizeAppend}.pdf",
                transparent=True
    )

# ---------

# # NLL error plot ---------
# filename = f"Training Outputs/meanNll_bas4_complete-25rep.csv"
# df = pd.read_csv(filename, comment="#", index_col=0)
# df = df.astype(float)
# df = df.iloc[0:lim_iter]
#
# for k in k_vals:
#     df[f"CD-{k}"].plot(ax=ax, linewidth=1, alpha=0.8)
#     errorPlus = df[f"CD-{k} q3"].to_numpy()
#     errorMinus = df[f"CD-{k} q1"].to_numpy()
#     ax.fill_between(df.index, errorMinus, errorPlus, alpha=0.3)
#
# plt.title(f"NLL evolution through complete RBM training")
# # ---------

# # Simple SGD optimization tests ---------
# repeat = 25
#
# # filenameComplete = f"Training Outputs/meanNll_bas4_complete-25rep.csv"
# # if lRate == 0.1:
# #     filenameComplete = f"Training Outputs/Old Learning Rate/meanNll_bas4_complete-25rep.csv"
# # dfC = pd.read_csv(filenameComplete, comment="#", index_col=0)
# # dfC = dfC.astype(float)
# # dfC = dfC.iloc[0:lim_iter]
#
# filename = f"Training Outputs/meanNll_bas4_SGD_lr{lRate}-25rep.csv"
# df = pd.read_csv(filename, comment="#", index_col=0)
# df = df.astype(float)
# df = df.iloc[0:lim_iter]
#
# if addOthers:
#     if lRate == 0.01:
#         filenameSNeigh = f"Training Outputs/meanNll_bas4_neighbors_spiral-25rep.csv"
#         dfNS = pd.read_csv(filenameSNeigh, comment="#", index_col=0)
#         dfNS = dfNS.astype(float)
#         dfNS = dfNS.iloc[0:lim_iter]
#
#         filenameLNeigh = f"Training Outputs/meanNll_bas4_neighbors_line-25rep.csv"
#         dfNL = pd.read_csv(filenameLNeigh, comment="#", index_col=0)
#         dfNL = dfNL.astype(float)
#         dfNL = dfNL.iloc[0:lim_iter]
#
#         # filenameSpec = f"Training Outputs/meanNll_bas4_BAScon-25rep.csv"
#         # dfS = pd.read_csv(filenameSpec, comment="#", index_col=0)
#         # dfS = dfS.astype(float)
#         # dfS = dfS.iloc[0:lim_iter]
#
# for k in k_vals:
#     fig, ax = plt.subplots(1, figsize=figSize[plotSize])
#
#     for p in p_vals:
#         df[f"CD-{k}, p = {p}"].plot(ax=ax, linewidth=1, alpha=0.9)
#
#     if addOthers:
#         if lRate == 0.01:
#             dfNLtmp = dfNL.rename(columns={f"CD-{k}, 10 neighbors in line": "10 neighbors in line"})
#             dfNLtmp["10 neighbors in line"].plot(ax=ax, linewidth=1, alpha=0.9)
#
#             dfNStmp = dfNS.rename(columns={f"CD-{k}, 12 neighbors in spiral": "12 neighbors in spiral"})
#             dfNStmp["12 neighbors in spiral"].plot(ax=ax, linewidth=1, alpha=0.9) #, color="tan")
#
#             # dfStmp = dfS.rename(columns={f"CD-{k}, Specialist v3": "Convolutional connectivity"})
#             # dfStmp["Convolutional connectivity"].plot(ax=ax, linewidth=1, alpha=0.9) #, color="lightsteelblue")
#
#     # compAlpha = 0.9 if addOthers else 0.6
#     # dfCtmp = dfC.rename(columns={f"CD-{k}": "Traditional RBM average"})
#     # dfCtmp["Traditional RBM average"].plot(ax=ax, linewidth=1, alpha=compAlpha, color="gray")
#
#     plt.title(f"Connectivity optimization for CD-{k}")
#
#     plt.xlabel("Iteration")
#     plt.ylabel("Average NLL")
#     plt.grid(color="gray", linestyle=":", linewidth=.2)
#
#     zoomStr = ""
#     if zoom:
#         zoomStr = "-zoom"
#         plt.xlim(6000, lim_iter)
#         if k == 100:
#             plt.ylim(3.5, 4.1)
#         elif k == 20:
#             plt.ylim(3.6, 4.1)
#         elif k == 10:
#             plt.ylim(3.7, 4.3)
#         elif k == 5:
#             plt.ylim(3.9, 4.6)
#         elif k == 2:
#             plt.ylim(4.5, 5.5)
#         elif k == 1:
#             plt.ylim(5.2, 6.2)
#     plt.legend()
#
#     # Lower limit of NLL
#     nSamples = 2 ** (basSize + 1)
#     limitante = - log(1.0 / nSamples)
#     plt.plot([0, lim_iter], [limitante, limitante], "r--")
#
#     errorPrint = f"-{errorType}Err" if errorType else ""
#
#     otherPlots = ""
#     if addOthers:
#         if lRate == 0.01:
#             otherPlots = "Full"
#
#     plt.savefig(f"Plots/meanNll_bas{basSize}_SGD_CD-{k}_lr{lRate}_pComparison{otherPlots}-{repeat}rep{errorPrint}{zoomStr}.pdf",
#                 transparent=True)
# # ---------

# # ---------
# filename = f"Training Outputs/meanNll_bas4_complete-25rep.csv"
# df = pd.read_csv(filename, comment="#", index_col=0)
# df = df.astype(float)
# df = df.iloc[0:lim_iter+1]
# df = df.rename(columns={f"CD-{k}": "Old code NLL mean"})
# df["Old code NLL mean"].plot(ax=ax, linewidth=1, alpha=0.8)
# filename = f"test_nll_novo_treino.csv"
# df = pd.read_csv(filename, comment="#", index_col=0)
# df = df.astype(float)
# df = df.iloc[0:lim_iter+1]
# df = df.rename(columns={"NLL": "Optimized code"})
# df.plot(ax=ax, linewidth=1, alpha=0.8)
#
# plt.title(f"NLL evolution through RBM training for CD-{k}")
# ---------

# # MNIST: Get all CD-k for a comparison ---------
# linwdth = 1  # .5
# shadalph = .3
# colormap = 'tab20'
# cmsize = 20.0
#
# cmap = cm.get_cmap(colormap)
#
# filenameC = "Training Outputs/meanNLL_mnist_complete_H500_lr0.1_mBatch50_iter100-5rep.csv"
# dfC = pd.read_csv(filenameC, comment="#", index_col=0)
# dfC = dfC.astype(float)
# dfC = dfC.iloc[0:lim_iter+1]
#
# filenameS = "Training Outputs/meanNLL_mnist_convolution_H484_lr0.1_mBatch50_iter100-5rep.csv"
# dfS = pd.read_csv(filenameS, comment="#", index_col=0)
# dfS = dfS.astype(float)
# dfS = dfS.iloc[0:lim_iter+1]
#
# filenameO = "Training Outputs/meanNLL_mnist_sgd_H500_lr0.1_mBatch50_iter100-5rep.csv"
# dfO = pd.read_csv(filenameO, comment="#", index_col=0)
# dfO = dfO.astype(float)
# dfO = dfO.iloc[0:lim_iter+1]
#
# for k in k_vals:
#     fig, ax = plt.subplots(1, figsize=figSize[plotSize])
#
#     dfC = dfC.rename(columns={f"CD-{k}": f"CD-{k}, Complete"})
#     dfC[f"CD-{k}, Complete"].plot(ax=ax, linewidth=linwdth, alpha=0.8)
#
#     dfS = dfS.rename(columns={f"CD-{k}": f"CD-{k}, Convolution"})
#     dfS[f"CD-{k}, Convolution"].plot(ax=ax, linewidth=linwdth, alpha=0.8)
#
#     for p in p_vals:
#         dfO[f"CD-{k}, p = {p}"].plot(ax=ax, linewidth=linwdth, alpha=0.8)
#
#     if errorType == "std":
#         ind = dfC.index
#
#         errorC = dfC[f"CD-{k} std"].to_numpy()
#         meanC = dfC[f"CD-{k}, Complete"].to_numpy()
#         ax.fill_between(ind, meanC - errorC, meanC + errorC, alpha=shadalph)
#
#         errorS = dfS[f"CD-{k} std"].to_numpy()
#         meanS = dfS[f"CD-{k}, Convolution"].to_numpy()
#         ax.fill_between(ind, meanS - errorS, meanS + errorS, alpha=shadalph)
#
#         for p in p_vals:
#             errorO = dfO[f"CD-{k}, p = {p} - std"].to_numpy()
#             meanO = dfO[f"CD-{k}, p = {p}"].to_numpy()
#             ax.fill_between(ind, meanO - errorO, meanO + errorO, alpha=shadalph)
#
#     elif errorType == "quartile":
#         ind = dfC.index
#
#         errorPlus = dfC[f"CD-{k} q3"].to_numpy()
#         errorMinus = dfC[f"CD-{k} q1"].to_numpy()
#         ax.fill_between(ind, errorMinus, errorPlus, alpha=shadalph)
#
#         errorPlus = dfS[f"CD-{k} q3"].to_numpy()
#         errorMinus = dfS[f"CD-{k} q1"].to_numpy()
#         ax.fill_between(ind, errorMinus, errorPlus, alpha=shadalph)
#
#         for p in p_vals:
#             errorPlus = dfO[f"CD-{k}, p = {p} - q3"].to_numpy()
#             errorMinus = dfO[f"CD-{k}, p = {p} - q1"].to_numpy()
#             ax.fill_between(ind, errorMinus, errorPlus, alpha=shadalph)
#
#
#     plt.title(f"NLL evolution for CD-{k}")
#
#     plt.xlabel("Epoch")
#     plt.ylabel("Average NLL")
#     plt.grid(color="gray", linestyle=":", linewidth=.2)
#     plt.legend(prop={'size': 6})
#
#     errorPrint = f"-{errorType}Err" if errorType else ""
#     sizeAppend = f"-{plotSize}" if plotSize != "default" else ""
#
#     plt.savefig(f"Plots/meanNLL_mnist_CD-{k}_comparison-{comparison}_H{H}_lr{lRate}_mBatch{bSize}"
#                 f"-{repeat}rep{errorPrint}{sizeAppend}.pdf", transparent=True)
# # ---------

#plt.xlabel("Epoch")
#plt.ylabel("Average NLL")
#plt.grid(color="gray", linestyle=":", linewidth=.2)
## plt.xlim(-10, lim_iter+10)
#plt.legend()

# # Lower limit of NLL
# nSamples = 2**(basSize+1)
# limitante = - log(1.0/nSamples)
# plt.plot([0, lim_iter], [limitante, limitante], "r--")

#plt.savefig(f"Plots/"
#            f"meanNll_bas{basSize}_comparison-{comparison}_H{H}_CD-{k}_lr{lRate}_mBatch{bSize}_iter{lim_iter}-rep{repeat}.pdf",
#            transparent=True)
