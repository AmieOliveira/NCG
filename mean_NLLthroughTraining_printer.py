#!/usr/bin/python3

import pandas as pd
import matplotlib.pyplot as plt
from math import log
import numpy as np

k_values = [10]  # [100, 20, 10, 5, 2, 1]
v_values = [588, 392, 196]  # [16, 32, 48]  # [8, 13, 18, 23]  # [16, 14, 12, 10, 8, 6, 4]
# versions = [2, 3, 4]  # [1, 2]
p_val = [1, 0.75, 0.5, 0.1]  # [1, 0.75, 0.5, 0.25]

size = "default"  # "default", "wide"
lim_iter = int(60)
plotType = "neighborsLine"
# "complete", "neighbors", "BAScon", "SGD" for BAS
# "complete", "convolution", "sgd" for MNIST
neighType = "line"
errorType = None  # None, "std", "quartile"
repeat = 10

periodoNLL = 10

dataType = "mnist"  # "bas", "mnist"
dataSize = 0
lRate = 0.1
bSize = 50
hiddenUnits = 784

basename = f"{dataType}{dataSize}" if dataType == "bas" else dataType
inputPath = "result/mnist_neighbors"  # f"result/{plotType}"
outputPath = inputPath

inputBaseBAS = {"complete": "nll_bas{}_complete_H{}_CD-{}_lr{}_mBatch{}_iter{}_run{}.csv",
                "neighbors": "nll_bas{}_neighbors{}-{}_H{}_CD-{}_lr{}_mBatch{}_iter{}_run{}.csv",
                "BAScon": "nll_bas{}_BASconV{}_H{}_CD-{}_lr{}_mBatch{}_iter{}_run{}.csv",
                "SGD": "nll_bas{}_sgd-{}_H{}_CD-{}_lr{}_mBatch{}_iter{}_run{}.csv"}
inputBaseMNIST = "nll_mnist_{}_H{}_CD-{}_lr{}_mBatch{}_iter{}_run{}.csv"

figSize = {"default": (6.7, 5), "wide": (13, 5)}

fig, ax = plt.subplots(1, figsize=figSize[size])

meanDF = pd.DataFrame()
indexes = np.array(list(range(0, lim_iter + 1, periodoNLL)))  # NOTE: THIS IS HANDMADE AND SHOULD BE CHANGED ACCORDINGLY
# print(indexes)


if plotType in ["complete", "convolution"]:
    # NOTE: Convolution is for MNIST only (for BAS use BAScon v3)
    for k in k_values:
        dfList = []

        for r in range(repeat):
            if dataType == "bas":
                try:
                    filename = inputPath + "/" + inputBaseBAS[plotType].format(dataSize, hiddenUnits, k, lRate, bSize,
                                                                               lim_iter, r)
                    df = pd.read_csv(filename, comment="#", index_col=0)
                except FileNotFoundError:
                    filename = inputPath + "/" + "nll_progress_complete_k{}-run{}.csv".format(k, r)
                    df = pd.read_csv(filename, comment="#")  # index_col=0
            elif dataType == "mnist":
                filename = inputPath + "/" + inputBaseMNIST.format(plotType, hiddenUnits, k, lRate, bSize, lim_iter, r)
                df = pd.read_csv(filename, comment="#", index_col=0)

            df = df.astype(float)
            df = df.iloc[0:lim_iter + 1]
            df = df.rename(columns={"NLL": f"iter{r}"})

            dfList.append(df)

        fullDf = pd.concat(dfList, axis=1)
        fullDf.set_index(indexes, inplace=True)
        # print(fullDf.head(10))
        # print(fullDf.tail(10))

        meanDF[f"CD-{k}"] = fullDf.mean(axis=1)  # mean
        meanDF[f"CD-{k} std"] = fullDf.std(axis=1)  # standard deviation
        meanDF[f"CD-{k} q1"] = fullDf.quantile(q=0.25, axis=1)  # first quartile
        meanDF[f"CD-{k} q3"] = fullDf.quantile(q=0.75, axis=1)  # third quartile

        meanDF[f"CD-{k}"].plot(ax=ax, linewidth=1, alpha=0.8)

        if errorType == "std":
            error = meanDF[f"CD-{k} std"].to_numpy()
            mean = meanDF[f"CD-{k}"].to_numpy()
            ax.fill_between(indexes, mean - error, mean + error, alpha=0.3)

        elif errorType == "quartile":
            errorPlus = meanDF[f"CD-{k} q3"].to_numpy()
            errorMinus = meanDF[f"CD-{k} q1"].to_numpy()
            ax.fill_between(indexes, errorMinus, errorPlus, alpha=0.3)

elif plotType == "neighbors":
    # NOTE: Only available for BAS so far
    for k in k_values:
        for v in v_values:
            dfList = []

            for r in range(repeat):
                if v == (dataSize * dataSize):
                    filename = "result/complete/" + inputBaseBAS["complete"].format(dataSize, hiddenUnits, k, lRate,
                                                                                    bSize, lim_iter, r)
                    # filename = "result/complete/nll_progress_complete_k{}-run{}.csv".format(k, r)
                else:
                    # filename = inputPath + "/nll_progress_bas{}_neighbors{}_k{}-run{}.csv".format(dataSize, v, k, r)
                    filename = inputPath + "/" + inputBaseBAS[plotType].format(dataSize, v, neighType, hiddenUnits, k,
                                                                               lRate, bSize, lim_iter, r)

                # df = pd.read_csv(filename, comment="#")  # index_col=0
                # if len(df.columns) == 2:
                df = pd.read_csv(filename, comment="#", index_col=0)
                df = df.astype(float)
                df = df.iloc[0:lim_iter + 1]
                df = df.rename(columns={"NLL": f"iter{r}"})

                dfList.append(df)

            fullDf = pd.concat(dfList, axis=1)
            # print(fullDf.head(10))
            # print(fullDf.tail(10))
            if periodoNLL != 1:
                fullDf.set_index(indexes, inplace=True)

            meanDF[f"CD-{k}, {v} neighbors in {neighType}"] = fullDf.mean(axis=1)  # mean
            meanDF[f"CD-{k}, {v} neighbors {neighType} - std"] = fullDf.std(axis=1)  # standard deviation
            meanDF[f"CD-{k}, {v} neighbors {neighType} - q1"] = fullDf.quantile(q=0.25, axis=1)  # first quartile
            meanDF[f"CD-{k}, {v} neighbors {neighType} - q3"] = fullDf.quantile(q=0.75, axis=1)  # third quartile

            meanDF[f"CD-{k}, {v} neighbors in {neighType}"].plot(ax=ax, linewidth=1, alpha=0.8)

            if errorType == "std":
                error = meanDF[f"CD-{k}, {v} neighbors {neighType} - std"].to_numpy()
                mean = meanDF[f"CD-{k}, {v} neighbors in {neighType}"].to_numpy()
                ax.fill_between(indexes, mean - error, mean + error, alpha=0.3)

            elif errorType == "quartile":
                errorPlus = meanDF[f"CD-{k}, {v} neighbors {neighType} - q3"].to_numpy()
                errorMinus = meanDF[f"CD-{k}, {v} neighbors {neighType} - q1"].to_numpy()
                ax.fill_between(indexes, errorMinus, errorPlus, alpha=0.3)

elif plotType == "BAScon":
    if dataType != "bas":
        raise ValueError("Only has BAScon for BAS dataset")

    for k in k_values:
        for v in versions:
            dfList = []

            for r in range(repeat):
                filename = inputPath + "/" + inputBaseBAS[plotType].format(dataSize, v, hiddenUnits, k, lRate, bSize,
                                                                           lim_iter, r)

                df = pd.read_csv(filename, comment="#", index_col=0)
                df = df.astype(float)
                df = df.iloc[0:lim_iter + 1]
                df = df.rename(columns={"NLL": f"iter{r}"})

                dfList.append(df)

            fullDf = pd.concat(dfList, axis=1)
            # if periodoNLL != 1:
            #     fullDf = fullDf.set_index(indexes)     # In this training indexes should already be set!

            meanDF[f"CD-{k}, Specialist v{v}"] = fullDf.mean(axis=1)  # mean
            meanDF[f"CD-{k}, Specialist v{v} - std"] = fullDf.std(axis=1)  # standard deviation
            meanDF[f"CD-{k}, Specialist v{v} - q1"] = fullDf.quantile(q=0.25, axis=1)  # first quartile
            meanDF[f"CD-{k}, Specialist v{v} - q3"] = fullDf.quantile(q=0.75, axis=1)  # third quartile

            meanDF[f"CD-{k}, Specialist v{v}"].plot(ax=ax, linewidth=1, alpha=0.8)

            if errorType == "std":
                error = meanDF[f"CD-{k}, Specialist v{v} - std"].to_numpy()
                mean = meanDF[f"CD-{k}, Specialist v{v}"].to_numpy()
                ax.fill_between(indexes, mean - error, mean + error, alpha=0.3)

            elif errorType == "quartile":
                errorPlus = meanDF[f"CD-{k}, Specialist v{v} - q3"].to_numpy()
                errorMinus = meanDF[f"CD-{k}, Specialist v{v} - q1"].to_numpy()
                ax.fill_between(indexes, errorMinus, errorPlus, alpha=0.3)

elif plotType.upper() == "SGD":
    for k in k_values:
        for p in p_val:
            dfListNLL = []
            degMeanH = np.zeros(shape=(lim_iter + 1, repeat))
            degMeanX = np.zeros(shape=(lim_iter + 1, repeat))
            degMaxH = np.zeros(shape=(lim_iter + 1, repeat))
            degMaxX = np.zeros(shape=(lim_iter + 1, repeat))
            degMinH = np.zeros(shape=(lim_iter + 1, repeat))
            degMinX = np.zeros(shape=(lim_iter + 1, repeat))

            for r in range(repeat):
                filenameNLL = ""

                if dataType == "bas":
                    filenameNLL = inputPath + "/" + inputBaseBAS[plotType].format(dataSize, p, hiddenUnits, k, lRate,
                                                                                  bSize, lim_iter, r)
                elif dataType == "mnist":
                    tmp = f"{plotType}-{p}"
                    filenameNLL = inputPath + "/" + inputBaseMNIST.format(tmp, hiddenUnits, k, lRate, bSize, lim_iter,
                                                                          r)

                # print(f"Opening file '{filenameNLL}'")

                df = pd.read_csv(filenameNLL, comment="#", index_col=0)
                df = df.astype(float)
                df = df.iloc[0:lim_iter + 1]
                df = df.rename(columns={"NLL": f"iter{r}"})
                dfListNLL.append(df)

            # NLL statistics
            fullDfNLL = pd.concat(dfListNLL, axis=1)
            if periodoNLL != 1:
                fullDfNLL.set_index(indexes, inplace=True)

            meanDF[f"CD-{k}, p = {p}"] = fullDfNLL.mean(axis=1)  # mean
            meanDF[f"CD-{k}, p = {p} - std"] = fullDfNLL.std(axis=1)  # standard deviation
            meanDF[f"CD-{k}, p = {p} - q1"] = fullDfNLL.quantile(q=0.25, axis=1)  # first quartile
            meanDF[f"CD-{k}, p = {p} - q3"] = fullDfNLL.quantile(q=0.75, axis=1)  # third quartile

            meanDF[f"CD-{k}, p = {p}"].plot(ax=ax, linewidth=1, alpha=0.8)

            if errorType == "std":
                error = meanDF[f"CD-{k}, p = {p} - std"].to_numpy()
                mean = meanDF[f"CD-{k}, p = {p}"].to_numpy()
                ax.fill_between(indexes, mean - error, mean + error, alpha=0.3)

            elif errorType == "quartile":
                errorPlus = meanDF[f"CD-{k}, p = {p} - q3"].to_numpy()
                errorMinus = meanDF[f"CD-{k}, p = {p} - q1"].to_numpy()
                ax.fill_between(indexes, errorMinus, errorPlus, alpha=0.3)


elif plotType in ["neighborsLine", "neighborsSpiral"]:
    for k in k_values:
        for v in v_values:
            dfList = []

            for r in range(repeat):
                tmp = f"{plotType}-{v}"
                filename = inputPath + "/" + inputBaseMNIST.format(tmp, hiddenUnits, k, lRate, bSize, lim_iter, r)

                df = pd.read_csv(filename, comment="#", index_col=0)
                df = df.astype(float)
                df = df.iloc[0:lim_iter + 1]
                df = df.rename(columns={"NLL": f"iter{r}"})

                dfList.append(df)

            fullDf = pd.concat(dfList, axis=1)

            if periodoNLL != 1:
                fullDf.set_index(indexes, inplace=True)

            tmp = plotType[9:].lower()

            meanDF[f"CD-{k}, {v} neighbors in {tmp}"] = fullDf.mean(axis=1)  # mean
            meanDF[f"CD-{k}, {v} neighbors {tmp} - std"] = fullDf.std(axis=1)  # standard deviation
            meanDF[f"CD-{k}, {v} neighbors {tmp} - q1"] = fullDf.quantile(q=0.25, axis=1)  # first quartile
            meanDF[f"CD-{k}, {v} neighbors {tmp} - q3"] = fullDf.quantile(q=0.75, axis=1)  # third quartile

            meanDF[f"CD-{k}, {v} neighbors in {tmp}"].plot(ax=ax, linewidth=1, alpha=0.8)

plt.legend()
plt.title("NLL evolution through RBM training")
plt.xlabel("Epoch")
plt.ylabel("Average NLL")
plt.grid(color="gray", linestyle=":", linewidth=.2)

if periodoNLL != 1:
    meanDF.set_index(indexes)

errorPrint = f"-{errorType}Err" if errorType else ""
neighPrint = f"_{neighType}" if plotType == "neighbors" else ""

if dataType == "bas":
    # Lower limit of NLL
    nSamples = 2 ** (dataSize + 1)
    limitante = - log(1.0 / nSamples)
    # print(f"NLL mínimo possível: {limitante}")
    plt.plot([0, lim_iter], [limitante, limitante], "r--")

plt.savefig(
    f"{outputPath}/meanNLL_{basename}_{plotType}{neighPrint}_H{hiddenUnits}_lr{lRate}_mBatch{bSize}_iter{lim_iter}-{repeat}rep{errorPrint}.pdf",
    transparent=True)
# plt.show()

meanDF.to_csv(
    f"{outputPath}/meanNLL_{basename}_{plotType}{neighPrint}_H{hiddenUnits}_lr{lRate}_mBatch{bSize}_iter{lim_iter}-{repeat}rep.csv")
