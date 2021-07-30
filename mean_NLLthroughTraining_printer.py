#!/usr/bin/python3

import pandas as pd
import matplotlib.pyplot as plt
from math import log
import numpy as np


k_values = [100, 20, 10, 5, 2, 1]
v_values = [16, 14, 12, 10, 8, 6, 4]
versions = [2, 3, 4]  # [1, 2]
p_val = [1, 0.75, 0.5, 0.25]

size = "default"  # "default", "wide"
lim_iter = int(20e3)
plotType = "SGD"  # "complete", "neighbors", "BAScon", "SGD"
neighType = "line"
errorType = None  # None, "std", "quartile"
repeat = 25

periodoNLL = 1

dataType = "bas"
dataSize = 4
lRate = 0.01
basename = f"meanNll_{dataType}{dataSize}"
inputPath = f"result/{plotType}"
outputPath = inputPath

imputBase = { "complete":   "nll_progress_bas{}_complete_k{}-run{}.csv",
              "neighbors":  "nll_progress_bas{}_neighbors{}_{}_k{}-run{}.csv",
              "BAScon":     "nll_progress_bas{}_BASconV{}_k{}-run{}.csv",
              "SGD":        ["nll_bas{}_SGD_CD-{}_lr{}_p{}_run{}.csv",
                             "connectivity_bas{}_SGD_CD-{}_lr{}_p{}_run{}.csv"] }

figSize = {"default": (6.7, 5), "wide": (13, 5)}

fig, ax = plt.subplots(1, figsize=figSize[size])

meanDF = pd.DataFrame()
connectivityDF = pd.DataFrame()
indexes = np.array(list(range(0, lim_iter + 1, periodoNLL)))  # NOTE: THIS IS HANDMADE AND SHOULD BE CHANGED ACCORDINGLY
# print(indexes)


def read_degree_stats(name):
    totsize = dataSize**2

    tmpdf = pd.read_csv(name, comment="#", index_col=0, header=None)
    tmpdf = tmpdf.astype(int)
    arr = tmpdf.to_numpy()
    arr = np.reshape(arr, (arr.shape[0], totsize, totsize))

    iterations = len(tmpdf.index)

    gh_max = np.zeros(iterations)
    gh_med = np.zeros(iterations)
    gh_min = np.zeros(iterations)

    gx_max = np.zeros(iterations)
    gx_med = np.zeros(iterations)
    gx_min = np.zeros(iterations)

    for i in range(iterations):
        gh = [sum(arr[i, r, :]) for r in range(totsize)]
        gh_med[i] = sum(gh)/len(gh)
        gh_min[i] = min(gh)
        gh_max[i] = max(gh)

        gx = [sum(arr[i, :, c]) for c in range(totsize)]
        gx_med[i] = sum(gx)/len(gx)
        gx_min[i] = min(gx)
        gx_max[i] = max(gx)

    return gh_med, gh_max, gh_min, gx_med, gx_max, gx_min



if plotType == "complete":
    for k in k_values:
        dfList = []

        for r in range(repeat):
            try:
                filename = inputPath + "/" + imputBase[plotType].format(dataSize, k, r)
                df = pd.read_csv(filename, comment="#", index_col=0)
            except FileNotFoundError:
                filename = inputPath + "/" + "nll_progress_complete_k{}-run{}.csv".format(k, r)
                df = pd.read_csv(filename, comment="#")  # index_col=0

            df = df.astype(float)
            df = df.iloc[0:lim_iter + 1]
            df = df.rename(columns={"NLL": f"iter{r}"})

            dfList.append(df)

        fullDf = pd.concat(dfList, axis=1)
        fullDf.set_index(indexes, inplace=True)
        # print(fullDf.head(10))
        # print(fullDf.tail(10))

        meanDF[f"CD-{k}"] = fullDf.mean(axis=1)                     # mean
        meanDF[f"CD-{k} std"] = fullDf.std(axis=1)                  # standard deviation
        meanDF[f"CD-{k} q1"] = fullDf.quantile(q=0.25, axis=1)      # first quartile
        meanDF[f"CD-{k} q3"] = fullDf.quantile(q=0.75, axis=1)      # third quartile

        meanDF[f"CD-{k}"].plot(ax=ax, linewidth=1, alpha=0.8)

        if errorType == "std":
            error = meanDF[f"CD-{k} std"].to_numpy()
            mean = meanDF[f"CD-{k}"].to_numpy()
            ax.fill_between(indexes, mean-error, mean+error, alpha=0.3)

        elif errorType == "quartile":
            errorPlus = meanDF[f"CD-{k} q3"].to_numpy()
            errorMinus = meanDF[f"CD-{k} q1"].to_numpy()
            ax.fill_between(indexes, errorMinus, errorPlus, alpha=0.3)

elif plotType == "neighbors":
    for k in k_values:
        for v in v_values:
            dfList = []

            for r in range(repeat):
                if v == (dataSize * dataSize):
                    filename = "result/complete/" + imputBase["complete"].format(dataSize, k, r)
                    # filename = "result/complete/nll_progress_complete_k{}-run{}.csv".format(k, r)
                else:
                    # filename = inputPath + "/nll_progress_bas{}_neighbors{}_k{}-run{}.csv".format(dataSize, v, k, r)
                    filename = inputPath + "/" + imputBase[plotType].format(dataSize, v, neighType, k, r)

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
    for k in k_values:
        for v in versions:
            dfList = []

            for r in range(repeat):
                filename = inputPath + "/" + imputBase[plotType].format(dataSize, v, k, r)

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

elif plotType == "SGD":
    for k in k_values:
        for p in p_val:
            dfListNLL = []
            degMean = np.zeros(shape=(lim_iter+1, repeat))
            degMaxH = np.zeros(shape=(lim_iter+1, repeat))
            degMaxX = np.zeros(shape=(lim_iter+1, repeat))
            degMinH = np.zeros(shape=(lim_iter+1, repeat))
            degMinX = np.zeros(shape=(lim_iter+1, repeat))

            for r in range(repeat):
                filenameNLL = inputPath + "/" + imputBase[plotType][0].format(dataSize, k, lRate, p, r)
                df = pd.read_csv(filenameNLL, comment="#", index_col=0)
                df = df.astype(float)
                df = df.iloc[0:lim_iter + 1]
                df = df.rename(columns={"NLL": f"iter{r}"})
                dfListNLL.append(df)

                filenameCon = inputPath + "/" + imputBase[plotType][1].format(dataSize, k, lRate, p, r)

                if (r == 0) and (periodoNLL != 1):
                    gh_med, gh_max, gh_min, _, gx_max, gx_min = read_degree_stats(filenameCon)

                    iterations = len(gh_med.index)
                    degMean = np.zeros(shape=(iterations, repeat))
                    degMaxH = np.zeros(shape=(iterations, repeat))
                    degMaxX = np.zeros(shape=(iterations, repeat))
                    degMinH = np.zeros(shape=(iterations, repeat))
                    degMinX = np.zeros(shape=(iterations, repeat))

                    degMean[:, 0] = gh_med
                    degMaxH[:, 0] = gh_max
                    degMaxX[:, 0] = gx_max
                    degMinH[:, 0] = gh_min
                    degMinX[:, 0] = gx_min

                else:
                    degMean[:, r], degMaxH[:, r], degMinH[:, r], \
                    _, degMaxX[:, r], degMinX[:, r] = read_degree_stats(filenameCon)

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


            # Connectivity statistics
            connectivityDF[f"Mean, CD-{k}, p = {p}"] = degMean.mean(axis=1)
            connectivityDF[f"Mean, CD-{k}, p = {p} - std"] = degMean.std(axis=1)
            connectivityDF[f"Mean, CD-{k}, p = {p} - q1"] = np.quantile(degMean, q=0.25,axis=1)
            connectivityDF[f"Mean, CD-{k}, p = {p} - q3"] = np.quantile(degMean, q=0.75,axis=1)

            connectivityDF[f"Maximum in X, CD-{k}, p = {p}"] = degMaxX.mean(axis=1)
            connectivityDF[f"Maximum in X, CD-{k}, p = {p} - std"] = degMaxX.std(axis=1)
            connectivityDF[f"Maximum in X, CD-{k}, p = {p} - q1"] = np.quantile(degMaxX, q=0.25,axis=1)
            connectivityDF[f"Maximum in X, CD-{k}, p = {p} - q3"] = np.quantile(degMaxX, q=0.75,axis=1)

            connectivityDF[f"Maximum in H, CD-{k}, p = {p}"] = degMaxH.mean(axis=1)
            connectivityDF[f"Maximum in H, CD-{k}, p = {p} - std"] = degMaxH.std(axis=1)
            connectivityDF[f"Maximum in H, CD-{k}, p = {p} - q1"] = np.quantile(degMaxH, q=0.25,axis=1)
            connectivityDF[f"Maximum in H, CD-{k}, p = {p} - q3"] = np.quantile(degMaxH, q=0.75,axis=1)

            connectivityDF[f"Minimum in X, CD-{k}, p = {p}"] = degMinX.mean(axis=1)
            connectivityDF[f"Minimum in X, CD-{k}, p = {p} - std"] = degMinX.std(axis=1)
            connectivityDF[f"Minimum in X, CD-{k}, p = {p} - q1"] = np.quantile(degMinX, q=0.25,axis=1)
            connectivityDF[f"Minimum in X, CD-{k}, p = {p} - q3"] = np.quantile(degMinX, q=0.75,axis=1)

            connectivityDF[f"Minimum in H, CD-{k}, p = {p}"] = degMinH.mean(axis=1)
            connectivityDF[f"Minimum in H, CD-{k}, p = {p} - std"] = degMinH.std(axis=1)
            connectivityDF[f"Minimum in H, CD-{k}, p = {p} - q1"] = np.quantile(degMinH, q=0.25,axis=1)
            connectivityDF[f"Minimum in H, CD-{k}, p = {p} - q3"] = np.quantile(degMinH, q=0.75,axis=1)

    connectivityDF.set_index(indexes, inplace=True)


plt.legend()
plt.title("NLL evolution through RBM training")
plt.xlabel("Iteration")
plt.ylabel("Average NLL")
plt.grid(color="gray", linestyle=":", linewidth=.2)

# Lower limit of NLL
nSamples = 2**(dataSize+1)
limitante = - log(1.0/nSamples)
# print(f"NLL mínimo possível: {limitante}")
plt.plot([0, lim_iter], [limitante, limitante], "r--")
if periodoNLL != 1:
    meanDF.set_index(indexes)


errorPrint = f"-{errorType}Err" if errorType else ""
neighPrint = f"_{neighType}" if plotType == "neighbors" else ""

plt.savefig(f"{outputPath}/{basename}_{plotType}{neighPrint}_lr{lRate}-{repeat}rep{errorPrint}.pdf", transparent=True)
# plt.show()

meanDF.to_csv(f"{outputPath}/{basename}_{plotType}{neighPrint}_lr{lRate}-{repeat}rep.csv")

if plotType == "SGD":
    connectivityDF.to_csv(f"{outputPath}/meanDeg_{dataType}{dataSize}_{plotType}{neighPrint}_lr{lRate}-{repeat}rep.csv")
