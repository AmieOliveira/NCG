#!/usr/bin/python3

import pandas as pd
import matplotlib.pyplot as plt
from math import log
import numpy as np


k_values = [1]  # [100, 20, 10, 5, 2, 1]
v_values = [13]  # [16, 12, 8, 6, 4]
versions = [1, 2]

size = "default"  # "default", "wide"
lim_iter = int(1e3)
plotType = "complete"  # "complete", "neighbors", "BAScon"
errorType = "quartile"  # None, "std", "quartile"
repeat = 2

periodoNLL = 1

dataType = "bas"
dataSize = 3
basename = f"meanNll_{dataType}{dataSize}"
inputPath = f"Training Outputs/Teste Servidor"
outputPath = f"Training Outputs/Teste Servidor"

imputBase = { "complete":   "nll_progress_bas{}_complete_k{}-run{}.csv",
              "neighbors":  "nll_progress_bas{}_neighbors{}_k{}-run{}.csv",
              "BAScon":     "nll_progress_bas{}_BASconV{}_k{}-run{}.csv" }

figSize = {"default": (6.7, 5), "wide": (13, 5)}

fig, ax = plt.subplots(1, figsize=figSize[size])

meanDF = pd.DataFrame()
indexes = np.array(list(range(0, lim_iter + 1, periodoNLL)))  # NOTE: THIS IS HANDMADE AND SHOULD BE CHANGED ACCORDINGLY
# print(indexes)

if plotType == "complete":
    for k in k_values:
        dfList = []

        for r in range(repeat):
            try:
                filename = outputPath + "/" + imputBase[plotType].format(dataSize, k, r)
                df = pd.read_csv(filename, comment="#", index_col=0)
            except FileNotFoundError:
                filename = outputPath + "/" + "nll_progress_complete_k{}-run{}.csv".format(k, r)
                df = pd.read_csv(filename, comment="#")  # index_col=0

            df = df.astype(float)
            df = df.iloc[0:lim_iter + 1]
            df = df.rename(columns={"NLL": f"iter{r}"})

            dfList.append(df)

        fullDf = pd.concat(dfList, axis=1)
        # print(fullDf.head(10))
        # print(fullDf.tail(10))
        fullDf.set_index(indexes)

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
                    filename = "result/complete/" + imputBase["complete"].format(k, r)
                else:
                    filename = outputPath + "/" + imputBase[plotType].format(dataSize, v, k, r)

                df = pd.read_csv(filename, comment="#")  # index_col=0
                if len(df.columns) == 2:
                    df = pd.read_csv(filename, comment="#", index_col=0)
                df = df.astype(float)
                df = df.iloc[0:lim_iter + 1]
                df = df.rename(columns={"NLL": f"iter{r}"})

                dfList.append(df)

            fullDf = pd.concat(dfList, axis=1)
            # print(fullDf.head(10))
            # print(fullDf.tail(10))
            if periodoNLL != 1:
                fullDf.set_index(indexes)

            meanDF[f"CD-{k}, {v} neighbors"] = fullDf.mean(axis=1)  # mean
            meanDF[f"CD-{k}, {v} neighbors - std"] = fullDf.std(axis=1)  # standard deviation
            meanDF[f"CD-{k}, {v} neighbors - q1"] = fullDf.quantile(q=0.25, axis=1)  # first quartile
            meanDF[f"CD-{k}, {v} neighbors - q3"] = fullDf.quantile(q=0.75, axis=1)  # third quartile

            meanDF[f"CD-{k}, {v} neighbors"].plot(ax=ax, linewidth=1, alpha=0.8)

            if errorType == "std":
                error = meanDF[f"CD-{k}, {v} neighbors - std"].to_numpy()
                mean = meanDF[f"CD-{k}, {v} neighbors"].to_numpy()
                ax.fill_between(indexes, mean - error, mean + error, alpha=0.3)

            elif errorType == "quartile":
                errorPlus = meanDF[f"CD-{k}, {v} neighbors - q3"].to_numpy()
                errorMinus = meanDF[f"CD-{k}, {v} neighbors - q1"].to_numpy()
                ax.fill_between(indexes, errorMinus, errorPlus, alpha=0.3)

elif plotType == "BAScon":
    for k in k_values:
        for v in versions:
            dfList = []

            for r in range(repeat):
                filename = outputPath + "/" + imputBase[plotType].format(dataSize, v, k, r)

                df = pd.read_csv(filename, comment="#", index_col=0)
                df = df.astype(float)
                df = df.iloc[0:lim_iter + 1]
                df = df.rename(columns={"NLL": f"iter{r}"})

                dfList.append(df)

            fullDf = pd.concat(dfList, axis=1)
            # if periodoNLL != 1:
            #     fullDf.set_index(indexes)     # In this training indexes should already be set!

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

plt.savefig(f"{outputPath}/{basename}_{plotType}-{repeat}rep{errorPrint}.pdf", transparent=True)
meanDF.to_csv(f"{outputPath}/{basename}_{plotType}-{repeat}rep.csv")
# plt.show()
