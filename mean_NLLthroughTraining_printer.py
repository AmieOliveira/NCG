#!/usr/bin/python3

import pandas as pd
import matplotlib.pyplot as plt
from math import log
import numpy as np


k_values = [100, 20, 10, 5, 2, 1]
v_values = [16, 12, 8, 6, 4]

size = "wide"  # "default", "wide"
lim_iter = int(6e3)
plotType = "neighbors"  # "complete", "neighbors"
repeat = 25

periodoNLL = 1

dataType = "bas"
dataSize = 4
basename = f"meanNll_{dataType}{dataSize}"
inputPath = "result/neighbors/"
outputPath = "result/neighbors/"

imputBase = { "complete": "nll_progress_complete_k{}-run{}.csv",
              "neighbors": "nll_progress_bas{}_neighbors{}_k{}-run{}.csv" }

figSize = {"default": (6.7, 5), "wide": (13, 5)}

fig, ax = plt.subplots(1, figsize=figSize[size])

meanDF = pd.DataFrame()
indexes = np.array(list(range(0, lim_iter + 1, periodoNLL)))  # NOTE: THIS IS HANDMADE AND SHOULD BE CHANGED ACCORDINGLY
# print(indexes)

if plotType == "complete":
    for k in k_values:
        dfList = []

        for r in range(repeat):
            filename = outputPath + "/" + imputBase[plotType].format(k, r)
            df = pd.read_csv(filename, comment="#")  # index_col=0
            df = df.astype(float)
            df = df.iloc[0:lim_iter + 1]
            df = df.rename(columns={"NLL": f"iter{r}"})

            dfList.append(df)

        fullDf = pd.concat(dfList, axis=1)
        fullDf[f"CD-{k}"] = fullDf.mean(axis=1)
        # print(fullDf.head(10))
        # print(fullDf.tail(10))
        fullDf.set_index(indexes)

        fullDf[f"CD-{k}"].plot(ax=ax, linewidth=1, alpha=0.8)
        meanDF[f"CD-{k}"] = fullDf[f"CD-{k}"]

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
            fullDf[f"CD-{k}, {v} neighbors"] = fullDf.mean(axis=1)
            # print(fullDf.head(10))
            # print(fullDf.tail(10))
            if periodoNLL != 1:
                fullDf.set_index(indexes)

            fullDf[f"CD-{k}, {v} neighbors"].plot(ax=ax, linewidth=1, alpha=0.8)
            meanDF[f"CD-{k}, {v} neighbors"] = fullDf[f"CD-{k}, {v} neighbors"]


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

plt.savefig(f"{outputPath}/{basename}_{plotType}.pdf", transparent=True)
meanDF.to_csv(f"{outputPath}/{basename}_{plotType}.csv")
#plt.show()

