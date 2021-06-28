#!/usr/bin/python3

import pandas as pd
import matplotlib.pyplot as plt
from math import log


k_values = [100, 20, 10, 5, 2, 1]

size = "wide"  # "default", "wide"
lim_iter = int(20e3)
plotType = "complete"  # "complete"
repeat = 25

dataType = "bas"
dataSize = 4
basename = f"meanNll_{dataType}{dataSize}"
inputPath = "."
outputPath = "."

inputBase = "nll_progress_complete_k{}-run{}.csv"
figSize = {"default": (6.7, 5), "wide": (13, 5)}

fig, ax = plt.subplots(1, figsize=figSize[size])

meanDF = pd.DataFrame()

for k in k_values:
    dfList = []

    for r in range(repeat):
        filename = outputPath + "/" + inputBase.format(k, r)
        df = pd.read_csv(filename, comment="#")
        df = df.astype(float)
        df = df.iloc[0:lim_iter]
        # df = df.rename(columns={"NLL": f"CD-{k}"})
        df = df.rename(columns={"NLL": f"iter{r}"})

        dfList.append(df)

    fullDf = pd.concat(dfList, axis=1)
    fullDf[f"CD-{k}"] = fullDf.mean(axis=1)
    # print(fullDf.head(10))

    fullDf[f"CD-{k}"].plot(ax=ax, linewidth=1, alpha=0.8)
    meanDF[f"CD-{k}"] = fullDf[f"CD-{k}"]

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

plt.savefig(f"{outputPath}/{basename}_{plotType}.pdf", transparent=True)
# plt.show()

meanDF.to_csv(f"{outputPath}/{basename}_{plotType}.csv")
