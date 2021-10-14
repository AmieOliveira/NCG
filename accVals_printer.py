#!/usr/bin/python3

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm


parser = argparse.ArgumentParser()
# Main arguments
plotting = parser.add_argument_group(title='Plot Settings',
                                     description="Paths and plots to be added")
plotting.add_argument("-t", "--plotType", type=str, nargs='+', required=True,
                    help="List all training types you want to add to the plots")

plotting.add_argument("-i", "--inputpath", type=str, default=".",
                    help="Input files' path")
plotting.add_argument("-o", "--outputpath", type=str, default=".",
                    help="Output files' path")

plotting.add_argument("-2", "--noFirst", action="store_true",
                      help="Activate this flag to remove first accuracy result (before training)")

# Training options
training = parser.add_argument_group(title='Training Settings',
                                     description="Training info, necessary in order to find correct files")
training.add_argument("-d", "--dataType", type=str, default="mnist",
                      help="So far can be either 'mnist' or 'basX', in which X is the size of the BAS dataset")
training.add_argument("-R", "--repeat", type=int, required=True,
                      help="Number of runs for each configuration")
training.add_argument("-H", "--hiddenNeurons", type=int, required=True,
                      help="Number of hidden neurons the RBM's have")
training.add_argument("-L", "--learningRate", type=float, required=True,
                      help="Learning rate utilized during training")
training.add_argument("-B", "--batchSize", type=int, required=True,
                      help="Size of the training mini-batchs")
training.add_argument("-I", "--iterations", type=int, required=True,
                      help="Number of training iterations (epochs)")



# Auxiliar lists
#    Contain possible values for k and p. More values can be added as needed
k_values = [1, 2, 5, 10, 20, 100]
p_values = [1, 0.5, 0.1]

cms = [cm.get_cmap("Blues"), cm.get_cmap("Oranges"), cm.get_cmap("Greens"), cm.get_cmap("Reds"), cm.get_cmap("Purples"),
       cm.get_cmap("copper"), cm.get_cmap("spring"), cm.get_cmap("Greys"), cm.get_cmap("summer"), cm.get_cmap("winter")]
# --------------

fileBase = "{}/acc_{}_{}_H{}_CD-{}_lr{}_mBatch{}_iter{}_withLabels_run{}.csv"

if __name__ == "__main__":
    args = parser.parse_args()

    list_types = args.plotType

    path = args.inputpath

    dataT = args.dataType
    H = args.hiddenNeurons
    lr = args.learningRate
    bSize = args.batchSize
    iters = args.iterations

    repeat = args.repeat

    # X = 784 if dataT == "mnist" else int(dataT[3:]) ** 2
    # nLabels = 10 if dataT == "mnist" else 2
    # size = X + nLabels

    for k in k_values:
        figTrain, axTrain = plt.subplots(nrows=1, ncols=1, figsize=(7, 5))
        figTrain.suptitle(f"Train Set Classification Performance for CD-{k}")

        figTest, axTest = plt.subplots(nrows=1, ncols=1, figsize=(7, 5))
        figTest.suptitle(f"Test Set Classification Performance for CD-{k}")

        hasKinstance = False
        figIdx = 0

        for pltT in list_types:

            if pltT == "sgd":
                for p in p_values:
                    pltTstr = f"sgd-{p}"

                    accTrainDF = pd.DataFrame()
                    accTestDF = pd.DataFrame()

                    hasPinstance = True

                    for r in range(repeat):
                        # print(f"Will try to open {fileBase.format(path, dataT, pltTstr, H, k, lr, bSize, iters, r)}")
                        try:
                            df = pd.read_csv( fileBase.format(path, dataT, pltTstr, H, k, lr, bSize, iters, r),
                                              comment="#", index_col=0 )
                        except FileNotFoundError:
                            hasPinstance = False
                            break

                        df = df.astype(float)
                        if args.noFirst:
                            df = df.iloc[1:iters + 1]
                        df = df.rename(columns={"Train": f"train iter {r}", "Test": f"test iter {r}"})

                        if accTrainDF.empty:
                            accTrainDF = pd.DataFrame(df[f"train iter {r}"])
                            accTestDF = pd.DataFrame(df[f"test iter {r}"])
                        else:
                            accTrainDF = accTrainDF.join(df[f"train iter {r}"])
                            accTestDF = accTestDF.join(df[f"test iter {r}"])

                        axTrain.scatter(df.index, df[f"train iter {r}"], marker="x", s=25, linewidth=0.9, zorder=5, alpha=0.8, color=cms[figIdx](0.1 + 0.8*r/repeat))
                        axTest.scatter(df.index, df[f"test iter {r}"], marker="x", s=25, linewidth=0.9, zorder=5, alpha=0.8, color=cms[figIdx](0.1 + 0.8*r/repeat))

                        # df.plot.scatter(y=f"train iter{r}", ax=axTrain, legend=False, alpha=0.8)
                        # df.plot.scatter(y=f"test iter{r}", ax=axTest, legend=False, alpha=0.8)

                    if not hasPinstance:
                        continue
                    else:
                        hasKinstance = True

                    accTrainDF.mean(axis=1).plot(ax=axTrain, label=f"p = {p}", legend=True)
                    accTestDF.mean(axis=1).plot(ax=axTest, label=f"p = {p}", legend=True)
                    # print(accTrainDF)

                    figIdx += 1


            else:
                accTrainDF = pd.DataFrame()
                accTestDF = pd.DataFrame()

                hasInstance = True

                for r in range(repeat):
                    try:
                        df = pd.read_csv( fileBase.format(path, dataT, pltT, H, k, lr, bSize, iters, r),
                                          comment="#", index_col=0 )
                    except FileNotFoundError:
                        hasInstance = False
                        break

                    df = df.astype(float)
                    if args.noFirst:
                        df = df.iloc[1:iters + 1]
                    df = df.rename(columns={"Train": f"train iter {r}", "Test": f"test iter {r}"})

                    if accTrainDF.empty:
                        accTrainDF = pd.DataFrame(df[f"train iter {r}"])
                        accTestDF = pd.DataFrame(df[f"test iter {r}"])
                    else:
                        accTrainDF = accTrainDF.join(df[f"train iter {r}"])
                        accTestDF = accTestDF.join(df[f"test iter {r}"])

                    axTrain.scatter(df.index, df[f"train iter {r}"], marker="*", s=25, linewidth=0.9, zorder=5, alpha=0.8, color=cms[figIdx](0.1 + 0.8*r/repeat))
                    axTest.scatter(df.index, df[f"test iter {r}"], marker="*", s=25, linewidth=0.9, zorder=5, alpha=0.8, color=cms[figIdx](0.1 + 0.8*r/repeat))

                if not hasInstance:
                    continue
                else:
                    hasKinstance = True

                accTrainDF.mean(axis=1).plot(ax=axTrain, label=pltT, legend=True, alpha=1)
                accTestDF.mean(axis=1).plot(ax=axTest, label=pltT, legend=True, alpha=1)
                # print(accTrainDF)

                figIdx += 1


        if hasKinstance:
            axTrain.grid(color="gray", linestyle=":", linewidth=.2)
            axTest.grid(color="gray", linestyle=":", linewidth=.2)

            axTrain.set_xlabel("Epoch")
            axTest.set_xlabel("Epoch")

            axTrain.set_ylabel("Accuracy")
            axTest.set_ylabel("Accuracy")

            print(f"Saving plots for k = {k}")
            # plt.show()
            figTrain.savefig(f"{args.outputpath}/meanScatter_acc-Train_{dataT}_H{H}_CD-{k}_lr{lr}_mBatch{bSize}_iter{iters}-{repeat}rep.pdf",
                             transparent=True)
            figTest.savefig(f"{args.outputpath}/meanScatter_acc-Test_{dataT}_H{H}_CD-{k}_lr{lr}_mBatch{bSize}_iter{iters}-{repeat}rep.pdf",
                            transparent=True)
