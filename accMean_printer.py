#!/usr/bin/python3

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

parser = argparse.ArgumentParser()
# Main arguments
plotting = parser.add_argument_group(title='Plot Settings',
                                     description="Files' paths; plots and erros to be added")
plotting.add_argument("-t", "--plotType", type=str, nargs='+', required=True,
                    help="List all training types you want to add to the plots")

plotting.add_argument("-i", "--inputpath", type=str, default=".",
                    help="Input files' path")
plotting.add_argument("-o", "--outputpath", type=str, default=".",
                    help="Output files' path")

plotting.add_argument("-e", "--printerrors", action="store_true",
                      help="Add errorbars in plot. Default is quartile errors, if you wish to use "
                           "standard deviation error specification, use 'stderr' argument")
plotting.add_argument("--stdErr", action="store_true",
                      help="Add errorbars in plot, using standard deviation error. If you wish to "
                           "use quartile error specify default error argument '-e'")
plotting.add_argument("--minmaxErr", action="store_true",
                      help="Add errorbars in plot, using maximum and minimum values as erros. If "
                           "you wish to use quartile error specify default error argument '-e'")

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

    errorPrint = ""
    if args.printerrors:
        errorPrint = "-quartileErr"
    elif args.stdErr:
        errorPrint = "-stdErr"
    elif args.minmaxErr:
        errorPrint = "-minMaxErr"

    for k in k_values:
        figTrain, axTrain = plt.subplots(nrows=1, ncols=1, figsize=(7, 5))
        figTrain.suptitle(f"Train Set Classification Performance for CD-{k}")

        figTest, axTest = plt.subplots(nrows=1, ncols=1, figsize=(7, 5))
        figTest.suptitle(f"Test Set Classification Performance for CD-{k}")

        hasKinstance = False

        for pltT in list_types:
            ifTrainName = f"{path}/accMean-train_{dataT}_{pltT}_H{H}_lr{lr}_mBatch{bSize}_iter{iters}.csv"
            ifTestName = f"{path}/accMean-test_{dataT}_{pltT}_H{H}_lr{lr}_mBatch{bSize}_iter{iters}.csv"

            try:
                inputTrainFile = pd.read_csv(ifTrainName, index_col=0)
                inputTestFile = pd.read_csv(ifTestName, index_col=0)
            except FileNotFoundError:
                print("Files not found. Check training parameters and which plots you wnat to add!")

            if args.noFirst:
                inputTrainFile = inputTrainFile.iloc[1:iters + 1]
                inputTestFile = inputTestFile.iloc[1:iters + 1]

            if pltT == "sgd":
                for p in p_values:
                    try:
                        inputTrainFile[f"CD-{k} p = {p}"].plot(ax=axTrain, linewidth=1, alpha=0.8)
                        inputTestFile[f"CD-{k} p = {p}"].plot(ax=axTest, linewidth=1, alpha=0.8)
                    except KeyError:
                        continue

                    hasKinstance = True

                    if args.printerrors:
                        indexes = inputTrainFile.index

                        errorPlus = inputTestFile[f"CD-{k} p = {p} - q3"].to_numpy()
                        errorMinus = inputTestFile[f"CD-{k} p = {p} - q1"].to_numpy()
                        axTrain.fill_between(indexes, errorMinus, errorPlus, alpha=0.3)

                        errorPlus = inputTrainFile[f"CD-{k} p = {p} - q3"].to_numpy()
                        errorMinus = inputTrainFile[f"CD-{k} p = {p} - q1"].to_numpy()
                        axTest.fill_between(indexes, errorMinus, errorPlus, alpha=0.3)

                    elif args.stdErr:
                        indexes = inputTrainFile.index

                        error = inputTestFile[f"CD-{k} p = {p} - std"].to_numpy()
                        mean = inputTestFile[f"CD-{k} p = {p}"].to_numpy()
                        axTrain.fill_between(indexes, mean - error, mean + error, alpha=0.3)

                        error = inputTrainFile[f"CD-{k} p = {p} - std"].to_numpy()
                        mean = inputTrainFile[f"CD-{k} p = {p}"].to_numpy()
                        axTest.fill_between(indexes, mean - error, mean + error, alpha=0.3)

                    elif args.minmaxErr:
                        indexes = inputTrainFile.index

                        maxV = inputTrainFile[f"CD-{k} p = {p} - Max"].to_numpy()
                        minV = inputTrainFile[f"CD-{k} p = {p} - Min"].to_numpy()
                        axTrain.fill_between(indexes, minV, maxV, alpha=0.3)

                        maxV = inputTestFile[f"CD-{k} p = {p} - Max"].to_numpy()
                        minV = inputTestFile[f"CD-{k} p = {p} - Min"].to_numpy()
                        axTest.fill_between(indexes, minV, maxV, alpha=0.3)


            else:
                try:
                    inputTrainFile[f"CD-{k} {pltT}"].plot(ax=axTrain, linewidth=1, alpha=0.8)
                    inputTestFile[f"CD-{k} {pltT}"].plot(ax=axTest, linewidth=1, alpha=0.8)
                except KeyError:
                    continue

                hasKinstance = True

                if args.printerrors:
                    indexes = inputTrainFile.index

                    errorPlus = inputTestFile[f"CD-{k} {pltT} - q3"].to_numpy()
                    errorMinus = inputTestFile[f"CD-{k} {pltT} - q1"].to_numpy()
                    axTrain.fill_between(indexes, errorMinus, errorPlus, alpha=0.3)

                    errorPlus = inputTrainFile[f"CD-{k} {pltT} - q3"].to_numpy()
                    errorMinus = inputTrainFile[f"CD-{k} {pltT} - q1"].to_numpy()
                    axTest.fill_between(indexes, errorMinus, errorPlus, alpha=0.3)

                elif args.stdErr:
                    indexes = inputTrainFile.index

                    error = inputTestFile[f"CD-{k} {pltT} - std"].to_numpy()
                    mean = inputTestFile[f"CD-{k} {pltT}"].to_numpy()
                    axTrain.fill_between(indexes, mean - error, mean + error, alpha=0.3)

                    error = inputTrainFile[f"CD-{k} {pltT} - std"].to_numpy()
                    mean = inputTrainFile[f"CD-{k} {pltT}"].to_numpy()
                    axTest.fill_between(indexes, mean - error, mean + error, alpha=0.3)

                elif args.minmaxErr:
                    indexes = inputTrainFile.index

                    maxV = inputTrainFile[f"CD-{k} {pltT} - Max"].to_numpy()
                    minV = inputTrainFile[f"CD-{k} {pltT} - Min"].to_numpy()
                    axTrain.fill_between(indexes, minV, maxV, alpha=0.3)

                    maxV = inputTestFile[f"CD-{k} {pltT} - Max"].to_numpy()
                    minV = inputTestFile[f"CD-{k} {pltT} - Min"].to_numpy()
                    axTest.fill_between(indexes, minV, maxV, alpha=0.3)


        if hasKinstance:
            axTrain.grid(color="gray", linestyle=":", linewidth=.2)
            axTest.grid(color="gray", linestyle=":", linewidth=.2)

            axTrain.xlabel("Epoch")
            axTest.xlabel("Epoch")

            axTrain.ylabel("Accuracy")
            axTest.ylabel("Accuracy")

            # plt.show()
            print(f"Saving plots for k = {k}")
            figTrain.savefig(
                f"{args.outputpath}/mean_acc-Train_{dataT}_H{H}_CD-{k}_lr{lr}_mBatch{bSize}_iter{iters}-{repeat}rep{errorPrint}.pdf",
                transparent=True)
            figTest.savefig(
                f"{args.outputpath}/mean_acc-Test_{dataT}_H{H}_CD-{k}_lr{lr}_mBatch{bSize}_iter{iters}-{repeat}rep{errorPrint}.pdf",
                transparent=True)

