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

plotting.add_argument("-i", "--inputpath", type=str, nargs='+', default=".",
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

plotting.add_argument("-n", "--plotname", type=str, default="",
                      help="Modifier string to change the plot name (so as not to always overwrite)")

plotting.add_argument("-l", "--limit-iterations", type=int, default=-1,
                      help="Use if you want a plot with less epochs than the RBM was trained for")

# Training options
training = parser.add_argument_group(title='Training Settings',
                                     description="Training info, necessary in order to find correct files")
training.add_argument("-d", "--dataType", type=str, default="mnist",
                      help="So far can be either 'mnist', 'mushrooms', 'connect-4' or 'basX', in which X is the size of the BAS dataset")
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
k_values = [10, 1]  # [1, 2, 5, 10, 20, 100]  # TODO: Fix this! removed others for simplicity
p_values = [1, 0.5, 0.1]  # [1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.3, 0.1]
v_values = [63, 12, 56, 11, 392, 79]  # [700, 500, 400, 392, 250, 79, 50, 16]
# Reduzi do total pra deixar as imagens menos poluídas!

figSize = (4, 3)  # (7, 5)
fillTransp = 0.2  # 0.3

cms = [cm.get_cmap("Blues"), cm.get_cmap("Oranges"), cm.get_cmap("Greens"), cm.get_cmap("Reds"), cm.get_cmap("Purples"),
       cm.get_cmap("copper"), cm.get_cmap("spring"), cm.get_cmap("Greys"), cm.get_cmap("summer"), cm.get_cmap("winter")]
# --------------

param_values = p_values + v_values
expandFig = False


if __name__ == "__main__":
    args = parser.parse_args()

    list_types = args.plotType

    if len(list_types) > 2:
        expandFig = True
        figSize = (4, 4)

    paths = args.inputpath

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

    if dataT == "mnist":
        X = 784
    elif dataT == "mushrooms":
        X = 112
    elif dataT == "connect-4":
        X = 126
    elif dataT[:3] == "bas":
        X = int(dataT[3:])**2
    else:
        raise KeyError(f"Unrecognized data type '{dataT}'")
    
    lim_it = args.limit_iterations
    if lim_it <= 0:
        lim_it = iters


    for k in k_values:
        figTrain, axTrain = plt.subplots(nrows=1, ncols=1, figsize=figSize)
        # figTrain.suptitle(f"Train Set Classification Performance for CD-{k}")

        figTest, axTest = plt.subplots(nrows=1, ncols=1, figsize=figSize)
        # figTest.suptitle(f"Test Set Classification Performance for CD-{k}")

        hasKinstance = False

        for pltT in list_types:
            path = ""
            for d in paths:
                try:
                    ifTrainName = f"{d}/accMean-train_{dataT}_{pltT}_H{H}_lr{lr}_mBatch{bSize}_iter{iters}-{repeat}rep.csv"
                    inputTrainFile = pd.read_csv(ifTrainName, index_col=0)

                    path = d
                except FileNotFoundError:
                    pass

            if not path:
                print("ERROR:\tFiles not found. Check training parameters and which plots you want to add!")
                raise FileNotFoundError(f"No '{pltT}' file on any given directory!")

            ifTrainName = f"{path}/accMean-train_{dataT}_{pltT}_H{H}_lr{lr}_mBatch{bSize}_iter{iters}-{repeat}rep.csv"
            ifTestName = f"{path}/accMean-test_{dataT}_{pltT}_H{H}_lr{lr}_mBatch{bSize}_iter{iters}-{repeat}rep.csv"

            inputTrainFile = pd.read_csv(ifTrainName, index_col=0)
            inputTestFile = pd.read_csv(ifTestName, index_col=0)

            if args.noFirst:
                inputTrainFile = inputTrainFile.iloc[1:lim_it + 1]
                inputTestFile = inputTestFile.iloc[1:lim_it + 1]
            else:
                inputTrainFile = inputTrainFile.iloc[:lim_it + 1]
                inputTestFile = inputTestFile.iloc[:lim_it + 1]

            if pltT in ["sgd", "random", "neighborsLine", "ncgh"]:
                for p in param_values:
                    try:
                        # inputTrainFile[f"CD-{k} p = {p}"].plot(ax=axTrain, linewidth=1, alpha=0.8, legend=True)
                        # inputTestFile[f"CD-{k} p = {p}"].plot(ax=axTest, linewidth=1, alpha=0.8, legend=True)

                        legendStr = ""
                        if pltT == "sgd":
                            legendStr = f"NCG, p = {p}"
                        elif pltT == "random":
                            legendStr = f"Random, d = {p}"
                        elif pltT == "neighborsLine":
                            legendStr = f"Line, v/X = {p/X:.1f}"
                        elif pltT == "ncgh":
                            legendStr = f"'NCG-H', p = {p}"

                        tmp = inputTrainFile.rename(columns={f"CD-{k} p = {p}": legendStr})[legendStr]
                        tmp.plot(ax=axTrain, linewidth=1, alpha=0.8, legend=True)

                        tmp = inputTestFile.rename(columns={f"CD-{k} p = {p}": legendStr})[legendStr]
                        tmp.plot(ax=axTest, linewidth=1, alpha=0.8, legend=True)
                    except KeyError:
                        continue

                    hasKinstance = True

                    if args.printerrors:
                        indexes = inputTrainFile.index

                        errorPlus = inputTestFile[f"CD-{k} p = {p} - q3"].to_numpy()
                        errorMinus = inputTestFile[f"CD-{k} p = {p} - q1"].to_numpy()
                        axTest.fill_between(indexes, errorMinus, errorPlus, alpha=0.3)

                        errorPlus = inputTrainFile[f"CD-{k} p = {p} - q3"].to_numpy()
                        errorMinus = inputTrainFile[f"CD-{k} p = {p} - q1"].to_numpy()
                        axTrain.fill_between(indexes, errorMinus, errorPlus, alpha=0.3)

                    elif args.stdErr:
                        indexes = inputTrainFile.index

                        error = inputTestFile[f"CD-{k} p = {p} - std"].to_numpy()
                        mean = inputTestFile[f"CD-{k} p = {p}"].to_numpy()
                        axTest.fill_between(indexes, mean - error, mean + error, alpha=0.3)

                        error = inputTrainFile[f"CD-{k} p = {p} - std"].to_numpy()
                        mean = inputTrainFile[f"CD-{k} p = {p}"].to_numpy()
                        axTrain.fill_between(indexes, mean - error, mean + error, alpha=0.3)

                    elif args.minmaxErr:
                        indexes = inputTrainFile.index

                        maxV = inputTrainFile[f"CD-{k} p = {p} - Max"].to_numpy()
                        minV = inputTrainFile[f"CD-{k} p = {p} - Min"].to_numpy()
                        axTrain.fill_between(indexes, minV, maxV, alpha=0.3)

                        maxV = inputTestFile[f"CD-{k} p = {p} - Max"].to_numpy()
                        minV = inputTestFile[f"CD-{k} p = {p} - Min"].to_numpy()
                        axTest.fill_between(indexes, minV, maxV, alpha=0.3)


            else:
                completeLegend = "Dense Network"  # "Traditional Network"
                try:
                    # inputTrainFile[f"CD-{k} {pltT}"].plot(ax=axTrain, linewidth=1, alpha=0.8, legend=True)
                    # inputTestFile[f"CD-{k} {pltT}"].plot(ax=axTest, linewidth=1, alpha=0.8, legend=True)

                    tmp = inputTrainFile.rename(columns={f"CD-{k} {pltT}": completeLegend})[completeLegend]
                    tmp.plot(ax=axTrain, linewidth=1, alpha=0.8, legend=True)

                    tmp = inputTestFile.rename(columns={f"CD-{k} {pltT}": completeLegend})[completeLegend]
                    tmp.plot(ax=axTest, linewidth=1, alpha=0.8, legend=True)
                except KeyError:
                    continue

                hasKinstance = True

                if args.printerrors:
                    indexes = inputTrainFile.index

                    errorPlus = inputTestFile[f"CD-{k} {pltT} - q3"].to_numpy()
                    errorMinus = inputTestFile[f"CD-{k} {pltT} - q1"].to_numpy()
                    axTest.fill_between(indexes, errorMinus, errorPlus, alpha=fillTransp)

                    errorPlus = inputTrainFile[f"CD-{k} {pltT} - q3"].to_numpy()
                    errorMinus = inputTrainFile[f"CD-{k} {pltT} - q1"].to_numpy()
                    axTrain.fill_between(indexes, errorMinus, errorPlus, alpha=fillTransp)

                elif args.stdErr:
                    indexes = inputTrainFile.index

                    error = inputTestFile[f"CD-{k} {pltT} - std"].to_numpy()
                    mean = inputTestFile[f"CD-{k} {pltT}"].to_numpy()
                    axTest.fill_between(indexes, mean - error, mean + error, alpha=fillTransp)

                    error = inputTrainFile[f"CD-{k} {pltT} - std"].to_numpy()
                    mean = inputTrainFile[f"CD-{k} {pltT}"].to_numpy()
                    axTrain.fill_between(indexes, mean - error, mean + error, alpha=fillTransp)

                elif args.minmaxErr:
                    indexes = inputTrainFile.index

                    maxV = inputTrainFile[f"CD-{k} {pltT} - Max"].to_numpy()
                    minV = inputTrainFile[f"CD-{k} {pltT} - Min"].to_numpy()
                    axTest.fill_between(indexes, minV, maxV, alpha=fillTransp)

                    maxV = inputTestFile[f"CD-{k} {pltT} - Max"].to_numpy()
                    minV = inputTestFile[f"CD-{k} {pltT} - Min"].to_numpy()
                    axTrain.fill_between(indexes, minV, maxV, alpha=fillTransp)


        if hasKinstance:
            axTrain.grid(color="gray", linestyle=":", linewidth=.2)
            axTest.grid(color="gray", linestyle=":", linewidth=.2)

            axTrain.set_xlabel("Epoch")
            axTest.set_xlabel("Epoch")

            axTrain.set_ylabel("Accuracy (%)")
            axTest.set_ylabel("Accuracy (%)")

            modifStr = args.plotname
            if modifStr:
                modifStr = "_" + modifStr

            printFirst = ""
            if args.noFirst:
                printFirst = "-2"

            # plt.show()
            print(f"Saving plots for k = {k}")
            plt.figure(figTrain)
            if expandFig:
                plt.legend(bbox_to_anchor=(0.5, 1.03), loc="lower center", borderaxespad=0, ncol=2, prop={'size': 9})
            plt.tight_layout()
            figTrain.savefig(
                f"{args.outputpath}/mean_acc-Train_{dataT}{modifStr}_H{H}_CD-{k}_lr{lr}_mBatch{bSize}_iter{lim_it}-{repeat}rep{errorPrint}{printFirst}.pdf",
                transparent=True)

            plt.figure(figTest)
            if expandFig:
                plt.legend(bbox_to_anchor=(0.5, 1.03), loc="lower center", borderaxespad=0, ncol=2, prop={'size': 9})
            plt.tight_layout()
            figTest.savefig(
                f"{args.outputpath}/mean_acc-Test_{dataT}{modifStr}_H{H}_CD-{k}_lr{lr}_mBatch{bSize}_iter{lim_it}-{repeat}rep{errorPrint}{printFirst}.pdf",
                transparent=True)

