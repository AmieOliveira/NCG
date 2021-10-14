#!/usr/bin/python3

import argparse
import pandas as pd
import numpy as np


parser = argparse.ArgumentParser()
# Main arguments
# Main arguments
plotting = parser.add_argument_group(title='Plot Settings',
                                     description="Paths and plots to be added")
plotting.add_argument("-t", "--plotType", type=str, nargs='+', required=True,
                    help="List all training types you want to add to the plots")

plotting.add_argument("-i", "--inputpath", type=str, default=".",
                    help="Input files' path")
plotting.add_argument("-o", "--outputpath", type=str, default=".",
                    help="Output files' path")

# Training options
training = parser.add_argument_group(title='train_settings',
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

    for pltT in list_types:
        outputTrainFileName = f"{args.outputpath}/accMean-train_{dataT}_{pltT}_H{H}_lr{lr}_mBatch{bSize}_iter{iters}.csv"
        outputTestFileName =  f"{args.outputpath}/accMean-test_{dataT}_{pltT}_H{H}_lr{lr}_mBatch{bSize}_iter{iters}.csv"

        accTrainDF = pd.DataFrame()
        accTestDF = pd.DataFrame()

        for k in k_values:

            if pltT == "sgd":
                for p in p_values:
                    pltTstr = f"sgd-{p}"

                    tmpTrain = pd.DataFrame()
                    tmpTest = pd.DataFrame()

                    hasPinstance = True

                    for r in range(repeat):
                        # print(f"Will try to open {fileBase.format(path, dataT, pltTstr, H, k, lr, bSize, iters, r)}")
                        try:
                            df = pd.read_csv(fileBase.format(path, dataT, pltTstr, H, k, lr, bSize, iters, r),
                                             comment="#", index_col=0)
                        except FileNotFoundError:
                            hasPinstance = False
                            break

                        df = df.astype(float)
                        # df = df.iloc[1:iters + 1]
                        df = df.rename(columns={"Train": f"train iter {r}", "Test": f"test iter {r}"})

                        if tmpTrain.empty:
                            tmpTrain = pd.DataFrame(df[f"train iter {r}"])
                            tmpTest = pd.DataFrame(df[f"test iter {r}"])
                        else:
                            tmpTrain = tmpTrain.join(df[f"train iter {r}"])
                            tmpTest = tmpTest.join(df[f"test iter {r}"])

                    if not hasPinstance:
                        continue

                    if accTrainDF.empty:
                        accTrainDF = pd.DataFrame(tmpTrain.mean(axis=1).rename(f"CD-{k} p = {p}"))
                        accTestDF = pd.DataFrame(tmpTest.mean(axis=1).rename(f"CD-{k} p = {p}"))
                    else:
                        accTrainDF[f"CD-{k} p = {p}"] = tmpTrain.mean(axis=1)
                        accTestDF[f"CD-{k} p = {p}"] = tmpTest.mean(axis=1)

                    accTrainDF[f"CD-{k} p = {p} - std"] = tmpTrain.std(axis=1)
                    accTrainDF[f"CD-{k} p = {p} - q1"] = tmpTrain.quantile(q=0.25, axis=1)
                    accTrainDF[f"CD-{k} p = {p} - q3"] = tmpTrain.quantile(q=0.75, axis=1)
                    accTrainDF[f"CD-{k} p = {p} - Max"] = tmpTrain.max(axis=1)
                    accTrainDF[f"CD-{k} p = {p} - Min"] = tmpTrain.min(axis=1)

                    accTestDF[f"CD-{k} p = {p} - std"] = tmpTest.std(axis=1)
                    accTestDF[f"CD-{k} p = {p} - q1"] = tmpTest.quantile(q=0.25, axis=1)
                    accTestDF[f"CD-{k} p = {p} - q3"] = tmpTest.quantile(q=0.75, axis=1)
                    accTestDF[f"CD-{k} p = {p} - Max"] = tmpTest.max(axis=1)
                    accTestDF[f"CD-{k} p = {p} - Min"] = tmpTest.min(axis=1)


            else:
                tmpTrain = pd.DataFrame()
                tmpTest = pd.DataFrame()

                hasInstance = True

                for r in range(repeat):
                    try:
                        df = pd.read_csv(fileBase.format(path, dataT, pltT, H, k, lr, bSize, iters, r),
                                         comment="#", index_col=0)
                    except FileNotFoundError:
                        hasInstance = False
                        break

                    df = df.astype(float)
                    # df = df.iloc[1:iters + 1]
                    df = df.rename(columns={"Train": f"train iter {r}", "Test": f"test iter {r}"})

                    if tmpTrain.empty:
                        tmpTrain = pd.DataFrame(df[f"train iter {r}"])
                        tmpTest = pd.DataFrame(df[f"test iter {r}"])
                    else:
                        tmpTrain = tmpTrain.join(df[f"train iter {r}"])
                        tmpTest = tmpTest.join(df[f"test iter {r}"])

                if not hasInstance:
                    continue

                accTrainDF = pd.DataFrame(tmpTrain.mean(axis=1).rename(f"CD-{k} {pltT}"))
                accTrainDF[f"CD-{k} {pltT} - std"] = tmpTrain.std(axis=1)
                accTrainDF[f"CD-{k} {pltT} - q1"] = tmpTrain.quantile(q=0.25, axis=1)
                accTrainDF[f"CD-{k} {pltT} - q3"] = tmpTrain.quantile(q=0.75, axis=1)
                accTrainDF[f"CD-{k} {pltT} - Max"] = tmpTrain.max(axis=1)
                accTrainDF[f"CD-{k} {pltT} - Min"] = tmpTrain.min(axis=1)

                accTestDF = pd.DataFrame(tmpTest.mean(axis=1).rename(f"CD-{k} {pltT}"))
                accTestDF[f"CD-{k} {pltT} - std"] = tmpTest.std(axis=1)
                accTestDF[f"CD-{k} {pltT} - q1"] = tmpTest.quantile(q=0.25, axis=1)
                accTestDF[f"CD-{k} {pltT} - q3"] = tmpTest.quantile(q=0.75, axis=1)
                accTestDF[f"CD-{k} {pltT} - Max"] = tmpTest.max(axis=1)
                accTestDF[f"CD-{k} {pltT} - Min"] = tmpTest.min(axis=1)

        accTrainDF.to_csv(outputTrainFileName)
        accTestDF.to_csv(outputTestFileName)
