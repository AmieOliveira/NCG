#!/usr/bin/python3

"""File used to plot density of connections at beginnning and end of training with NCG"""


import argparse
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from sklearn.linear_model import LinearRegression


parser = argparse.ArgumentParser()
# Main arguments
plotting = parser.add_argument_group(title='Plot Settings',
                                     description="Files' paths; plots and erros to be added")
plotting.add_argument("-i", "--inputpath", type=str, default=".",
                      help="Input files' path")
plotting.add_argument("-o", "--outputpath", type=str, default=".",
                      help="Output files' path")

# plotting.add_argument("-e", "--printerrors", action="store_true",
#                       help="Add errorbars in plot. Default is quartile errors, if you wish to use "
#                            "standard deviation error specification, use 'stderr' argument")
# plotting.add_argument("--stdErr", action="store_true",
#                       help="Add errorbars in plot, using standard deviation error. If you wish to "
#                            "use quartile error specify default error argument '-e'")
# plotting.add_argument("--minmaxErr", action="store_true",
#                       help="Add errorbars in plot, using maximum and minimum values as erros. If "
#                            "you wish to use quartile error specify default error argument '-e'")
plotting.add_argument("--limitEpochs", type=int, default=-1,
                      help="If you want the plot to use less training epochs than the total, specify here")

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
p_values = [1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]

figSize = (4.5, 4.5)
fillTransp = 0.2  # 0.3
# --------------


if __name__ == "__main__":
    args = parser.parse_args()

    path = args.inputpath

    dataT = args.dataType
    H = args.hiddenNeurons
    lr = args.learningRate
    bSize = args.batchSize
    iters = args.iterations

    repeat = args.repeat

    lim_iters = args.limitEpochs
    if lim_iters == -1:
        lim_iters = iters

    # errorPrint = ""
    # if args.printerrors:
    #     errorPrint = "-quartileErr"
    # elif args.stdErr:
    #     errorPrint = "-stdErr"
    # elif args.minmaxErr:
    #     errorPrint = "-minMaxErr"

    dSize = 0
    if dataT == "mnist":
        dSize = 784
    else:
        # TODO: plots for BAS
        print("Error: code not implemented for this data set!")
        exit(1)
    dSize = float(dSize)

    connFileName = f"{path}/meanDeg_{dataT}_sgd_H{H}_lr{lr}_mBatch{bSize}_iter{iters}-{repeat}rep.csv"

    try:
        inputFile = pd.read_csv(connFileName, index_col=0)
    except FileNotFoundError:
        print("ERROR:\tFiles not found. Check training parameters and which plots you wnat to add!")
        raise
    else:
        print(f"Found file '{connFileName}'")


    fig = plt.figure(figsize=figSize)

    for k in k_values:
        means = {"init": [], "fim": []}
        # mins = {"init": [], "fim": []}
        # maxs = {"init": [], "fim": []}

        plt.clf()
        ax = fig.add_subplot(111)
        plt.plot([0.1, 1], [0.1, 1], color="burlywood", linestyle="--", alpha=0.8, linewidth=0.6)

        hasK = False

        for p in p_values:
            try:
                meanI = inputFile[f"Mean in H, CD-{k}, p = {p}"][0]/dSize
                meanF = inputFile[f"Mean in H, CD-{k}, p = {p}"][lim_iters]/dSize
                minI = inputFile[f"Minimum in H, CD-{k}, p = {p}"][0]/dSize
                minF = inputFile[f"Minimum in H, CD-{k}, p = {p}"][lim_iters]/dSize
                maxI = inputFile[f"Maximum in H, CD-{k}, p = {p}"][0]/dSize
                maxF = inputFile[f"Maximum in H, CD-{k}, p = {p}"][lim_iters]/dSize
            except KeyError:
                continue
            else:
                hasK = True

            ax.add_patch(patches.Rectangle((minI, minF), maxI-minI, maxF-minF, color="lightcoral", alpha=0.3))

            plt.plot([meanI, meanI], [minF, maxF], color="k")
            plt.plot([minI, maxI], [meanF, meanF], color="k")
            plt.plot(meanI, meanF, marker="o", markersize=5, color="red", alpha=0.7)

            means["init"].append( meanI )
            means["fim"].append( meanF )
            # mins["init"].append( minI )
            # mins["fim"].append( minF )
            # maxs["init"].append( maxI )
            # maxs["fim"].append( maxF )


        if hasK:
            x = np.array(means["init"]).reshape((-1, 1))
            y = np.array(means["fim"]).reshape((-1, 1))
            linreg = LinearRegression().fit(x, y)

            plt.plot([0.1, 1], linreg.intercept_+linreg.coef_[0]*[0.1, 1], color="darkorange",
                     alpha=0.8, linestyle="--", label="Regression", zorder=-10)

            plt.text(0.1, 0.99, "Regression: \n" + rf"$d_f = {linreg.coef_[0][0]:.2f}d_i + {linreg.intercept_[0]:.2f}$",
                     ha="left", va="top")

            plt.grid(linestyle=':', linewidth=0.2)
            plt.xticks(p_values)
            plt.yticks(p_values)
            plt.xlabel("Initial density")
            plt.ylabel("Final density")
            plt.xlim([0, 1.1])
            plt.ylim([0, 1.1])
            # plt.show()

            filename = f"{args.outputpath}/denseAnalysis_{dataT}_H{H}_CD-{k}_lr{lr}_mBatch{bSize}_iter{lim_iters}-{repeat}rep.pdf"

            fig.savefig(filename, transparent=True)

            print(f"Saved file '{filename}'")
