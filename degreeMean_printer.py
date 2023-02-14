""" File to make plots illustrating average connectivity behavior during training: node's degrees plots """

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# from matplotlib import cm, ticker

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", type=str, required=True,
                    help="Input file where the degree information is stored")
parser.add_argument("-p", "--outputpath", type=str, default=".",
                    help="Path where outputs should be saved")
parser.add_argument("-e", "--printerrors", action="store_true",
                    help="Add errorbars in plot. Default is quartile errors, if you wish to use "
                         "standard deviation error specification, use 'stderr' argument")
parser.add_argument("--stderr", action="store_true",
                    help="Add errorbars in plot, using standard deviation error. If you wish to "
                         "use quartile error specify default error argument '-e'")
parser.add_argument("--iter", type=int, default=None,
                    help="Limits number of iterations to be plotted. If specified, plots will "
                         "show behavior only up to the i-th iteration given")
parser.add_argument("-s", "--separate-plots", action="store_true",
                    help="Select this option in order to have each plot saved individually")


# TODO: Tenho que conseguir o tamanho "basSize" automaticamente dos dados! E o learning rate!
basSize = 4
lRate = 0.01
bSize = 10
H = 100

p_val = [1, 0.75, 0.5, 0.25, 0.1]
k_val = [100, 20, 10, 5, 2, 1]

figSize = (4, 3)
filebasename = "{}/mean_nodeDegree_{}_sgd-{}_in{}_H{}_CD-{}_lr{}_mBatch{}{}.pdf"

# cms = [cm.get_cmap("Blues"), cm.get_cmap("Oranges"),
#        cm.get_cmap("Greens"), cm.get_cmap("Reds")]
# tones = [0.8, 0.6, 0.4]
# --------------------------

if __name__ == "__main__":
    args = parser.parse_args()

    dataType = ""

    if f"bas{basSize}_" in args.input:
        dataType = f"bas{basSize}"
    elif "_mnist_" in args.input:
        dataType = "mnist"
    elif "_mushrooms_" in args.input:
        dataType = "mushrooms"
    elif "_connect-4_" in args.input:
        dataType = "connect-4"

    if not dataType:
        print("ERROR: Data type not ascertained. Maybe specified BAS size does not match?")
        exit(1)

    if f"lr{lRate}" not in args.input:
        print("ERROR: Specified learning rate does not match")
        exit(1)
    # TODO: Checks para os outros parâmetros

    df = pd.read_csv(args.input)
    if args.iter:
        if (args.iter > 0) and (args.iter < len(df.index)):
            df = df.iloc[0:args.iter]

    errorPrint = ""

    if args.printerrors:
        errorPrint = "-quartileErr"
    elif args.stderr:
        errorPrint = "-stdErr"

    for k in k_val:
        n_ps = 0

        for p in p_val:
            try:
                df[f"Maximum in H, CD-{k}, p = {p}"].head(1)
                n_ps += 1
            except KeyError:
                pass

        if n_ps == 0:
            continue

        fig, ax = plt.subplots(nrows=2, ncols=n_ps, sharex='all', figsize=(n_ps*4, 5))
        fig.suptitle(f"Unit's degree for CD-{k}")
        ax = ax.flatten()


        pIdx = 0
        for p in p_val:

            if args.separate_plots:
                figSH, axSH = plt.subplots(nrows=1, ncols=1, figsize=figSize)
                figSX, axSX = plt.subplots(nrows=1, ncols=1, figsize=figSize)

            try:
                dfAux = df.rename(columns={f"Maximum in H, CD-{k}, p = {p}": "Max",
                                           f"Mean in H, CD-{k}, p = {p}": "Mean",
                                           f"Minimum in H, CD-{k}, p = {p}": "Min"})

                nH = pIdx
                nX = n_ps + pIdx

                dfAux[f"Max"].plot(ax=ax[nH])
                dfAux[f"Mean"].plot(ax=ax[nH])
                dfAux[f"Min"].plot(ax=ax[nH])
                # , linewidth=.5, linestyle=":", color=cms[pIdx](tones[0])

                if args.separate_plots:
                    dfAux[f"Max"].plot(ax=axSH)
                    dfAux[f"Mean"].plot(ax=axSH)
                    dfAux[f"Min"].plot(ax=axSH)

                dfAux = df.rename(columns={f"Maximum in X, CD-{k}, p = {p}": "Max",
                                           f"Mean in X, CD-{k}, p = {p}": "Mean",
                                           f"Minimum in X, CD-{k}, p = {p}": "Min"})
                dfAux[f"Max"].plot(ax=ax[nX])
                dfAux[f"Mean"].plot(ax=ax[nX])
                dfAux[f"Min"].plot(ax=ax[nX])

                if args.separate_plots:
                    dfAux[f"Max"].plot(ax=axSX)
                    dfAux[f"Mean"].plot(ax=axSX)
                    dfAux[f"Min"].plot(ax=axSX)
            except KeyError:
                continue

            if args.printerrors:
                indexes = df.index
                errorPlus = df[f"Maximum in H, CD-{k}, p = {p} - q3"].to_numpy()
                errorMinus = df[f"Maximum in H, CD-{k}, p = {p} - q1"].to_numpy()
                ax[nH].fill_between(indexes, errorMinus, errorPlus, alpha=0.3)
                if args.separate_plots:
                    axSH.fill_between(indexes, errorMinus, errorPlus, alpha=0.3)

                errorPlus = df[f"Maximum in X, CD-{k}, p = {p} - q3"].to_numpy()
                errorMinus = df[f"Maximum in X, CD-{k}, p = {p} - q1"].to_numpy()
                ax[nX].fill_between(indexes, errorMinus, errorPlus, alpha=0.3)
                if args.separate_plots:
                    axSX.fill_between(indexes, errorMinus, errorPlus, alpha=0.3)


                errorPlus = df[f"Mean in H, CD-{k}, p = {p} - q3"].to_numpy()
                errorMinus = df[f"Mean in H, CD-{k}, p = {p} - q1"].to_numpy()
                ax[nH].fill_between(indexes, errorMinus, errorPlus, alpha=0.3)
                if args.separate_plots:
                    axSH.fill_between(indexes, errorMinus, errorPlus, alpha=0.3)

                errorPlus = df[f"Mean in X, CD-{k}, p = {p} - q3"].to_numpy()
                errorMinus = df[f"Mean in X, CD-{k}, p = {p} - q1"].to_numpy()
                ax[nX].fill_between(indexes, errorMinus, errorPlus, alpha=0.3)
                if args.separate_plots:
                    axSX.fill_between(indexes, errorMinus, errorPlus, alpha=0.3)


                errorPlus = df[f"Minimum in H, CD-{k}, p = {p} - q3"].to_numpy()
                errorMinus = df[f"Minimum in H, CD-{k}, p = {p} - q1"].to_numpy()
                ax[nH].fill_between(indexes, errorMinus, errorPlus, alpha=0.3)
                if args.separate_plots:
                    axSH.fill_between(indexes, errorMinus, errorPlus, alpha=0.3)

                errorPlus = df[f"Minimum in X, CD-{k}, p = {p} - q3"].to_numpy()
                errorMinus = df[f"Minimum in X, CD-{k}, p = {p} - q1"].to_numpy()
                ax[nX].fill_between(indexes, errorMinus, errorPlus, alpha=0.3)
                if args.separate_plots:
                    axSX.fill_between(indexes, errorMinus, errorPlus, alpha=0.3)

            elif args.stderr:
                indexes = df.index
                error = df[f"Maximum in H, CD-{k}, p = {p} - std"].to_numpy()
                mean = df[f"Maximum in H, CD-{k}, p = {p}"].to_numpy()
                ax[nH].fill_between(indexes, mean - error, mean + error, alpha=0.3)
                error = df[f"Maximum in X, CD-{k}, p = {p} - std"].to_numpy()
                mean = df[f"Maximum in X, CD-{k}, p = {p}"].to_numpy()
                ax[nX].fill_between(indexes, mean - error, mean + error, alpha=0.3)

                error = df[f"Mean in H, CD-{k}, p = {p} - std"].to_numpy()
                mean = df[f"Mean in H, CD-{k}, p = {p}"].to_numpy()
                ax[nH].fill_between(indexes, mean - error, mean + error, alpha=0.3)
                error = df[f"Mean in X, CD-{k}, p = {p} - std"].to_numpy()
                mean = df[f"Mean in X, CD-{k}, p = {p}"].to_numpy()
                ax[nX].fill_between(indexes, mean - error, mean + error, alpha=0.3)

                error = df[f"Minimum in H, CD-{k}, p = {p} - std"].to_numpy()
                mean = df[f"Minimum in H, CD-{k}, p = {p}"].to_numpy()
                ax[nH].fill_between(indexes, mean - error, mean + error, alpha=0.3)
                error = df[f"Minimum in X, CD-{k}, p = {p} - std"].to_numpy()
                mean = df[f"Minimum in X, CD-{k}, p = {p}"].to_numpy()
                ax[nX].fill_between(indexes, mean - error, mean + error, alpha=0.3)

            ax[nH].set_title(f"p = {p}")
            ax[nX].set_xlabel("Epoch")

            if args.separate_plots:
                axSH.set_xlabel("Epoch")
                axSH.set_ylabel("Degree")

                plt.figure(figSH)
                if dataType == "minst":
                    plt.ylim(0, 800)
                elif dataType == "mushrooms":
                    plt.ylim(0, 120)
                elif dataType == "connect-4":
                    plt.ylim(0, 130)
                axSH.grid(color="gray", linestyle=":", linewidth=.2)
                axSH.legend()  # loc="upper right", prop={'size': 6}
                plt.tight_layout()
                plt.savefig(filebasename.format(args.outputpath, dataType, p, "H", H, k, lRate, bSize, errorPrint), transparent=True)
                plt.close(figSH)


                axSX.set_xlabel("Epoch")
                axSX.set_ylabel("Degree")

                plt.figure(figSX)
                plt.ylim(0, H+10)
                axSX.grid(color="gray", linestyle=":", linewidth=.2)
                axSX.legend()  # loc="upper right", prop={'size': 6}
                plt.tight_layout()
                plt.savefig(filebasename.format(args.outputpath, dataType, p, "X", H, k, lRate, bSize, errorPrint), transparent=True)
                plt.close(figSX)


            pIdx += 1

            if pIdx == n_ps:
                break

        for i in range(2):
            for j in range(n_ps):
                n = n_ps*i + j

                ax[n].grid(color="gray", linestyle=":", linewidth=.2)
                ax[n].legend(prop={'size': 6})  # loc="upper right",

                # ax[i, j].set_ylim(-1, 17)
                # ax[i, j].set_yticks(np.arange(0, 17, step=4), minor=False)
                ax[n].yaxis.set_tick_params(labelbottom=True)
                # ax[i, j].tick_params(left=False)

        ax[0].set_ylabel("Hidden\nDegree")
        ax[n_ps].set_ylabel("Visible\nDegree")

        plt.figure(fig)
        plt.tight_layout()

        filename = f"{args.outputpath}/mean_nodeDegree_{dataType}_SGD_H{H}_CD-{k}_lr{lRate}_mBatch{bSize}{errorPrint}.pdf"
        print(f"Saving as: {filename}")
        plt.savefig(filename, transparent=True)
