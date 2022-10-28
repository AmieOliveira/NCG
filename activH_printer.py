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
parser.add_argument("--mmerr", "--minmaxerr", action="store_true",
                    help="Add errorbars in plot, using minimum and maximum values. If you wish "
                         "to use quartile error specify default error argument '-e'")
parser.add_argument("--iter", type=int, default=None,
                    help="Limits number of iterations to be plotted. If specified, plots will "
                         "show behavior only up to the i-th iteration given")
# parser.add_argument("-s", "--separate-plots", action="store_true",
#                     help="Select this option in order to have each plot saved individually")


# TODO: Tenho que conseguir o tamanho "basSize" automaticamente dos dados! E o learning rate!
basSize = 4
lRate = 0.1
bSize = 50
H = 500

p_val = [1, 0.75, 0.5, 0.25, 0.1]
k_val = [100, 20, 10, 5, 2, 1]

figSize = (4, 3)
filebasename = "{}/meanActUnts_{}_ncgh-{}_in{}_H{}_CD-{}_lr{}_mBatch{}{}.pdf"

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

    if not dataType:
        print("ERROR: Data type not ascertained. Maybe specified BAS size does not match?")
        exit(1)

    if f"lr{lRate}" not in args.input:
        print("ERROR: Specified learning rate does not match")
        exit(1)
    # TODO: Checks para os outros parâmetros

    df = pd.read_csv(args.input, index_col=0)
    if args.iter:
        if (args.iter > 0) and (args.iter < len(df.index)):
            df = df.iloc[0:args.iter]

    errorPrint = ""

    if args.printerrors:
        errorPrint = "-quartileErr"
    elif args.stderr:
        errorPrint = "-stdErr"
    elif args.mmerr:
        errorPrint = "-minMaxErr"

    for k in k_val:
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=figSize)
        hasFig = False

        for p in p_val:
            try:
                df[f"Acting H, CD-{k}, p = {p}"].head(1)
            except KeyError:
                continue

            hasFig = True

            tmp = df.rename(columns={f"Acting H, CD-{k}, p = {p}": f"p = {p}"})
            # tmp[f"p = {p}"].plot(ax=ax, linewidth=linwdth, alpha=0.8)
            tmp[f"p = {p}"].plot()

            if args.printerrors:
                indexes = df.index
                errorPlus = df[f"Acting H, CD-{k}, p = {p} - q3"].to_numpy()
                errorMinus = df[f"Acting H, CD-{k}, p = {p} - q1"].to_numpy()
                ax.fill_between(indexes, errorMinus, errorPlus, alpha=0.3)

            elif args.stderr:
                indexes = df.index
                error = df[f"Acting H, CD-{k}, p = {p} - std"].to_numpy()
                mean = df[f"Acting H, CD-{k}, p = {p}"].to_numpy()
                ax.fill_between(indexes, mean - error, mean + error, alpha=0.3)

            elif args.mmerr:
                indexes = df.index
                errorPlus = df[f"Acting H, CD-{k}, p = {p} - Max"].to_numpy()
                errorMinus = df[f"Acting H, CD-{k}, p = {p} - Min"].to_numpy()
                ax.fill_between(indexes, errorMinus, errorPlus, alpha=0.3)

        if hasFig:
            ax.set_title(f"CD-{k}")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Active hidden units (H)")

            # plt.plot([0, 3000], [1, 1], "--k", linewidth=0.5)

            ax.grid(color="gray", linestyle=":", linewidth=.2)
            ax.legend(prop={'size': 8})  # loc="upper right"

            plt.figure(fig)
            plt.tight_layout()

            filename = f"{args.outputpath}/mean_nodeDegree_{dataType}_ncgh_H{H}_CD-{k}_lr{lRate}_mBatch{bSize}_iter{df.index[-1]}{errorPrint}.pdf"
            print(f"Saving as: {filename}")
            plt.savefig(filename, transparent=True)

