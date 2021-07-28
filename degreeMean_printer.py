""" File to make plots illustrating average connectivity behavior during training: node's degrees plots """

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", type=str, required=True,
                    help="Input file where the degree information is stored")
parser.add_argument("-p", "--outputpath", type=str, default=".",
                    help="Path where outputs should be saved")
parser.add_argument("-e", "--printerrors", action="store_true",
                    help="Add errorbars in plot. Default is quartile errors, if you wish to use "
                         "standard deviation error specification, use 'stderr' argument")
parser.add_argument("--stderr", action="store_true",
                    help="Add errorbars in plot, using standard deviation error. If you wish to use "
                         "quartile error specify default error argument '-e")


figSize = {"default": (6.7, 5), "wide": (13, 5)}

# TODO: Tenho que conseguir o tamanho "basSize" automaticamente dos dados!
basSize = 4
p_val = [1, 0.75, 0.5]
k_val = [100, 20, 10, 5, 2, 1]
# --------------------------

if __name__ == "__main__":
    args = parser.parse_args()

    df = pd.read_csv(args.input)

    errorPrint = ""

    if args.printerrors:
        errorPrint = "-quartileErr"
    elif args.stderr:
        errorPrint = "-stdErr"

    for k in k_val:
        hasK = False

        fig, ax = plt.subplots(nrows=2, ncols=1, sharex='col', figsize=figSize["default"])
        ax[0].set_title(f"Unit's degree for CD-{k}")

        for p in p_val:
            try:
                df[f"Maximum in H, CD-{k}, p = {p}"].plot(ax=ax[0])
                df[f"Mean, CD-{k}, p = {p}"].plot(ax=ax[0])
                df[f"Minimum in H, CD-{k}, p = {p}"].plot(ax=ax[0])

                df[f"Maximum in X, CD-{k}, p = {p}"].plot(ax=ax[1])
                df[f"Mean, CD-{k}, p = {p}"].plot(ax=ax[1])
                df[f"Minimum in X, CD-{k}, p = {p}"].plot(ax=ax[1])
            except KeyError:
                continue

            if args.printerrors:
                indexes = df.index
                errorPlus = df[f"Maximum in H, CD-{k}, p = {p} - q3"].to_numpy()
                errorMinus = df[f"Maximum in H, CD-{k}, p = {p} - q1"].to_numpy()
                ax[0].fill_between(indexes, errorMinus, errorPlus, alpha=0.3)
                errorPlus = df[f"Maximum in X, CD-{k}, p = {p} - q3"].to_numpy()
                errorMinus = df[f"Maximum in X, CD-{k}, p = {p} - q1"].to_numpy()
                ax[1].fill_between(indexes, errorMinus, errorPlus, alpha=0.3)

                errorPlus = df[f"Mean, CD-{k}, p = {p} - q3"].to_numpy()
                errorMinus = df[f"Mean, CD-{k}, p = {p} - q1"].to_numpy()
                ax[0].fill_between(indexes, errorMinus, errorPlus, alpha=0.3)
                ax[1].fill_between(indexes, errorMinus, errorPlus, alpha=0.3)

                errorPlus = df[f"Minimum in H, CD-{k}, p = {p} - q3"].to_numpy()
                errorMinus = df[f"Minimum in H, CD-{k}, p = {p} - q1"].to_numpy()
                ax[0].fill_between(indexes, errorMinus, errorPlus, alpha=0.3)
                errorPlus = df[f"Minimum in X, CD-{k}, p = {p} - q3"].to_numpy()
                errorMinus = df[f"Minimum in X, CD-{k}, p = {p} - q1"].to_numpy()
                ax[1].fill_between(indexes, errorMinus, errorPlus, alpha=0.3)

            elif args.stderr:
                indexes = df.index
                error = df[f"Maximum in H, CD-{k}, p = {p} - std"].to_numpy()
                mean = df[f"Maximum in H, CD-{k}, p = {p}"].to_numpy()
                ax[0].fill_between(indexes, mean - error, mean + error, alpha=0.3)
                error = df[f"Maximum in X, CD-{k}, p = {p} - std"].to_numpy()
                mean = df[f"Maximum in X, CD-{k}, p = {p}"].to_numpy()
                ax[1].fill_between(indexes, mean - error, mean + error, alpha=0.3)

                error = df[f"Mean, CD-{k}, p = {p} - std"].to_numpy()
                mean = df[f"Mean, CD-{k}, p = {p}"].to_numpy()
                ax[0].fill_between(indexes, mean - error, mean + error, alpha=0.3)
                ax[1].fill_between(indexes, mean - error, mean + error, alpha=0.3)

                error = df[f"Minimum in H, CD-{k}, p = {p} - std"].to_numpy()
                mean = df[f"Minimum in H, CD-{k}, p = {p}"].to_numpy()
                ax[0].fill_between(indexes, mean - error, mean + error, alpha=0.3)
                error = df[f"Minimum in X, CD-{k}, p = {p} - std"].to_numpy()
                mean = df[f"Minimum in X, CD-{k}, p = {p}"].to_numpy()
                ax[1].fill_between(indexes, mean - error, mean + error, alpha=0.3)

            hasK = True

        if hasK:
            ax[0].grid(color="gray", linestyle=":", linewidth=.2)
            ax[0].legend(loc="upper right", prop={'size': 6})
            ax[0].set_ylabel("Degree of hidden unit")

            ax[1].grid(color="gray", linestyle=":", linewidth=.2)
            ax[1].legend(loc="upper right", prop={'size': 6})
            ax[1].set_ylabel("Degree of visible unit")
            plt.xlabel("Iteration")

            filename = f"{args.outputpath}/mean_nodeDegree_bas{basSize}_SGD_CD-{k}{errorPrint}.pdf"
            print(f"Saving as: {filename}")
            plt.savefig(filename, transparent=True)
