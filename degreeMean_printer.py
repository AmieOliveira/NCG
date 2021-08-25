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


# TODO: Tenho que conseguir o tamanho "basSize" automaticamente dos dados! E o learning rate!
basSize = 4
lRate = 0.05

p_val = [1, 0.75, 0.5, 0.25]
k_val = [100, 20, 10, 5, 2, 1]

# cms = [cm.get_cmap("Blues"), cm.get_cmap("Oranges"),
#        cm.get_cmap("Greens"), cm.get_cmap("Reds")]
# tones = [0.8, 0.6, 0.4]
# --------------------------

if __name__ == "__main__":
    args = parser.parse_args()

    if f"bas{basSize}_" not in args.input:
        print("ERROR: Specified BAS size does not match")
        exit(1)
    if f"lr{lRate}-" not in args.input:
        print("ERROR: Specified learning rate does not match")
        exit(1)

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
                df[f"Maximum in H, CD-{k}, p = {p}"].head()
                n_ps += 1
            except KeyError:
                pass

        if n_ps == 0:
            continue

        fig, ax = plt.subplots(nrows=2, ncols=n_ps, sharex='all', figsize=(n_ps*4, 5))
        fig.suptitle(f"Unit's degree for CD-{k}")


        pIdx = 0
        for p in p_val:
            try:
                dfAux = df.rename(columns={f"Maximum in H, CD-{k}, p = {p}": "Max",
                                           f"Mean, CD-{k}, p = {p}": "Mean",
                                           f"Minimum in H, CD-{k}, p = {p}": "Min"})
                dfAux[f"Max"].plot(ax=ax[0, pIdx])
                dfAux[f"Mean"].plot(ax=ax[0, pIdx])
                dfAux[f"Min"].plot(ax=ax[0, pIdx])
                # , linewidth=.5, linestyle=":", color=cms[pIdx](tones[0])

                dfAux = df.rename(columns={f"Maximum in X, CD-{k}, p = {p}": "Max",
                                           f"Mean, CD-{k}, p = {p}": "Mean",
                                           f"Minimum in X, CD-{k}, p = {p}": "Min"})
                dfAux[f"Max"].plot(ax=ax[1, pIdx])
                dfAux[f"Mean"].plot(ax=ax[1, pIdx])
                dfAux[f"Min"].plot(ax=ax[1, pIdx])
            except KeyError:
                continue

            if args.printerrors:
                indexes = df.index
                errorPlus = df[f"Maximum in H, CD-{k}, p = {p} - q3"].to_numpy()
                errorMinus = df[f"Maximum in H, CD-{k}, p = {p} - q1"].to_numpy()
                ax[0, pIdx].fill_between(indexes, errorMinus, errorPlus, alpha=0.3)
                errorPlus = df[f"Maximum in X, CD-{k}, p = {p} - q3"].to_numpy()
                errorMinus = df[f"Maximum in X, CD-{k}, p = {p} - q1"].to_numpy()
                ax[1, pIdx].fill_between(indexes, errorMinus, errorPlus, alpha=0.3)

                errorPlus = df[f"Mean, CD-{k}, p = {p} - q3"].to_numpy()
                errorMinus = df[f"Mean, CD-{k}, p = {p} - q1"].to_numpy()
                ax[0, pIdx].fill_between(indexes, errorMinus, errorPlus, alpha=0.3)
                ax[1, pIdx].fill_between(indexes, errorMinus, errorPlus, alpha=0.3)

                errorPlus = df[f"Minimum in H, CD-{k}, p = {p} - q3"].to_numpy()
                errorMinus = df[f"Minimum in H, CD-{k}, p = {p} - q1"].to_numpy()
                ax[0, pIdx].fill_between(indexes, errorMinus, errorPlus, alpha=0.3)
                errorPlus = df[f"Minimum in X, CD-{k}, p = {p} - q3"].to_numpy()
                errorMinus = df[f"Minimum in X, CD-{k}, p = {p} - q1"].to_numpy()
                ax[1, pIdx].fill_between(indexes, errorMinus, errorPlus, alpha=0.3)

            elif args.stderr:
                indexes = df.index
                error = df[f"Maximum in H, CD-{k}, p = {p} - std"].to_numpy()
                mean = df[f"Maximum in H, CD-{k}, p = {p}"].to_numpy()
                ax[0, pIdx].fill_between(indexes, mean - error, mean + error, alpha=0.3)
                error = df[f"Maximum in X, CD-{k}, p = {p} - std"].to_numpy()
                mean = df[f"Maximum in X, CD-{k}, p = {p}"].to_numpy()
                ax[1, pIdx].fill_between(indexes, mean - error, mean + error, alpha=0.3)

                error = df[f"Mean, CD-{k}, p = {p} - std"].to_numpy()
                mean = df[f"Mean, CD-{k}, p = {p}"].to_numpy()
                ax[0, pIdx].fill_between(indexes, mean - error, mean + error, alpha=0.3)
                ax[1, pIdx].fill_between(indexes, mean - error, mean + error, alpha=0.3)

                error = df[f"Minimum in H, CD-{k}, p = {p} - std"].to_numpy()
                mean = df[f"Minimum in H, CD-{k}, p = {p}"].to_numpy()
                ax[0, pIdx].fill_between(indexes, mean - error, mean + error, alpha=0.3)
                error = df[f"Minimum in X, CD-{k}, p = {p} - std"].to_numpy()
                mean = df[f"Minimum in X, CD-{k}, p = {p}"].to_numpy()
                ax[1, pIdx].fill_between(indexes, mean - error, mean + error, alpha=0.3)

            ax[0, pIdx].set_title(f"p = {p}")
            ax[1, pIdx].set_xlabel("Iteration")

            pIdx += 1

            if pIdx == n_ps:
                break

        for i in range(2):
            for j in range(n_ps):
                ax[i, j].grid(color="gray", linestyle=":", linewidth=.2)
                ax[i, j].legend(prop={'size': 6})  # loc="upper right",

                ax[i, j].set_ylim(-1, 17)
                ax[i, j].set_yticks(np.arange(0, 17, step=4), minor=False)
                ax[i, j].yaxis.set_tick_params(labelbottom=True)
                # ax[i, j].tick_params(left=False)

        ax[0, 0].set_ylabel("Hidden\nMean degree")
        ax[1, 0].set_ylabel("Visible\nMean degree")

        filename = f"{args.outputpath}/mean_nodeDegree_bas{basSize}_SGD_CD-{k}_lr{lRate}{errorPrint}.pdf"
        print(f"Saving as: {filename}")
        plt.savefig(filename, transparent=True)