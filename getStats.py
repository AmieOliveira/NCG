""" File to get RBM training statistics """
# So far, implemented to get minimum and last 10 iterations'

import argparse
import pandas as pd
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument("-i", "--inputs", type=str, nargs="+", required=True,
                    help="Input file where the degree information is stored")
parser.add_argument("-p", "--outputpath", type=str, default=".",
                    help="Path where outputs should be saved")
parser.add_argument("--iter", type=int, default=None,
                    help="Limits number of iterations to be plotted. If specified, plots will "
                         "show behavior only up to the i-th iteration given")

k_val = [1, 2, 5, 10, 20, 100]
v_val = [4, 6, 8, 10, 12, 14]
s_val = [2, 3, 4]
p_val = [0.25, 0.5, 0.75, 1]

# TODO: get automatically
basSize = 4
repeat = 25
lRate = 0.01


if __name__ == "__main__":
    args = parser.parse_args()

    dfList = []

    for fname in args.inputs:
        if f"bas{basSize}" not in fname:
            print(f"ERROR:\tWrong BAS size!! Not processing file '{fname}'")
            continue
        if f"-{repeat}rep" not in fname:
            print(f"ERROR:\tWrong number of repetitions!! Not processing file '{fname}'")
            continue
        # TODO: Learning rate. So far only SGD is using different learnign rates...

        df = pd.read_csv(fname, index_col=0)

        iter = args.iter
        if not iter:
            iter = int(df.index[-1])

        if "complete" in fname:

            for k in k_val:
                try:
                    minVal = df[f"CD-{k}"].min()
                    idxMin = df[f"CD-{k}"].idxmin()

                    lastVals = df[f"CD-{k}"].iloc[iter-9:iter+1].mean()
                except KeyError:
                    continue

                dfK = pd.DataFrame(data={f"CD-{k}, Complete": [minVal, idxMin, lastVals]},
                                   index=["Min", "MinIdx", "Mean 10 last"])
                dfList += [dfK]

        elif "neighbors" in fname:
            nType = "line"
            if "spiral" in fname:
                nType = "spiral"

            for k in k_val:
                for v in v_val:
                    try:
                        minVal = df[f"CD-{k}, {v} neighbors in {nType}"].min()
                        idxMin = df[f"CD-{k}, {v} neighbors in {nType}"].idxmin()

                        lastVals = df[f"CD-{k}, {v} neighbors in {nType}"].iloc[iter-9:iter+1].mean()
                    except KeyError:
                        continue

                    dfKV = pd.DataFrame(data={f"CD-{k}, {v} neighbors in {nType}": [minVal, idxMin, lastVals]},
                                        index=["Min", "MinIdx", "Mean 10 last"])
                    dfList += [dfKV]

        elif "BAScon" in fname:
            for k in k_val:
                for s in s_val:
                    try:
                        minVal = df[f"CD-{k}, Specialist v{s}"].min()
                        idxMin = df[f"CD-{k}, Specialist v{s}"].idxmin()

                        lastVals = df[f"CD-{k}, Specialist v{s}"].iloc[iter-9:iter+1].mean()
                    except KeyError:
                        continue

                    dfKS = pd.DataFrame(data={f"CD-{k}, Specialist v{s}": [minVal, idxMin, lastVals]},
                                        index=["Min", "MinIdx", "Mean 10 last"])
                    dfList += [dfKS]

        elif "SGD" in fname:
            for k in k_val:
                for p in p_val:
                    try:
                        minVal = df[f"CD-{k}, p = {p}"].min()
                        idxMin = df[f"CD-{k}, p = {p}"].idxmin()

                        lastVals = df[f"CD-{k}, p = {p}"].iloc[iter - 9:iter + 1].mean()
                    except KeyError:
                        continue

                    dfKP = pd.DataFrame(data={f"CD-{k}, SGD p = {p}": [minVal, idxMin, lastVals]},
                                        index=["Min", "MinIdx", "Mean 10 last"])
                    dfList += [dfKP]

    data = pd.concat(dfList, axis=1)

    data.to_csv(f"{args.outputpath}/stats_meanNLLvalues_bas{basSize}_lr{lRate}-{repeat}rep.csv")
