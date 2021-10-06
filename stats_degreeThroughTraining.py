import argparse
import pandas as pd
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("-p", "--outputpath", type=str, default=".",
                    help="Input and output files' path")
parser.add_argument("-d", "--dataType", type=str, default="mnist",
                    help="So far can be either 'mnist' or 'basX', in which X is the size of the BAS dataset")

parser.add_argument("-l", "--hasLabels", action="store_true",
                    help="Number of runs for each configuration")

parser.add_argument("-R", "--repeat", type=int, required=True,
                    help="Number of runs for each configuration")
parser.add_argument("-H", "--hiddenNeurons", type=int, required=True,
                    help="Number of hidden neurons the RBM's have")
parser.add_argument("-L", "--learningRate", type=float, required=True,
                    help="Learning rate utilized during training")
parser.add_argument("-B", "--batchSize", type=int, required=True,
                    help="Size of the training mini-batchs")
parser.add_argument("-I", "--iterations", type=int, required=True,
                    help="Number of training iterations (epochs)")


# Auxiliar lists
#    Contain possible values for k and p. More values can be added as needed
k_values = [1, 2, 5, 10, 20, 100]
p_values = [1, 0.5, 0.1]
# --------------

fileBase = "{}/connectivity_{}_sgd-{}_H{}_CD-{}_lr{}_mBatch{}_iter{}{}_run{}.csv"

if __name__ == "__main__":
    args = parser.parse_args()

    path = args.outputpath

    dataT = args.dataType
    H = args.hiddenNeurons
    lr = args.learningRate
    bSize = args.batchSize
    iters = args.iterations

    labels = "_withLabels" if args.hasLabels else ""
    X = 784 if dataT == "mnist" else int(dataT[3:])**2
    nLabels = 10 if dataT == "mnist" else 2

    repeat = args.repeat

    size = X + nLabels

    outputFileName = f"{path}/degreeMean_{dataT}_sgd_H{H}_lr{lr}_mBatch{bSize}_iter{iters}{labels}.csv"

    indexes = []
    firstFile = True

    connectivityDF = pd.DataFrame()

    for k in k_values:
        for p in p_values:
            gh_max = np.zeros((iters + 1, repeat))
            gh_med = np.zeros((iters + 1, repeat))
            gh_min = np.zeros((iters + 1, repeat))

            gx_max = np.zeros((iters + 1, repeat))
            gx_med = np.zeros((iters + 1, repeat))
            gx_min = np.zeros((iters + 1, repeat))

            hasInstance = True

            for r in range(repeat):
                try:
                    netCon = open(fileBase.format(path, dataT, p, H, k, lr, bSize, iters, labels, r))
                    print(f"File found! \t (p = {p}, k = {k})")
                except FileNotFoundError:
                    hasInstance = False
                    break

                itIdx = 0
                while True:
                    line = netCon.readline()

                    if not line:
                        break
                    if line[0] == "#":
                        continue

                    connections = line.split(",")

                    tmp = int(connections[0])
                    if tmp > iters:
                        break

                    if firstFile:
                        indexes += [tmp]

                    connections = connections[1:]

                    xDegrees = np.zeros(X)
                    hDegrees = np.zeros(H)

                    for i in range(H):
                        for j in range(X):
                            xDegrees[j] += int(connections[size*i + j])
                            hDegrees[i] += int(connections[size*i + j])

                    sumOfdegs = hDegrees[0]
                    gh_max[itIdx, r] = hDegrees[0]
                    gh_min[itIdx, r] = hDegrees[0]
                    for i in range(1, H):
                        sumOfdegs += hDegrees[i]

                        if hDegrees[i] < gh_min[itIdx, r]:
                            gh_min[itIdx, r] = hDegrees[i]
                        if hDegrees[i] > gh_max[itIdx, r]:
                            gh_max[itIdx, r] = hDegrees[i]

                    gh_med[itIdx] = float(sumOfdegs)/H

                    sumOfdegs = xDegrees[0]
                    gx_max[itIdx, r] = xDegrees[0]
                    gx_min[itIdx, r] = xDegrees[0]
                    for j in range(1, X):
                        sumOfdegs += xDegrees[j]

                        if xDegrees[j] < gx_min[itIdx, r]:
                            gx_min[itIdx, r] = xDegrees[j]
                        if xDegrees[j] > gx_max[itIdx, r]:
                            gx_max[itIdx, r] = xDegrees[j]

                    gx_med[itIdx, r] = float(sumOfdegs)/X

                    itIdx += 1

                firstFile = False

            # print("Max in H\n", gh_max)
            # print("Mean in H\n", gh_med)
            # print("Min in H\n", gh_min)
            # print("Max in X\n", gx_max)
            # print("Mean in X\n", gx_med)
            # print("Min in X\n", gx_min)

            if not hasInstance:
                continue

            connectivityDF[f"Mean in H, CD-{k}, p = {p}"] = gh_med.mean(axis=1)
            connectivityDF[f"Mean in H, CD-{k}, p = {p} - std"] = gh_med.std(axis=1)
            connectivityDF[f"Mean in H, CD-{k}, p = {p} - q1"] = np.quantile(gh_med, q=0.25, axis=1)
            connectivityDF[f"Mean in H, CD-{k}, p = {p} - q3"] = np.quantile(gh_med, q=0.75, axis=1)

            connectivityDF[f"Mean in X, CD-{k}, p = {p}"] = gx_med.mean(axis=1)
            connectivityDF[f"Mean in X, CD-{k}, p = {p} - std"] = gx_med.std(axis=1)
            connectivityDF[f"Mean in X, CD-{k}, p = {p} - q1"] = np.quantile(gx_med, q=0.25, axis=1)
            connectivityDF[f"Mean in X, CD-{k}, p = {p} - q3"] = np.quantile(gx_med, q=0.75, axis=1)

            connectivityDF[f"Maximum in X, CD-{k}, p = {p}"] = gx_max.mean(axis=1)
            connectivityDF[f"Maximum in X, CD-{k}, p = {p} - std"] = gx_max.std(axis=1)
            connectivityDF[f"Maximum in X, CD-{k}, p = {p} - q1"] = np.quantile(gx_max, q=0.25, axis=1)
            connectivityDF[f"Maximum in X, CD-{k}, p = {p} - q3"] = np.quantile(gx_max, q=0.75, axis=1)

            connectivityDF[f"Maximum in H, CD-{k}, p = {p}"] = gh_max.mean(axis=1)
            connectivityDF[f"Maximum in H, CD-{k}, p = {p} - std"] = gh_max.std(axis=1)
            connectivityDF[f"Maximum in H, CD-{k}, p = {p} - q1"] = np.quantile(gh_max, q=0.25, axis=1)
            connectivityDF[f"Maximum in H, CD-{k}, p = {p} - q3"] = np.quantile(gh_max, q=0.75, axis=1)

            connectivityDF[f"Minimum in X, CD-{k}, p = {p}"] = gx_min.mean(axis=1)
            connectivityDF[f"Minimum in X, CD-{k}, p = {p} - std"] = gx_min.std(axis=1)
            connectivityDF[f"Minimum in X, CD-{k}, p = {p} - q1"] = np.quantile(gx_min, q=0.25, axis=1)
            connectivityDF[f"Minimum in X, CD-{k}, p = {p} - q3"] = np.quantile(gx_min, q=0.75, axis=1)

            connectivityDF[f"Minimum in H, CD-{k}, p = {p}"] = gh_min.mean(axis=1)
            connectivityDF[f"Minimum in H, CD-{k}, p = {p} - std"] = gh_min.std(axis=1)
            connectivityDF[f"Minimum in H, CD-{k}, p = {p} - q1"] = np.quantile(gh_min, q=0.25, axis=1)
            connectivityDF[f"Minimum in H, CD-{k}, p = {p} - q3"] = np.quantile(gh_min, q=0.75, axis=1)

    connectivityDF.set_index(np.array(indexes), inplace=True)
    connectivityDF.to_csv(
        f"{path}/meanDeg_{dataT}_sgd_H{H}_lr{lr}_mBatch{bSize}_iter{iters}-{repeat}rep.csv")