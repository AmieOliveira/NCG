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
    nLabels = 0
    if args.hasLabels:
        nLabels = 10 if dataT == "mnist" else 2

    repeat = args.repeat

    size = X + nLabels

    outputFileName = f"{path}/degreeMean_{dataT}_sgd_H{H}_lr{lr}_mBatch{bSize}_iter{iters}{labels}.csv"

    indexes = []
    firstFile = True

    connectivityDF = pd.DataFrame()

    for k in k_values:
        for p in p_values:
            n_connections = np.zeros((iters + 1, repeat))

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

                    for i in range(H):
                        for j in range(X):
                            n_connections += int(connections[size*i + j])

                    itIdx += 1

                firstFile = False

            if not hasInstance:
                continue

            connectivityDF[f"Connections, CD-{k}, p = {p}"] = n_connections.mean(axis=1)
            connectivityDF[f"Connections, CD-{k}, p = {p} - std"] = n_connections.std(axis=1)
            connectivityDF[f"Connections, CD-{k}, p = {p} - q1"] = np.quantile(n_connections, q=0.25, axis=1)
            connectivityDF[f"Connections, CD-{k}, p = {p} - q3"] = np.quantile(n_connections, q=0.75, axis=1)

            n_connections /= X*H

            connectivityDF[f"Sparsity, CD-{k}, p = {p}"] = n_connections.mean(axis=1)
            connectivityDF[f"Sparsity, CD-{k}, p = {p} - std"] = n_connections.std(axis=1)
            connectivityDF[f"Sparsity, CD-{k}, p = {p} - q1"] = np.quantile(n_connections, q=0.25, axis=1)
            connectivityDF[f"Sparsity, CD-{k}, p = {p} - q3"] = np.quantile(n_connections, q=0.75, axis=1)

    connectivityDF.set_index(np.array(indexes), inplace=True)
    connectivityDF.to_csv(
        f"{path}/meanSparsity_{dataT}_sgd_H{H}_lr{lr}_mBatch{bSize}_iter{iters}-{repeat}rep.csv")