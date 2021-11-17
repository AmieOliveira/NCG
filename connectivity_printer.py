""" File to make plots illustrating connectivity behavior during training """

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.axes_grid1 import make_axes_locatable

# Inputs ---------------
# basSize = 4
k = 10
p = 1
H = 15
seed = 40
lrate = 0.01
bsize = 50
iters = 100

cBatch = "Batch1k"
timeScale = 5.3
limiar = 0.08

# inputfilename = f"Training Outputs/Teste SGD/connectivity_bas{basSize}_SGD_CD-{k}_p{p}_seed{seed}.csv"
inputfilename = f"Training Outputs/Teste Gradiente/Batch1k_scale5-intermitente-20-40_lim0.08/connectivity_mnist_sgd-{p}_H{H}_CD-{k}_lr{lrate}_mBatch{bsize}_iter{iters}_seed{seed}_run0.csv"
outpath = "Plots/Teste Gradiente"
# ---------------

size = 784  # basSize**2

figSize = {"default": (6.7, 5), "wide": (13, 5)}

extras = f"-{cBatch}-timeScale{timeScale}-limiar{limiar}"

def read_input(filename):
    df = pd.read_csv(filename, comment="#", index_col=0, header=None)
    df = df.astype(int)
    arr = df.to_numpy()
    return arr  # np.reshape(arr, (arr.shape[0], H, size))


if __name__ == "__main__":
    conn = read_input(inputfilename)
    # print(conn[-1])  # Verificação parece ok!

    X = size

    iterations = conn.shape[0]
    size = conn.shape[1]

    # Degree information graphs
    gh_max = np.zeros(iterations)
    gh_med = np.zeros(iterations)
    gh_min = np.zeros(iterations)

    gx_max = np.zeros(iterations)
    gx_med = np.zeros(iterations)
    gx_min = np.zeros(iterations)

    for itIdx in range(iterations):
        xDegrees = np.zeros(X)
        hDegrees = np.zeros(H)

        for i in range(H):
            for j in range(X):
                xDegrees[j] += conn[itIdx, X * i + j]
                hDegrees[i] += conn[itIdx, X * i + j]

        sumOfdegs = hDegrees[0]
        gh_max[itIdx] = hDegrees[0]
        gh_min[itIdx] = hDegrees[0]

        for i in range(1, H):
            sumOfdegs += hDegrees[i]

            if hDegrees[i] < gh_min[itIdx]:
                gh_min[itIdx] = hDegrees[i]
            if hDegrees[i] > gh_max[itIdx]:
                gh_max[itIdx] = hDegrees[i]

        gh_med[itIdx] = float(sumOfdegs) / H

        sumOfdegs = xDegrees[0]
        gx_max[itIdx] = xDegrees[0]
        gx_min[itIdx] = xDegrees[0]

        for j in range(1, X):
            sumOfdegs += xDegrees[j]

            if xDegrees[j] < gx_min[itIdx]:
                gx_min[itIdx] = xDegrees[j]
            if xDegrees[j] > gx_max[itIdx]:
                gx_max[itIdx] = xDegrees[j]

        gx_med[itIdx] = float(sumOfdegs) / X

    fig, ax = plt.subplots(nrows=2, ncols=1, sharex='col', figsize=figSize["default"])
    fig.suptitle(f"Unit's degree for CD-{k}, p = {p}")

    x_axis = list(range(0, iters+1, int(timeScale)))

    ax[0].step(x_axis, gh_max, where='post', label="Maximum")
    ax[0].step(x_axis, gh_med, where='post', label="Mean")
    ax[0].step(x_axis, gh_min, where='post', label="Minimum")
    ax[0].set_ylabel("Degree of hidden units")
    ax[0].grid(color="gray", linestyle=":", linewidth=.2)
    ax[0].legend(loc="upper right", prop={'size': 6})

    ax[1].step(x_axis, gx_max, where='post', label="Maximum")
    ax[1].step(x_axis, gx_med, where='post', label="Mean")
    ax[1].step(x_axis, gx_min, where='post', label="Minimum")
    ax[1].set_ylabel("Degree of visible units")
    ax[1].grid(color="gray", linestyle=":", linewidth=.2)
    ax[1].legend(loc="upper right", prop={'size': 6})

    plt.xlabel("Iteration")

    # plt.show()
    plt.savefig(f"{outpath}/nodeDegree_mnist_SGD_CD-{k}_lr{lrate}_mBatch{bsize}_iter{iters}_seed{seed}{extras}.pdf", transparent=True)


    # # Edge activation graphs
    # activation = [[sum(conn[:, r, c])/float(iterations) for c in range(size)] for r in range(size)]
    #
    # spacingX = 20
    # spacingY = 100
    #
    # cmapname = "copper"  # "Greys"
    # cmap = cm.get_cmap(cmapname)
    #
    # fig, ax = plt.subplots(1, figsize=figSize["wide"])
    # ax.set_aspect('equal')
    # plt.title(f"Activation pattern through training for CD-{k}, p = {p}")
    #
    # for r in range(size):
    #     for c in range(size):
    #         plt.plot([r*spacingX, c*spacingX], [spacingY, 0], color=cmap(activation[r][c]), linewidth=1, alpha=0.85)
    #
    # for i in range(size):
    #     c_h = plt.Circle( (spacingX*i, spacingY), spacingX/4, color='grey', zorder=5 )
    #     ax.add_patch(c_h)
    #     plt.annotate(r"$h_{{ {} }}$".format(i), (spacingX*i, spacingY), ha="center", va="center", zorder=10)
    #
    #     c_x = plt.Circle( (spacingX*i, 0), spacingX/4, color='firebrick', zorder=5 )
    #     ax.add_patch(c_x)
    #     plt.annotate(r"$x_{{ {} }}$".format(i), (spacingX*i, 0), ha="center", va="center", zorder=10)
    #
    # plt.tick_params(
    #     axis='both',
    #     which='major',
    #     bottom=False,
    #     left=False,
    #     labelbottom=False,
    #     labelleft=False
    # )
    #
    # divider = make_axes_locatable(ax)
    # cax = divider.append_axes("right", size="5%", pad=0.05)
    # fig.colorbar(cm.ScalarMappable(cmap=cmapname), cax=cax)
    #
    # # fig.tight_layout()
    # # plt.show()
    # plt.savefig(f"{outpath}/edgeActivation_bas{basSize}_SGD_CD-{k}_p{p}_seed{seed}.pdf", transparent=True)
