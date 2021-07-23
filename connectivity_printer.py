""" File to make plots illustrating connectivity behavior during training """

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.axes_grid1 import make_axes_locatable

# Inputs ---------------
basSize = 4
k = 1
p = 1
seed = 89237

inputfilename = f"Training Outputs/Teste SGD/connectivity_bas{basSize}_SGD_CD-{k}_p{p}_seed{seed}.csv"
outpath = "Plots/Teste SGD"
# ---------------


figSize = {"default": (6.7, 5), "wide": (13, 5)}


def read_input(filename):
    size = basSize**2
    df = pd.read_csv(filename, comment="#", index_col=0, header=None)
    df = df.astype(int)
    arr = df.to_numpy()
    return np.reshape(arr, (arr.shape[0],size,size))


if __name__ == "__main__":
    conn = read_input(inputfilename)
    # print(conn[-1])  # Verificação parece ok!

    iterations = conn.shape[0]
    size = conn.shape[1]

    # Degree information graphs
    gh_max = np.zeros(iterations)
    gh_med = np.zeros(iterations)
    gh_min = np.zeros(iterations)

    gx_max = np.zeros(iterations)
    gx_med = np.zeros(iterations)
    gx_min = np.zeros(iterations)

    for i in range(iterations):
        gh = [sum(conn[i, r, :]) for r in range(size)]
        gh_med[i] = sum(gh)/len(gh)
        gh_min[i] = min(gh)
        gh_max[i] = max(gh)

        gx = [sum(conn[i, :, c]) for c in range(size)]
        gx_med[i] = sum(gx)/len(gx)
        gx_min[i] = min(gx)
        gx_max[i] = max(gx)

    fig, ax = plt.subplots(nrows=2, ncols=1, sharex='col', figsize=figSize["default"])
    fig.suptitle(f"Unit's degree for CD-{k}, p = {p}")

    ax[0].plot(gh_max, label="Maximum")
    ax[0].plot(gh_med, label="Mean")
    ax[0].plot(gh_min, label="Minimum")
    ax[0].set_ylabel("Degree of hidden unit")
    ax[0].grid(color="gray", linestyle=":", linewidth=.2)
    ax[0].legend(loc="upper right", prop={'size': 6})

    ax[1].plot(gx_max, label="Maximum")
    ax[1].plot(gx_med, label="Mean")
    ax[1].plot(gx_min, label="Minimum")
    ax[1].set_ylabel("Degree of visible unit")
    ax[1].grid(color="gray", linestyle=":", linewidth=.2)
    ax[1].legend(loc="upper right", prop={'size': 6})

    plt.xlabel("Iteration")

    # plt.show()
    plt.savefig(f"{outpath}/nodeDegree_bas{basSize}_SGD_CD-{k}_p{p}_seed{seed}.pdf", transparent=True)


    # Edge activation graphs
    activation = [[sum(conn[:, r, c])/float(iterations) for c in range(size)] for r in range(size)]

    spacingX = 20
    spacingY = 100

    cmapname = "copper"  # "Greys"
    cmap = cm.get_cmap(cmapname)

    fig, ax = plt.subplots(1, figsize=figSize["wide"])
    ax.set_aspect('equal')
    plt.title(f"Activation pattern through training for CD-{k}, p = {p}")

    for r in range(size):
        for c in range(size):
            plt.plot([r*spacingX, c*spacingX], [spacingY, 0], color=cmap(activation[r][c]), linewidth=1, alpha=0.85)

    for i in range(size):
        c_h = plt.Circle( (spacingX*i, spacingY), spacingX/4, color='grey', zorder=5 )
        ax.add_patch(c_h)
        plt.annotate(r"$h_{{ {} }}$".format(i), (spacingX*i, spacingY), ha="center", va="center", zorder=10)

        c_x = plt.Circle( (spacingX*i, 0), spacingX/4, color='firebrick', zorder=5 )
        ax.add_patch(c_x)
        plt.annotate(r"$x_{{ {} }}$".format(i), (spacingX*i, 0), ha="center", va="center", zorder=10)

    plt.tick_params(
        axis='both',
        which='major',
        bottom=False,
        left=False,
        labelbottom=False,
        labelleft=False
    )

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(cm.ScalarMappable(cmap=cmapname), cax=cax)

    #fig.tight_layout()
    # plt.show()
    plt.savefig(f"{outpath}/edgeActivation_bas{basSize}_SGD_CD-{k}_p{p}_seed{seed}.pdf", transparent=True)
