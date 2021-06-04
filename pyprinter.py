#!/usr/bin/python3

import pandas as pd
import matplotlib.pyplot as plt

k_values = [100, 20, 10, 5, 2, 1]
# [1, 2, 5, 10, 20, 100]

fig, ax = plt.subplots(1)

for k in k_values:
    filename = f"nll_progress_single_k{k}.csv"

    df = pd.read_csv(filename, comment="#")
    df = df.astype(float)
    df = df.rename(columns={"NLL": f"CD-{k}"})
    df.plot(ax=ax)  # alpha=0.7

plt.title("NLL evolution through training")
plt.xlabel("Iteration")
plt.ylabel("NLL")
plt.show()
