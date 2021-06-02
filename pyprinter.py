#!/usr/bin/python3

import pandas as pd
import matplotlib.pyplot as plt

filename = "nll_progress.csv"

df = pd.read_csv(filename, comment="#")
df = df.astype(float)

df.plot()
plt.show()