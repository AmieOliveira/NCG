# References on tensorflow website:
# https://www.tensorflow.org/datasets/catalog/mnist
# https://www.tensorflow.org/datasets/catalog/binarized_mnist

"""
    Script for binarization of MNIST dataset
"""

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from random import random, seed

dataset = "mnist"
split = "test"  # "train"

ds, info = tfds.load(dataset, split=split, with_info=True)
# ds, info = tfds.load('binarized_mnist', split='train', with_info=True)

print(info)
# fig = tfds.show_examples(ds, info, rows=5, cols=4)
# plt.show()

seed(684)   # Fixed seed for reproducibility
limite = 255.0
figsToPrint = []
outfile = f"bin_{dataset}-{split}.data"

f = open(outfile, "w")
f.write(f"Name: Binarized {dataset.upper()}\n")
f.write(f"Split: {split} set\n")
f.write(f"Number of examples: {info.splits[split].num_examples}\n")
f.write(f"Example size: {info.features['image'].shape[0]*info.features['image'].shape[1]*info.features['image'].shape[2]}\n")
f.write(f"Is 2D image: Yes\n")
f.write(f"Has labels: Yes\n")
f.write("\n")

idx = 0

for datum in tfds.as_numpy(ds):
    image = datum['image']
    label = datum['label']

    if idx in figsToPrint:
        print(f"{idx+1}th figure. Image label: {label}")

        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
        ax[0].imshow(image, cmap='gray')
        ax[0].set_title("Original Image")

    image = image/limite

    f.write(f"Label: {label}\n")
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            moeda = random()
            prob = image[i, j]
            if moeda <= prob:
                image[i, j] = 1
            else:
                image[i, j] = 0

            f.write(f"{int(image[i, j][0])} ")
    f.write("\n")

    if idx in figsToPrint:
        ax[1].imshow(image, cmap='gray')
        ax[1].set_title("Binary Image")
        plt.show()

    idx += 1

f.close()