# Datasets downloaded from https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/

"""
    Script to create data files for RBMs from LIBSVM format files
    >> Assumes that it is a classification dataset, and each sample has a label
"""


import argparse
import os
import random


parser = argparse.ArgumentParser()

parser.add_argument("-i", "--input", type=str, required=True,
                    help="Input file (with path, if necessary)")
parser.add_argument("-o", "--output", type=str, required=True,
                    help="Output file name")

parser.add_argument("-n", "--name", type=str, default="",
                    help="Dataset name")
parser.add_argument("--split", type=str, default="all data",
                    help="Use this argument if this is only the "
                         "train/test/validation set")

parser.add_argument("-s", "--number-samples", type=int, required=True,
                    help="Number of samples in the dataset. \n"
                         "WARNING: if flag --shuffle-split is active, this "
                         "will be the number of samples in the train set")
parser.add_argument("-c", "--number-classes", type=int, required=True,
                    help="Number of classes of data in the dataset")
parser.add_argument("-d", "--data-size", type=int, required=True,
                    help="Size of each sample, e.g. number of features")

parser.add_argument("-0", "--first0", action="store_true",
                    help="Set this flag if the class numbers start in 0. "
                         "Otherwise, code assumes the fiest class is 1")
parser.add_argument("--set-start", type=int, default=0,
                    help="Use to split datasets without creating a new file."
                         "Use samples from the index given onward")

parser.add_argument("--shuffle-split", action="store_true",
                    help="Use flag if you wish to shuffle the data samples order.\n"
                         "It disables the --set-start argument")
parser.add_argument("--rseed", type=int, default=0,
                    help="Set random seed for shuffling order of samples")

# TODO: Add arguments for class names



def read_data_sample(line):
    words = line.split(" ")

    ret = f"Label: {int(words[0]) - bias}\n"

    idx = 1
    next_true = int(words[idx].split(":")[0]) - 1

    for i in range(datasize):
        if i == next_true:
            ret += "1 "
            idx += 1
            if words[idx] != "\n":
                next_true = int(words[idx].split(":")[0]) - 1
            else:
                next_true = None
        else:
            ret += "0 "

    ret += "\n"

    return ret



if __name__ == "__main__":
    args = parser.parse_args()

    split = args.split

    bias = 0 if args.first0 else 1
    shuf = args.shuffle_split

    dataset = args.name
    if not dataset:
        dataset = args.output

    extra = "" if (split == "all data" or shuf) else f"-{split}"

    ifname = args.input

    ofname = args.output + extra + ".data"

    num_examples = args.number_samples
    count = 0

    num_classes = args.number_classes

    datasize = args.data_size

    start = args.set_start

    data_samples = []

    with open(ifname, "r") as ifile:
        l_idx = -1

        while True:
            line = ifile.readline()
            l_idx += 1

            if not line:
                break
            if not shuf:
                if count >= num_examples:
                    break
                if l_idx < start:
                    continue

            data_samples += [read_data_sample(line)]

            if shuf:
                count += 1
            elif l_idx >= start:
                count += 1


    if count < num_examples:
        print(f"WARNING: There were less samples than expected. Expected"
              f" {num_examples}, but your file has only {count} samples.")


    if shuf:
        random.seed(args.rseed)
        random.shuffle(data_samples)

        data_samples = {
                        "train": data_samples[:num_examples],
                        "test": data_samples[num_examples:]
                        }
        nsamples = {"train": num_examples, "test": count-num_examples}

        for dset in ["train", "test"]:
            ofname = args.output + "-" + dset + ".data"
            with open(ofname, "w") as ofile:
                ofile.write(f"Name: {dataset.upper()}\n")
                ofile.write(f"Split: {dset} set\n")
                ofile.write(f"Number of examples: {nsamples[dset]}\n")
                ofile.write(f"Example size: {datasize}\n")
                ofile.write(f"Has labels: Yes\n")
                ofile.write(f"Number of labels: {num_classes}\n")
                ofile.write(f"Data shuffled with seed: {args.rseed}\n")
                ofile.write("\n")

                for line in data_samples[dset]:
                    ofile.write(line)

    else:
        with open(ofname, "w") as ofile:
            ofile.write(f"Name: {dataset.upper()}\n")
            ofile.write(f"Split: {split} set\n")
            ofile.write(f"Number of examples: {count}\n")
            ofile.write(f"Example size: {datasize}\n")
            ofile.write(f"Has labels: Yes\n")
            ofile.write(f"Number of labels: {num_classes}\n")

            ofile.write("\n")

            for line in data_samples:
                ofile.write(line)



