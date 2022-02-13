# Network Connectivity Gradient (NCG)

This project was designed to evaluate the effect of 
using different connectivity patterns in a neural 
network performance, and optimize the network 
connectivity. More specifically, it works 
with Restricted Boltzmann Machines (RBM), a model 
that can be understood as a feedforward neural 
network. This is a relatively simple model, 
containing only two layers (a visible and a hidden 
layer) and thus found ideal to start the research 
on connectivity.

This code is designed to train both RBMs parameters 
and the connectivity between hidden and visible units 
simultaneously.


## How to run the code

### 1. Compilation

**Dependencies:** This repository needs to have the Eigen 
library installed (tested with Eigen3). 
The path to the Eigen library must be given during 
compilation.

The script ``create_executable.sh`` is designed to 
compile this program. To run it, you should first create 
a file in your directory called ``eigenPath`` that 
contains the path to the Eigen library. This can be made 
running the command:

```shell
echo "/path/to/eigen/" >> eigenPath
```

Changing ``/path/to/eigen`` with your own computer path.
After that, your computer should be set to go, and you 
can compile the code:

```shell
source create_executable.sh
```

This command should compile all scripts, but you can also 
stipulate a single script to be compiled with the flag 
``-s``, adding the paramter ``-s 'main'``, for example, to 
compile file ``main.cpp``.

**Note:** This code developed in MacOS Bir Sur, with compiler 
Clang (version 12.0.5), using C++ 14. Tested on Ubuntu 18,
with compiler GCC (version 7.5.0).


### 2. Running

There are 6 available scripts used to train the RBMs: 
  - ``completeGraph.cpp``: trains traditional RBMs with the BAS dataset
  - ``vNeighborsGraph.cpp``: trains RBMs with ``v``-neighbors patterns 
    (you can specify a connectivity pattern of ``v`` neighbors organized 
    in a line or in a spiral) for BAS dataset
  - ``BASconnectGraph.cpp``: trains RBMs with some specific, manually 
    designed patters for the BAS dataset
  - ``sgd.cpp``: trains RBMs with the NCG method (optimizes connectivity) 
    for the BAS dataset
  - ``mnist.cpp``: trains RBMs with the MNIST dataset, for the traditional 
    RBM, with ``v``-neighbors patterns or with the NCG method
  - ``mnist_acc.cpp``: trains RBMs with the MNIST dataset for the 
    classification task

#### Usage example:
One can run train traditional RBMs for BAS dataset with CD 
algorithm using the executable ``complete.exe``. To run it, 
simply type: 

```shell
./complete.exe IDX [OUT_PATH SEED BAS_SIZE K ITER B_SIZE L_RATE F_NLL H]
```

The arguments are defined as:
- **IDX:** Corresponds to the run identifier number. It is 
  important to assign different identifiers to run multiple 
  runs with the same characteristics without overwriting 
  the output files.
  
- **OUT_PATH:** The path where the outputs will be saved.

- **SEED:** The random seed to be used during training. It 
  is important to assign different seeds if you wish to run 
  multiple runs, so as to get different results at each run.
  
- **BAS_SIZE:** The number of rows/columns of the BAS 
  dataset used in training.
  
- **K:** The number of steps CD will go through during 
  training (it implements the CD-K algorithm).

- **ITER:** The number of iterations performed during 
  training.
  
- **B_SIZE:** The size of the mini-batch to be used in 
  training (number of samples used for each parameter 
  update).
  
- **L_RATE:** The training algorithm learning rate. So far 
  the code supports only training with a fixed learning 
  rate.

- **H:** The number of hidden units of the trained RBM.(If 
  nothing is specified, the RBM is created with the same 
  number of visible and hidden units)

The first argument is the only required one, but it is 
recommended to define all to suit your needs. So far 
there is no parser used to organize inputs, so to give an 
optional argument, all of the previous ones must be given 
as well.

You can, of course, create your own script to train and use 
the RBMs as you see fit. Check out the available scripts as 
a base to see the commands used to set and train the RBM. 
The important aspects are that you create an RBM object, to 
which you assign your desired characteristics 
(visible and hidden layers sizes, training parameters, 
desired connectivity network, NCG or normal training, etc.) 
and a Data object, to which you add the dataset which will 
interact with the RBM. The dataset can be added through a 
text file, see the examples in the Datasets folder for 
format specification. 


### 3. Using the RBM

Once you have trained the RBM, you can use it as you see fit. 
There are several functions implemented for usage, such as 
sampling and prediction methods, but you can also export your 
RBM using the ``save`` method, which saves the weights and 
biases in a text file. 