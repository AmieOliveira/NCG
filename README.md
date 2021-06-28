# Connectivity Patterns in Neural Networks

This project was designed to evaluate the effect of 
using different connectivity patterns in a neural 
network performance. More specifically, it works 
with Restricted Boltzmann Machines (RBM), a model 
that can be understood as a feedforward neural 
network. This is a relatively simple model, 
containing only two layers (a visible and a hidden 
layer) and thus found ideal to start the research 
on connectivity.

> TODO: Colocar explicação simplificada do que são RBMS? 
> Provavelmente o ideal é colocar um link...

This code is designed to train both RBMs parameters 
and the connectivity between hidden and visible units 
simultaneously.

> TODO: Explicar qual a inovação proposta, e explicar 
> o que esse código faz


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

Note: This code developed in MacOS Bir Sur, with compiler 
Clang (version 12.0.5), using C++ 14. Tested on Ubuntu 18,
with compiler GCC (version 7.5.0).
> TODO: verificar quais os OS do cluster


### 2. Running

> TODO: Give a reference for BAS dataset

One can run train traditional RBMs for BAS dataset with CD 
algorithm using the executable ``complete.exe``. To run it, 
one can simply type: 

```shell
./complete.exe IDX [OUT_PATH SEED BAS_SIZE K ITER H]
```

The possible arguments are defined as:
- **IDX:** Corresponds to the run identifier number. It is 
  important to assign different identifiers to run multiple 
  runs with the same characteristics without overwriting 
  the output files.
  
- **OUT_PATH:** The path where the output will be saved.

- **SEED:** The random seed to be used during training. It 
  is important to assign different seeds if you wish to run 
  multiple runs, so as to get different results at each run.
  
- **BAS_SIZE:** The number of rows/columns of the BAS 
  dataset used in training.
  
- **K:** The number of steps CD will go through during 
  training (it implements the CD-k algorithm).

- **ITER:** The number of iterations performed during 
  training.

- **H:** The number of hidden units of the trained RBM.(If 
  nothing is specified, the RBM is created with the same 
  number of visible and hidden units)

The first algorithm is the only required one, but so far 
there is no parser used to organize inputs, so to give one, 
all of the previous ones must be given as well.

> TODO: How to run the code with different connectivities?
> (And to optimize the connectivity, when implemented)

> Explanations of classes? Of how to create scripts? 

