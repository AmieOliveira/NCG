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

> TODO: How to run the code...

Dependencies: This repository needs to have the Eigen 
library installed (tested with Eigen3). 
The path to the Eigen library must be given during 
compilation.

The script ``create_executable.sh`` is designed to 
compile this program, and you must change the Eigen 
library path to suit your computer's instalation.

Note: Code developed in MacOS, with compiler Clang 
(version 12.0.5), using C++ 14. Tested for Linux...vim  
> TODO