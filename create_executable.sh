#!/bin/bash
# Command to compile code

g++ -std=c++14 -I/usr/local/include/eigen3/ main.cpp basics.cpp RBM.cpp Data.cpp -o main.exe

# TODO: Add possibility of executing code too...
# Eu posso adicionar um argumento para compilar ou não,
# um para executar ou não e dar os argumentos para a execução,
# se for o caso (argumentos default, se não?)