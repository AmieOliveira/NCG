#!/bin/bash
# Command to compile code

# NOTE: You must either have a file "eigenPath" with the path to the Eigen library in this directory or switch this variable value with the path
eigenPath=$(<eigenPath)

g++ -std=c++14 -I$eigenPath main.cpp basics.cpp RBM.cpp Data.cpp -o main.exe
g++ -std=c++14 -I$eigenPath completeGraph.cpp basics.cpp RBM.cpp Data.cpp -o complete.exe

echo "----------> Done!"

# TODO: Add possibility of executing code too...
# Eu posso adicionar um argumento para compilar ou não,
# um para executar ou não e dar os argumentos para a execução,
# se for o caso (argumentos default, se não?)