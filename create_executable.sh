#!/bin/bash
# Command to compile code

# NOTE: Change Eigen path here as needed
eigenPath=/usr/local/include/eigen3/

g++ -std=c++14 -I$eigenPath main.cpp basics.cpp RBM.cpp Data.cpp -o main.exe
g++ -std=c++14 -I$eigenPath completeGraph.cpp basics.cpp RBM.cpp Data.cpp -o complete.exe

echo "----------> Done!"

# TODO: Can I automatically extract eigenpath from a file? (Should I?)
# TODO: Add possibility of executing code too...
# Eu posso adicionar um argumento para compilar ou não,
# um para executar ou não e dar os argumentos para a execução,
# se for o caso (argumentos default, se não?)