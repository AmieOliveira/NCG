#!/bin/bash
# Command to compile code
# One can give as an argument the name of the specific file they wish to compile (without ".cpp")

FILE="all"

while getopts s: flag
do
    case "${flag}" in
        s) FILE=${OPTARG};;
        *)
    esac
done

# NOTE: You must either have a file "eigenPath" with the path to the Eigen library in this directory or switch this variable value with the path
eigenPath=$(<eigenPath)

if [ "$FILE" = "all" ]; then
    echo "Compiling all scripts"
    g++ -std=c++14 -I$eigenPath main.cpp basics.cpp RBM.cpp Data.cpp -o main.exe
    g++ -std=c++14 -I$eigenPath completeGraph.cpp basics.cpp RBM.cpp Data.cpp -o complete.exe
    g++ -std=c++14 -I$eigenPath vNeighborsGraph.cpp basics.cpp RBM.cpp Data.cpp -o neighbors.exe

    if [ -n "$string" ] || [ -n "$string2" ] || [ -n "$string3" ]; then
      string="error"
    fi
else
    echo "Compiling $FILE.cpp"
    declare -a OUTPUTS
    OUTPUTS=( ["main"]="main" ["completeGraph"]="complete" ["vNeighborsGraph"]="neighbors" )
    g++ -std=c++14 -I$eigenPath "$FILE.cpp" basics.cpp RBM.cpp Data.cpp -o "${OUTPUTS[$FILE]}.exe"
fi

echo "----------> Done!"

# TODO: Add possibility of executing code too...
# Eu posso adicionar um argumento para compilar ou não,
# um para executar ou não e dar os argumentos para a execução,
# se for o caso (argumentos default, se não?)