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
    g++ -std=c++14 -I$eigenPath BASconnectGraph.cpp basics.cpp RBM.cpp Data.cpp -o BAScon.exe
    g++ -std=c++14 -I$eigenPath sgd.cpp basics.cpp RBM.cpp Data.cpp -o sgd.exe

    if [ -n "$string" ] || [ -n "$string2" ] || [ -n "$string3" ]; then
      string="error"
    fi
else
    if [ "$FILE" = "main" ]; then
        OUT=$FILE
    else
        if [ "$FILE" = "completeGraph" ]; then
        OUT=complete
        else
            if [ "$FILE" = "vNeighborsGraph" ]; then
                OUT=neighbors
            else
                if [ "$FILE" = "BASconnectGraph" ]; then
                    OUT=BAScon
                else
                    if [ "$FILE" = "sgd" ]; then
                        OUT=sgd
                    else
                        echo "No viable script selected. Exiting."
                        exit 1
                    fi
                fi
            fi
        fi
    fi
    echo "Compiling $FILE.cpp into $OUT.exe"
    g++ -std=c++14 -I$eigenPath "$FILE.cpp" basics.cpp RBM.cpp Data.cpp -o "$OUT.exe"
fi

echo "----------> Done!"

# TODO: Melhorar estrutura do código e da compilação