#!/bin/bash
# Command to compile code
# One can give as an argument the name of the specific file they wish to compile (without ".cpp")

FILE="all"
OPTIMIZER=""

while getopts s:fh flag
do
    case "${flag}" in
        s) FILE=${OPTARG};;
        f) OPTIMIZER=-O3
           echo "Compiling with O3 flag, for faster code"
           ;;
        h) echo "Optional flags:"
           echo "-s SCRIPT, given when you want to compile a single script (the specified SCRIPT). If you don't give this argument all scripts will be compiled."
           echo "-f for fast code. Compiles script with O3 flag"
           echo "-h display this help"
           exit 0
           ;;
        *)
    esac
done

# NOTE: You must either have a file "eigenPath" with the path to the Eigen library in this directory or switch this variable value with the path
eigenPath=$(<eigenPath)


if [ "$FILE" = "all" ]; then
    echo "Compiling all scripts"
    g++ -std=c++14 -I$eigenPath $OPTIMIZER main.cpp basics.cpp RBM.cpp Data.cpp -o main.exe
    g++ -std=c++14 -I$eigenPath $OPTIMIZER completeGraph.cpp basics.cpp RBM.cpp Data.cpp -o complete.exe
    g++ -std=c++14 -I$eigenPath $OPTIMIZER vNeighborsGraph.cpp basics.cpp RBM.cpp Data.cpp -o neighbors.exe
    g++ -std=c++14 -I$eigenPath $OPTIMIZER BASconnectGraph.cpp basics.cpp RBM.cpp Data.cpp -o BAScon.exe
    g++ -std=c++14 -I$eigenPath $OPTIMIZER sgd.cpp basics.cpp RBM.cpp Data.cpp -o SGD.exe
    g++ -std=c++14 -I$eigenPath $OPTIMIZER mnist.cpp basics.cpp RBM.cpp Data.cpp -o mnist.exe
    g++ -std=c++14 -I$eigenPath $OPTIMIZER classificationStats.cpp basics.cpp RBM.cpp Data.cpp -o c_stats.exe
    g++ -std=c++14 -I$eigenPath $OPTIMIZER mnist_acc.cpp basics.cpp RBM.cpp Data.cpp -o accuracy.exe
    g++ -std=c++14 -I$eigenPath $OPTIMIZER timeNLL.cpp basics.cpp RBM.cpp Data.cpp -o times.exe
    g++ -std=c++14 -I$eigenPath $OPTIMIZER misc_acc.cpp basics.cpp RBM.cpp Data.cpp -o accuracy_all.exe
    g++ -std=c++14 -I$eigenPath $OPTIMIZER bas_ncgH.cpp basics.cpp RBM.cpp Data.cpp -o optHbas.exe

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
                        OUT=SGD
                    else
                        if [ "$FILE" = "mnist" ]; then
                            OUT=mnist
                        else
                            if [ "$FILE" = "classificationStats" ]; then
                                OUT=c_stats
                            else
                                if [ "$FILE" = "mnist_acc" ]; then
                                    OUT=accuracy
                                else
                                    if [ "$FILE" = "timeNLL" ]; then
                                        OUT="times"
                                    else
                                        if [ "$FILE" = "misc_acc" ]; then
                                            OUT=accuracy_all
                                        else
                                            if [ "$FILE" = "bas_ncgH" ]; then
                                                OUT="optHbas"
                                            else
                                                echo "No viable script selected. Exiting."
                                                exit 1
                                            fi
                                        fi
                                    fi
                                fi
                            fi
                        fi
                    fi
                fi
            fi
        fi
    fi
    echo "Compiling $FILE.cpp into $OUT.exe"
    g++ -std=c++14 -I$eigenPath $OPTIMIZER "$FILE.cpp" basics.cpp RBM.cpp Data.cpp -o "$OUT.exe"
fi

echo "----------> Done!"

# TODO: Melhorar estrutura do código e da compilação