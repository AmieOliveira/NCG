//
// Created by Amanda Oliveira on 24/05/21.
//

#ifndef BASICS_H
#define BASICS_H

namespace colors {
    // Colors
    #define RESET   "\033[0m"
    #define RED     "\033[31m"
    #define GREEN   "\033[32m"
    #define YELLOW  "\033[33m"
    #define BOLDRED     "\033[1m\033[31m"
    #define BOLDGREEN   "\033[1m\033[32m"
    #define BOLDYELLOW  "\033[1m\033[33m"
}

#include <iostream>
#include <string>
#include <vector>
//#include <Python.h>

#include "Eigen/Dense"

using namespace std;
using namespace colors;

// Log messages
void printError(string msg);
void printWarning(string msg);
void printInfo(string msg);

// Connectivity Matrixes
Eigen::MatrixXd n_neightbors(int nRows, int nCols, int nNeighs);
Eigen::MatrixXd bas_connect(int basSize);

//// Plotting functions
//void plotVectorPython(vector<double>);

#endif //BASICS_H
