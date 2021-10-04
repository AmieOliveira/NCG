//
// Helper functions
//

#ifndef BASICS_H
#define BASICS_H

namespace colors {
    // Colors
    #define RESET   "\033[0m"
    #define RED     "\033[31m"
    #define GREEN   "\033[32m"
    #define YELLOW  "\033[33m"
    #define GRAY    "\u001b[30;1m"
    #define BOLDRED     "\033[1m\033[31m"
    #define BOLDGREEN   "\033[1m\033[32m"
    #define BOLDYELLOW  "\033[1m\033[33m"
    #define BOLDGRAY    "\u001b[1m\u001b[30;1m"
}

#include <iostream>
#include <string>
#include <vector>
#include <random>
#include <cmath>
//#include <Python.h>

#include "Eigen/Dense"

using namespace std;
using namespace colors;

// Log messages
void printError(string msg);
void printWarning(string msg);
void printInfo(string msg);
void printVerbose(string msg);

// Connectivity Matrixes
Eigen::MatrixXd v_neighbors(int nRows, int nCols, int nNeighs);
Eigen::MatrixXd v_neighbors_spiral(int nRows, int nCols, int nNeighs);

Eigen::MatrixXd bas_connect(int basSize);
Eigen::MatrixXd bas_connect_2(int basSize);
Eigen::MatrixXd bas_connect_3(int basSize);
Eigen::MatrixXd bas_connect_4(int basSize);

Eigen::MatrixXd square_convolution(int X, int H, int n_labels);

// Helper functions
int wraparound(int x, int lim);

// Randomizing Connectivity
class Mixer {
public:
    unsigned seed;
    mt19937 generator;

    Mixer();
    Mixer(unsigned s);

    Eigen::MatrixXd mix_neighbors(Eigen::MatrixXd regPattern, int iter);
};

//// Plotting functions
//void plotVectorPython(vector<double>);

#endif //BASICS_H
