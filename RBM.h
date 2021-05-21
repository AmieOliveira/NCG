//
// Created by Amanda Oliveira on 04/05/21.
//

#ifndef TESE_CÓDIGO_RBM_H
#define TESE_CÓDIGO_RBM_H

// Colors
#define RESET   "\033[0m"
#define RED     "\033[31m"
#define GREEN   "\033[32m"
#define YELLOW  "\033[33m"
#define BOLDRED     "\033[1m\033[31m"
#define BOLDGREEN   "\033[1m\033[32m"
#define BOLDYELLOW  "\033[1m\033[33m"

// Libraries
#include <iostream>
#include <random>
#include <vector>
#include "Eigen/Dense"
#include "Data.h"

using namespace Eigen;
using namespace std;

enum SampleType {
    CD,
    //PCD,
    //FPCD,
    //PT
};

class RBM {
    // Flags
    bool initialized;   // True if RBM has dimensions
    bool patterns;      // True if connectivity patterns are active
    bool hasSeed;       // True if a seed for the random
                        //      generator has been set

    // Dimensions
    int xSize, hSize;

    // RBM units
    VectorXd x;     // Visible units (binary)
    VectorXd h;     // Hidden units (binary)

    // RBM parameters
    VectorXd d;     // Visible units' biases
    VectorXd b;     // Hidden units' biases
    MatrixXd W;     // Connections' weights
    MatrixXd A;     // Connection pattern (must be binary!)

    MatrixXd C;     // Resulting weight matrix

    //int sampleType;

    // Sampling attributes and methods
    //unsigned genSeed;
    mt19937 generator;
    uniform_real_distribution<double>* p_dis;

    VectorXd sampleXfromH();
    VectorXd sampleHfromX();

    vector<VectorXd> sampleXtilde(SampleType sType, int k, //int b_size,
                                  vector<VectorXd> vecs);

public:
    // Constructors
    RBM();
    RBM(int X, int H);
    RBM(int X, int H, bool use_pattern);

    // Connectivity (de)ativation
    void connectivity(bool activate);

    // Set Dimentions
    void setDimensions(int X, int H);
    void setDimensions(int X, int H, bool use_pattern);

    // Set and Get variables
    int setVisibleUnits(VectorXd vec);
    //int setVisibleUnits(int* vec);
    VectorXd getVisibleUnits();

    int setHiddenUnits(VectorXd vec);
    //int setHiddenUnits(int* vec);
    VectorXd getHiddenUnits();

    VectorXd getVisibleBiases();
    //int setVisibleBiases(/* TODO */);

    VectorXd getHiddenBiases();
    //int setHiddenBiases(/* TODO */);

    MatrixXd getWeights();
    int setWeights(MatrixXd mat);

    MatrixXd getConnectivity();
    int setConnectivity(MatrixXd mat);

    // Random generator functions
    void setRandomSeed(unsigned seed);

    // Training methods
    void fit();
    // TODO: Argumentos do treinamento: Devo setar antes e só dar os dados? Ou dar tudo aqui?
    // TODO: Retornar alguma coisa na função?

    // Test Functions
    void printVariables();
    //double getRandomNumber();
    void sampleXH();
};

// Auxiliares
void printError(string msg);
void printWarning(string msg);
void printInfo(string msg);

#endif //TESE_CÓDIGO_RBM_H
