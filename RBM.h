//
// Created by Amanda Oliveira on 04/05/21.
//

#ifndef TESE_CÓDIGO_RBM_H
#define TESE_CÓDIGO_RBM_H

// Libraries
#include <random>
#include <vector>
#include <cmath>
#include "Eigen/Dense"
#include "basics.h"
#include "Data.h"

using namespace Eigen;


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
    bool hasSeed;       // True if a seed for the random generator has been set
    bool trainReady;    // True if RBM has been setup for training
    bool isTrained;     // True if RBM has been trained on some dataset

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
    MatrixXd* p_W;

    // Sampling attributes and methods
    //unsigned genSeed;
    mt19937 generator;
    uniform_real_distribution<double>* p_dis;

    VectorXd sample_x();
    VectorXd sample_h();

    vector<VectorXd> sampleXtilde(SampleType sType, int k, //int b_size,
                                  vector<VectorXd> vecs);

    // Energy methods
    double energy(); // TODO
    double freeEnergy(); // FIXME: Should give parameters?

    double normalizationConstant();
    double normalizationConstant_effX();
    double partialZ(int n);
    double partialZ_effX(int n);

    // Training variables
    SampleType stype;       // training method
    int k_steps;            // gibbs sampling steps
    int n_iter;             // number of iterations over data
    int b_size;             // batch size
    double l_rate;          // learning rate
    bool calcNLL;           // flag to calculate NLL over iterations (or not)
    vector<double> history; // NLL

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
    int setVisibleBiases(VectorXd vec);

    VectorXd getHiddenBiases();
    int setHiddenBiases(VectorXd vec);

    void startBiases();     // Starting randomly. Do not think this will be used for actual training

    MatrixXd getWeights();
    int setWeights(MatrixXd mat);
    void startWeights();    // Starting randomly, but maybe will want to add choices

    MatrixXd getConnectivity();
    int setConnectivity(MatrixXd mat);

    // Random generator functions
    void setRandomSeed(unsigned seed);

    // RBM probabilities
    VectorXd getProbabilities_x();
    VectorXd getProbabilities_h();

    // Training methods
    void trainSetup();
    void trainSetup(bool NLL);
    void trainSetup(SampleType sampleType, int k, int iterations,
                    int batchSize, double learnRate, bool NLL);
    void fit(Data trainData);
    // TODO: Retornar alguma coisa na função?

    // Evaluation methods
    double negativeLogLikelihood(Data data);
    vector<double> getTrainingHistory();

    // Test Functions
    void printVariables();
    //double getRandomNumber();
    void sampleXH();
    void generatorTest();
    void validateSample(unsigned seed, int rep);
};

#endif //TESE_CÓDIGO_RBM_H
