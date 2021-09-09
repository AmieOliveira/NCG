//
// Data format and datasets to be used by the RBM
//

#ifndef DATA_H
#define DATA_H

#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <random>
#include <vector>
#include <cmath>
#include "basics.h"
#include "Eigen/Dense"

using namespace Eigen;
using namespace std;

enum DataDistribution {
    BAS,    // Bars and Stripes, from A. Fischer, C. Igel "Training restricted Boltzmann machines: An introduction"
};

class Data {
    // Data variables
    int _size;                  // Size of a data sample
    int _n;                     // Number of samples
    vector<VectorXd> _data;     // Data
    vector<int> _labels;        // Data labels, when they exist
    vector<int> _indexMap;      // Mapping of data indexes
    bool hasLabels;             // True if Data has the dataset labels
    int _nLabels;               // Number of different labels (needs to have labels)
    bool giveLabels;            // True if one wishes to get both data and labels
                                //   together as a single "data" (needs to have labels)

    // Random number variables
    bool hasSeed;
    mt19937 generator;
    uniform_real_distribution<double>* p_dis;

    // Methods to be called during initialization
    void createData(DataDistribution distr, int size);
    void createData(DataDistribution distr, int size, int nSamples);
    void loadData(string filename, bool labels);

    // Support data creation methods
    void fill_bas(int n, vector<int> state);

    // Manipulating the data
    void _shuffle();
    int _randomSample(int i);

public:
    // Constructors
    Data(MatrixXd mat);
    Data(vector<VectorXd> vec);
    Data(DataDistribution distr, int size);
    Data(DataDistribution distr, int size, int nSamples);
    Data(unsigned seed,
         DataDistribution distr, int size, int nSamples);
    Data(string filename);
    Data(string filename, bool labels);

    // Random auxiliars
    void setRandomSeed(unsigned seed);

    // Data statistics
    double marginal_relativeFrequence(int jdx);
    int get_number_of_samples();
    int get_sample_size();
    int get_number_of_labels();

    void joinLabels(bool join);

    // Sampling
    VectorXd& get_sample(int idx);
    vector<Data> separateTrainTestSets(double trainPercentage);

    int& get_label(int idx);
    VectorXd get_label_vector(int idx);

    // Ideia inicial. Depois posso adicionar k-fold...

    // Manipulating the data
    void shuffle();
    void shuffle(unsigned seed);

    void printData();
};


#endif //DATA_H
