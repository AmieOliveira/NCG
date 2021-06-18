//
// Data format and datasets to be used by the RBM
//

#ifndef DATA_H
#define DATA_H

#include <random>
#include <vector>
#include <cmath>
#include "basics.h"
#include "Eigen/Dense"

using namespace Eigen;
using namespace std;

// IDEIA: Posso criar um DataGenerator, para gerar dados de BAS,
//        por exemplo, mas de outras distribuições tb. Ai serviria
//        como input para Data (ou Data criaria um DataGenerator
//        se eu criasse com um construtor específico...)

enum DataDistribution {
    BAS,    // Bars and Stripes, from A. Fischer, C. Igel "Training restricted Boltzmann machines: An introduction"
};

class Data {
    // Data variables
    int _size;          // Size of a data sample
    int _n;             // Number of samples
    MatrixXd _data;     // Data

    // Random number variables
    bool hasSeed;
    mt19937 generator;
    uniform_real_distribution<double>* p_dis;

    // Create data (to be called during initialization
    void createData(DataDistribution distr, int size);
    void createData(DataDistribution distr, int size, int nSamples);

    // Support data creation attributes and methods
    int _idx;
    void fill_bas(int n, vector<int> state);
public:
    // Constructors
    //Data(char* data_path);
    Data(MatrixXd mat);
    Data(DataDistribution distr, int size);
    Data(DataDistribution distr, int size, int nSamples);
    Data(unsigned seed,
         DataDistribution distr, int size, int nSamples);

    // Random auxiliars
    void setRandomSeed(unsigned seed);

    // Data statistics
    double marginal_relativeFrequence(int jdx);
    int get_number_of_samples();
    int get_sample_size();

    // Sampling
    VectorXd get_sample(int idx);
    vector<VectorXd> get_batch(int idx, int size);
    vector<Data> separateTrainTestSets(double trainPercentage);
    // Ideia inicial. Depois posso adicionar k-fold...
};


#endif //DATA_H
