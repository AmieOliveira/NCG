//
// Created by Amanda Oliveira on 20/05/21.
//

#ifndef TESE_CÓDIGO_DATA_H
#define TESE_CÓDIGO_DATA_H

#include <random>
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
    int _size;          // Size of a data sample
    int _n;             // Number of samples
    MatrixXd _data;     // Data

    bool hasSeed;
    mt19937 generator;
    uniform_real_distribution<double>* p_dis;

    // Create data (to be called during initialization
    void createData(DataDistribution distr, int size, int nSamples);
public:
    // Constructors
    //Data(char* data_path);
    //Data(MatrixXd mat);
    Data(DataDistribution distr, int size, int nSamples);
    Data(unsigned seed,
         DataDistribution distr, int size, int nSamples);

    // Random seed
    void setRandomSeed(unsigned seed);

    // Data statistics
    int get_number_of_samples();
    VectorXd get_sample(int idx);
    // Função de obter conjunto de amostras (slice)
    // Separação em pacotes direto? treino e teste...
    // Frequência relativa de cada marginal (e de todas)
};


#endif //TESE_CÓDIGO_DATA_H
