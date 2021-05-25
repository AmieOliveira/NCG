//
// Created by Amanda Oliveira on 20/05/21.
//

#include "Data.h"

Data::Data(DataDistribution distr, int size, int nSamples) {
    random_device rd;
    generator.seed(rd());
    p_dis = new uniform_real_distribution<double>(0.0, 1.0);
    hasSeed = true;

    printInfo("Starting Data creation");

    createData(distr, size, nSamples);
}

Data::Data(unsigned seed, DataDistribution distr, int size, int nSamples) {
    generator.seed(seed);
    p_dis = new uniform_real_distribution<double>(0.0, 1.0);
    hasSeed = true;

    printInfo("Starting Data creation");

    createData(distr, size, nSamples);
}

void Data::createData(DataDistribution distr, int size, int nSamples) {
    switch (distr) {
        case DataDistribution::BAS:
            _size = size*size;
            _n = nSamples;
            _data = MatrixXd::Zero(_size,_n);

            bool orientation;
            int state;

            for (int s = 0; s < _n; ++s) {
                orientation = ( (*p_dis)(generator) < 0.5 );

                for (int i = 0; i < size; ++i) {
                    state = int( (*p_dis)(generator) < 0.5 );

                    for (int j = 0; j < size; ++j) {
                        if (orientation) {  // Horizontal
                            _data(4 * i + j, s) = state;
                        }
                        else {  // Vertical
                            _data(4 * j + i, s) = state;
                        }
                    }
                }
            }
            break;
        default:
            string errorMessage = "Data distribution not implemented";
            printError(errorMessage);
            throw runtime_error(errorMessage);
    }
}

void Data::setRandomSeed(unsigned int seed) {
    hasSeed = true;
    generator.seed(seed);
    p_dis = new uniform_real_distribution<double>(0.0, 1.0);
}

int Data::get_number_of_samples() {
    return _n;
}

VectorXd Data::get_sample(int idx) {
    return _data.col(idx);
}