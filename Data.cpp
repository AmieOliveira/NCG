//
// Created by Amanda Oliveira on 20/05/21.
//

#include "Data.h"

// Constructors
Data::Data(MatrixXd mat) {
    _data = mat;
    _size = mat.rows();
    _n = mat.cols();
    hasSeed = false;

    //cout << _data << endl;
    //cout << "sample size: " << _size << endl;
    //cout << _n << " samples" << endl;
}

Data::Data(DataDistribution distr, int size, int nSamples) {
    random_device rd;
    generator.seed(rd());
    p_dis = new uniform_real_distribution<double>(0.0, 1.0);
    hasSeed = true;

    createData(distr, size, nSamples);
}

Data::Data(unsigned seed, DataDistribution distr, int size, int nSamples) {
    generator.seed(seed);
    p_dis = new uniform_real_distribution<double>(0.0, 1.0);
    hasSeed = true;

    createData(distr, size, nSamples);
}

void Data::createData(DataDistribution distr, int size, int nSamples) {
    printInfo("Starting Data creation");

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
                            _data(size * i + j, s) = state;
                        }
                        else {  // Vertical
                            _data(size * j + i, s) = state;
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

// Random auxiliars
void Data::setRandomSeed(unsigned int seed) {
    hasSeed = true;
    generator.seed(seed);
    p_dis = new uniform_real_distribution<double>(0.0, 1.0);
}

// Data statistics
double Data::marginal_relativeFrequence(int jdx) {
    // Can also compute as "_data.row(jdx).sum()/_n"
    return _data.row(jdx).mean();
}

int Data::get_number_of_samples() {
    return _n;
}

int Data::get_sample_size() {
    return _size;
}

// Sampling
VectorXd Data::get_sample(int idx) {
    return _data.col(idx);
}

vector<VectorXd> Data::get_batch(int idx, int size) {
    int init = idx*size;
    vector<VectorXd> batch;
    for (int k = init; k < init+size; ++k) {
        if (k >= _n) break;
        batch.push_back(_data.col(k));
    }
    return batch;
}

Data* Data::separateTrainTestSets(double trainPercentage) {
    // Function generates two data objects, one with the train set
    //      and the other with the test set.
    // I am separating by the order the data already has, but
    //      variations could be added later (according to necessity)
    double limit = int(trainPercentage*_n);

    Data trainData(_data.block(0,0,_size,limit));
    Data testData(_data.block(0,limit,_size,_n-limit));
    Data datasets[2] = {trainData, testData};

    return datasets;
}