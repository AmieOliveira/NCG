//
// Data format and datasets to be used by the RBM
//

#include "Data.h"

// Constructors
Data::Data(MatrixXd mat) {
    _data = mat;
    _size = mat.rows();
    _n = mat.cols();
    hasSeed = false;
    hasLabels = false;

    //cout << _data << endl;
    //cout << "sample size: " << _size << endl;
    //cout << _n << " samples" << endl;
}

Data::Data(DataDistribution distr, int size) {
    hasSeed = false;
    createData(distr, size);
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

void Data::createData(DataDistribution distr, int size) {
    printInfo("Starting Data creation");

    switch (distr) {
        case DataDistribution::BAS:
        {
            _size = size*size;
            _n = pow(2, size + 1);
            _data = MatrixXd::Zero(_size,_n);
            hasLabels = false;

            vector<int> state(size, 0);
            _idx = 0;

            fill_bas(size, state);

            break;
        }
        default:
        {
            string errorMessage = "Data distribution not implemented";
            printError(errorMessage);
            throw runtime_error(errorMessage);
        }
    }
}

void Data::fill_bas(int n, vector<int> state) {
    if (n == 0) {
        if ( all_of( state.begin(), state.end(), [](int i){return (i==0);} ) ) {
            // Skips this samples, it's already zeros
            _idx = _idx + 2;
            return;
        }

        int s = state.size();
        for (int i = 0; i < s; i++) {
            for (int j = 0; j < s; j++) {
                // Horizontal
                _data(s*i + j, _idx) = state.at(i);

                // Vertical
                _data(s*j + i, _idx+1) = state.at(i);
            }
        }

        _idx = _idx + 2;
        return;
    }

    fill_bas(n-1, state);
    state.at(n-1) = abs(1 - state.at(n-1));
    fill_bas(n-1, state);
}

void Data::createData(DataDistribution distr, int size, int nSamples) {
    printInfo("Starting Data creation");

    switch (distr) {
        case DataDistribution::BAS:
            _size = size*size;
            _n = nSamples;
            _data = MatrixXd::Zero(_size,_n);
            hasLabels = false;

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

Data::Data(string filename) {
    // Constructor to get data set from a DATA file
    hasSeed = false;
    hasLabels = false;

    fstream datafile;
    datafile.open(filename.c_str(), ios::in);

    string line;
    string name;

    int idx = 0;
    while (getline(datafile, line)) {
         // cout << line << endl;

         if (line == "") break;
         else if (line.substr(0, 5) == "Name:") { name = line.substr(6); }
         else if (line.substr(0, 19) == "Number of examples:") {
            _n = atoi(line.substr(20).c_str());
         }
         else if (line.substr(0, 13) == "Example size:") {
            _size = atoi(line.substr(14).c_str());
         }
         else if (line == "Has labels: Yes") {
            hasLabels = true;
         }

         idx++;
    }
    stringstream msg;
    msg << "Dataset name is '" << name << "'. Has " << _n
        << " samples of size " << _size << ".";
    printInfo( msg.str() );

    _data = MatrixXd::Zero(_size,_n);
    if (hasLabels) {
        _labels = MatrixXi::Zero(1,_n);
    }

    // Filling data information
    int i, j;
    j = 0;

    while (getline(datafile, line)) {
        if (hasLabels) {
            if (line.substr(0, 6) == "Label:") {
               _labels(0, j) = atoi(line.substr(7).c_str());
            } else {
                stringstream ss(line);
                for (i = 0; i < _size; i++) {
                    ss >> _data(i, j);
                }
                j++;
            }
        } else {
            stringstream ss(line);
            for (i = 0; i < _size; i++) {
                ss >> _data(i, j);
            }
            j++;
        }
    }

    // cout << "LABELS:\n" << _labels.block(0,0,1,10) << endl;
    // cout << "DATA:\n" << _data.block(0,0,_size,10) << endl;

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

vector<Data> Data::separateTrainTestSets(double trainPercentage) {
    // Function generates two data objects, one with the train set
    //      and the other with the test set.
    // I am separating by the order the data already has, but
    //      variations could be added later (according to necessity)
    double limit = int(trainPercentage*_n);

    Data trainData(_data.block(0,0,_size,limit));
    Data testData(_data.block(0,limit,_size,_n-limit));
    vector<Data> datasets;
    datasets.push_back(trainData);
    datasets.push_back(testData);

    return datasets;
}