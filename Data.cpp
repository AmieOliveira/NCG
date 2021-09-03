//
// Data format and datasets to be used by the RBM
//

#include "Data.h"

// Constructors
Data::Data(MatrixXd mat) {
    _size = mat.rows();
    _n = mat.cols();
    hasSeed = false;
    hasLabels = false;

    for (int i=0; i<_n; i++) {
        _data.push_back(mat.col(i));
        _indexMap.push_back(i);
    }
}

Data::Data(vector<VectorXd> vec) {
    _size = vec.at(0).size();
    _n = vec.size();
    hasSeed = false;
    hasLabels = false;

    _data = vec;

    for (int i=0; i<_n; i++) {
        _indexMap.push_back(i);
    }

    // for (auto s: _data) { cout << "Sample: " << s.transpose() << endl; }
    // cout << "sample size: " << _size << endl;
    // cout << _n << " samples" << endl;
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
            hasLabels = false;

            vector<int> state(size, 0);

            fill_bas(size, state);

            for (int i=0; i<_n; i++) {
                _indexMap.push_back(i);
            }

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
        VectorXd horizontal(_size), vertical(_size);

        int s = state.size();
        for (int i = 0; i < s; i++) {
            for (int j = 0; j < s; j++) {
                // Horizontal
                horizontal(s*i + j) = state.at(i);

                // Vertical
                vertical(s*j + i) = state.at(i);
            }
        }

        _data.push_back(horizontal);
        _data.push_back(vertical);



        return;
    }

    fill_bas(n-1, state);
    state.at(n-1) = abs(1 - state.at(n-1));
    fill_bas(n-1, state);
}

void Data::createData(DataDistribution distr, int size, int nSamples) {
    printInfo("Starting Data creation");

    VectorXd aux(_size);

    switch (distr) {
        case DataDistribution::BAS:
            _size = size*size;
            _n = nSamples;
            hasLabels = false;

            bool orientation;
            int state;
            int sIdx;

            sIdx = 0;
            for (int s = 0; s < _n; ++s) {
                orientation = ( (*p_dis)(generator) < 0.5 );

                aux = VectorXd::Zero(_size);

                for (int i = 0; i < size; ++i) {
                    state = int( (*p_dis)(generator) < 0.5 );

                    for (int j = 0; j < size; ++j) {
                        if (orientation) {  // Horizontal
                            aux(size * i + j) = state;
                        }
                        else {  // Vertical
                            aux(size * j + i) = state;
                        }
                    }
                }
                _data.push_back(aux);
                _indexMap.push_back( sIdx );
                sIdx++;
            }
            break;
        default:
            string errorMessage = "Data distribution not implemented";
            printError(errorMessage);
            throw runtime_error(errorMessage);
    }
}

Data::Data(string filename) {
    loadData(filename, false);
}

Data::Data(string filename, bool labels) {
    loadData(filename, labels);
}

void Data::loadData(string filename, bool labels) {
    // Constructor to get data set from a DATA file
    hasSeed = false;
    hasLabels = false;
    giveLabels = false;
    _nLabels = 0;

    fstream datafile;
    datafile.open(filename.c_str(), ios::in);

    string line;
    string name;

    while (getline(datafile, line)) {
        // cout << line << endl;

        if (line == "") break;
        else if (line.substr(0, 5) == "Name:") { name = line.substr(6); }
        else if (line.substr(0, 19) == "Number of examples:") {
            _n = atoi(line.substr(20).c_str());
        }
        else if (line.substr(0, 13) == "Example size:") {
            _size = atoi(line.substr(14).c_str());
        } if (line == "Has labels: Yes") {
            hasLabels = true;
        } else if (line.substr(0, 17) == "Number of labels:") {
            if (labels) { _nLabels = atoi(line.substr(18).c_str()); }
        }
    }
    stringstream msg;
    msg << "Dataset name is '" << name << "'. Has " << _n
        << " samples of size " << _size << ".";
    printInfo( msg.str() );

    bool noNLabels = false;
    if ( labels && hasLabels && (_nLabels == 0) ) {
        printWarning("Data set is said to have labels, but no number of labels is "
                     "specified. Input file should be rectified");
        noNLabels = true;
    }

    VectorXd aux(_size);

    // Filling data information
    int i, j=0;

    while (getline(datafile, line)) {
        if (hasLabels) {
            if (line.substr(0, 6) == "Label:") {
                if (labels) {
                    _labels.push_back( atoi(line.substr(7).c_str()) );
                    if (noNLabels) {
                        if ( _labels.back() > _nLabels ) { _nLabels = _labels.back(); }
                    }
                }
            } else {
                stringstream ss(line);
                for (i = 0; i < _size; i++) {
                    ss >> aux(i);
                }
                _data.push_back( aux );
                _indexMap.push_back( j );
                j++;
            }
        } else {
            stringstream ss(line);
            for (i = 0; i < _size; i++) {
                ss >> aux(i);
            }
            _data.push_back( aux );
            _indexMap.push_back( j );
            j++;
        }
    }

    if ( !labels ) hasLabels = false;

    cout << "Data extracted";
    if (hasLabels) { cout << " with labels"; }
    cout << endl;

    // if (hasLabels) cout << "LABELS:\n" << _labels.at(0) << ", " << _labels.at(1) << endl;
    // cout << "DATA:\n" << _data.at(0).transpose() << endl << endl << _data.at(1).transpose() << endl;
}

// Random auxiliars
void Data::setRandomSeed(unsigned int seed) {
    hasSeed = true;
    generator.seed(seed);
    p_dis = new uniform_real_distribution<double>(0.0, 1.0);
}

// Data statistics
double Data::marginal_relativeFrequence(int jdx) {
    if (jdx >= _size) {
        printError("Tried to calculate statistics for an inexistent data feature");
        cerr << "Data sample has only " << _size << " feature, and therefore one cannot compute "
             << "statistics for " << jdx << "-th feature" << endl;
        exit(1);
    }

    double sum = 0;
    for (int s=0; s<_n; s++) {
        sum += _data.at(s)(jdx);
    }

    return sum/_n;
}

int Data::get_number_of_samples() {
    return _n;
}

int Data::get_sample_size() {
    if (giveLabels) return _size + _nLabels;
    return _size;
}

int Data::get_number_of_labels() {
    if (hasLabels) { return _nLabels; }
    printError("Requested number of labels, but Data set has no labels");
    return 0;
}

void Data::joinLabels(bool join) {
    if (hasLabels) { giveLabels = join; }
    else { printError("Cannot output label values if data set has no labels"); }
}


// Sampling
VectorXd & Data::get_sample(int idx) {
    if (hasLabels && giveLabels) {
        VectorXd lab = VectorXd::Zero(_nLabels);
        lab( _labels.at( _indexMap.at(idx) ) ) = 1;

        static VectorXd ret(_size+_nLabels);
        ret << _data.at( _indexMap.at(idx) ), lab;

        return ret;
    }
    return _data.at( _indexMap.at(idx) );
}

int & Data::get_sample_label(int idx) {
    if (hasLabels) { return _labels.at( _indexMap.at(idx) ); }
    else {
        printError("Data set has no labels");
        cerr << "Requested a label sample, when data has no labels" << endl;
        exit(1);
    }
}

vector<Data> Data::separateTrainTestSets(double trainPercentage) {
    // Function generates two data objects, one with the train set
    //      and the other with the test set.
    // I am separating by the order the data already has, but
    //      variations could be added later (according to necessity)

    // FIXME: Tenho que resolver a questão dos labels!

    double limit = int(trainPercentage*_n);

    Data trainData( std::vector<VectorXd>(_data.begin(),_data.begin()+limit) );
    Data testData( std::vector<VectorXd>(_data.begin()+limit,_data.end()) );
    vector<Data> datasets;
    datasets.push_back(trainData);
    datasets.push_back(testData);

    return datasets;
}


// Manipulating the data
int Data::_randomSample(int i) {
    return int((*p_dis)(generator)*i);
}

void Data::_shuffle() {
    // Function generates a data permutation through the shuffling of the index mapping
    // Shuffling extracted from std::random_shuffle template
    // http://www.cplusplus.com/reference/algorithm/random_shuffle/
    auto first = _indexMap.begin();
    for (int i = _n-1; i > 0; --i) {
        swap(first[i], first[ _randomSample(i+1) ]);
    }
}

void Data::shuffle() {
    if ( !hasSeed ) {
        random_device rd;
        generator.seed(rd());
        p_dis = new uniform_real_distribution<double>(0.0, 1.0);
        hasSeed = true;
    }
    _shuffle();
}

void Data::shuffle(unsigned seed) {
    setRandomSeed(seed);
    _shuffle();
}

void Data::printData() {
    printInfo("Data samples");

    for (auto s: _data) {
        cout << s.transpose() << endl;
    }
}
