/*
 *  Script to train RBMs with complete graphs
 *  Author: Amanda
 *
 */

#include "RBM.h"
#include "basics.h"

#include <fstream>
#include <stdlib.h>
//#include <chrono>

using namespace std;

// TODO: Add parser and logger



int main(int argc, char **argv) {
    stringstream msg;

    int fileIDX;
    if (argc < 2) {
        msg << "Error! Must have file Idx to save output! Enter it as first argument.";
        printError(msg.str());
        throw runtime_error(msg.str());
    } else {
        fileIDX = atoi(argv[1]);
        msg.str("");
        msg << "File " << fileIDX;
        printInfo(msg.str());
    }

    unsigned seed = 18763258;
    if (argc >= 3) {
        seed = atoi(argv[2]);
        msg.str("");
        msg << "Setting seed as: " << seed;
        printInfo(msg.str());
    }

    int size = 4;
    if (argc >= 4) {
        size = atoi(argv[3]);
        msg.str("");
        msg << "Setting BAS size as: " << size;
        printInfo(msg.str());
    }

    int k = 10;
    if (argc >= 5) {
        k = atoi(argv[4]);
        msg.str("");
        msg << "Setting number of sample steps: " << k;
        printInfo(msg.str());
    }

    int iter = 6000;
    if (argc >= 6) {
        iter = atoi(argv[5]);
        msg.str("");
        msg << "Setting number of iterations: " << iter;
        printInfo(msg.str());
    }

    Data bas(DataDistribution::BAS, size);
    int s_size = bas.get_sample_size();

    RBM model(s_size, s_size);

    model.setRandomSeed(seed);
    model.trainSetup(SampleType::CD, k, iter, 5, 0.1, true);
    model.fit(bas);

    vector<double> h = model.getTrainingHistory();

    ofstream outdata;
    stringstream fname;
    fname << "nll_progress_complete_k" << k << "-run" << fileIDX << ".csv";
    outdata.open(fname.str()); // opens the file
    if( !outdata ) { // file couldn't be opened
        cerr << "Error: file could not be opened" << endl;
        exit(1);
    }

    outdata << "# NLL through RBM training for BAS" << size << ". CD-" << k << " All weights." << endl;
    outdata << "NLL" << endl;
    for (auto i: h)
        outdata << i << endl;
    outdata.close();

    model.printVariables();

    return 0;

}