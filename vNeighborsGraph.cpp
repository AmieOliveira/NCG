/*
 *  Script to train RBMs with graphs in which each unit has v neighbors.
 *      Hardcodes number of hidden units (H=X). Hardcodes NLL calculation
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

    string filePath = "./";
    if (argc > 2) {
        filePath = argv[2];
        if ( !(filePath[filePath.length() - 1] == '/') ) filePath = filePath + "/";
    }
    msg.str("");
    msg << "File directory: " << filePath;
    printInfo(msg.str());

    unsigned seed = 18763258;
    if (argc > 3) {
        seed = atoi(argv[3]);
    }
    msg.str("");
    msg << "Setting seed as: " << seed;
    printInfo(msg.str());

    int size = 4;
    if (argc > 4) {
        size = atoi(argv[4]);
        msg.str("");
        msg << "Setting BAS size as: " << size;
        printInfo(msg.str());
    }

    int v = 8;
    if (argc > 5) {
        v = atoi(argv[5]);
        msg.str("");
        msg << "Setting number of neighbors as: " << v;
        printInfo(msg.str());
    }

    int k = 10;
    if (argc > 6) {
        k = atoi(argv[6]);
        msg.str("");
        msg << "Setting number of sample steps: " << k;
        printInfo(msg.str());
    }

    int iter = 6000;
    if (argc > 7) {
        iter = atoi(argv[7]);
        msg.str("");
        msg << "Setting number of iterations: " << iter;
        printInfo(msg.str());
    }

    int f_nll = 1;
    if (argc > 8) {
        f_nll = atoi(argv[8]);
        msg.str("");
        msg << "Setting frequence of NLL calculation: " << f_nll;
        printInfo(msg.str());
    }

    string neighType = "line";  // Alternative should be "spiral"
    // TODO: Add spiral neighbors option. Will need to give setting as argument, change this variable accordingly and use the spiral pattern creation function
    // All still needs to be implemented

    Data bas(DataDistribution::BAS, size);
    int s_size = bas.get_sample_size();
    MatrixXd connectivity = v_neighbors(s_size, s_size, v);

    RBM model(s_size, s_size);
    model.connectivity(true);
    model.setConnectivity(connectivity);

    model.setRandomSeed(seed);
    model.trainSetup(SampleType::CD, k, iter, 5, 0.1, true, f_nll);
    model.fit(bas);

    vector<double> h = model.getTrainingHistory();

    ofstream outdata;
    stringstream fname;
    fname << filePath << "nll_progress_bas" << size << "_neighbors" << v << "_" << neighType << "_k" << k;
    fname << "-run" << fileIDX << ".csv";
    cout << "Saving output as " << fname.str() << endl;
    outdata.open(fname.str()); // opens the file
    if( !outdata ) { // file couldn't be opened
        cerr << "Error: file could not be opened" << endl;
        exit(1);
    }

    outdata << "# NLL through RBM training for BAS" << size << ". CD-" << k << ", " << v << " neighbors in line." << endl;
    outdata << "# Seed = " << seed << ", Batch size = " << b_size << " and learning rate of " << l_rate << endl;
    if (f_nll != 1) outdata << "# NLL calculated every " << f_nll << " iterations." << endl;
    //for (auto i: h)
    //    outdata << i << endl;
    //outdata.close();

    outdata << ",NLL" << endl;
    for (int i=0; i<=(float(iter)/f_nll); i++) {
        outdata << i*f_nll << "," << h.at(i) << endl;
    }
    if ((iter % f_nll) != 0) outdata << iter-1 << "," << h.back() << endl;

    model.printVariables();

    return 0;

}