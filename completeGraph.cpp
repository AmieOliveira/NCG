/*
 *  Script to train RBMs with complete graphs. Hardcodes NLL calculation
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
    if (argc >= 3) {
        filePath = argv[2];
        if ( !(filePath[filePath.length() - 1] == '/') ) filePath = filePath + "/";
    }
    msg.str("");
    msg << "File directory: " << filePath;
    printInfo(msg.str());

    unsigned seed = 18763258;
    if (argc >= 4) {
        seed = atoi(argv[3]);
    }
    msg.str("");
    msg << "Setting seed as: " << seed;
    printInfo(msg.str());

    int size = 4;
    if (argc >= 5) {
        size = atoi(argv[4]);
        msg.str("");
        msg << "Setting BAS size as: " << size;
        printInfo(msg.str());
    }

    int k = 10;
    if (argc >= 6) {
        k = atoi(argv[5]);
        msg.str("");
        msg << "Setting number of sample steps: " << k;
        printInfo(msg.str());
    }

    int iter = 6000;
    if (argc >= 7) {
        iter = atoi(argv[6]);
        msg.str("");
        msg << "Setting number of iterations: " << iter;
        printInfo(msg.str());
    }

    int f_nll = 1;
    if (argc > 7) {
        f_nll = atoi(argv[7]);
        msg.str("");
        msg << "Setting frequence of NLL calculation: " << f_nll;
        printInfo(msg.str());
    }

    Data bas(DataDistribution::BAS, size);
    int s_size = bas.get_sample_size();

    int H = s_size;
    if (argc > 8) {
        H = atoi(argv[8]);
        msg.str("");
        msg << "Setting number of hidden neurons: " << H;
        printInfo(msg.str());
    }

    RBM model(s_size, H);

    model.setRandomSeed(seed);
    model.trainSetup(SampleType::CD, k, iter, 5, 0.1, true, f_nll);
    model.fit(bas);

    vector<double> h = model.getTrainingHistory();

    ofstream outdata;
    stringstream fname;
    fname << filePath << "nll_progress_complete_k" << k;
    if (H != s_size) fname << "_H" << H;
    fname << "-run" << fileIDX << ".csv";
    cout << "Saving output as " << fname.str() << endl;

    outdata.open(fname.str()); // opens the file
    if( !outdata ) { // file couldn't be opened
        cerr << "Error: file could not be opened" << endl;
        exit(1);
    }
    outdata << "# NLL through RBM training for BAS" << size << ". CD-" << k << " All weights." << endl;
    if (f_nll != 1) outdata << "# NLL calculated every " << f_nll << " iterations." << endl;
    //outdata << "NLL" << endl;
    //for (auto i: h)
    //    outdata << i << endl;

    outdata << ",NLL" << endl;
    for (int i=0; i<(float(iter)/f_nll); i++) {
        outdata << i*f_nll << "," << h.at(i) << endl;
    }
    if ((iter % f_nll) != 0) outdata << iter-1 << "," << h.back() << endl;

    outdata.close();

    model.printVariables();

    return 0;

}