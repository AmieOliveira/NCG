/*
 *  Script to train RBMs with graphs with connectivity pattern specifically
 *      (and manually) designed for the BAS Dataset. Hardcodes NLL calculation.
 *  Author: Amanda
 *
 */

#include "RBM.h"
#include "basics.h"

#include <fstream>
#include <stdlib.h>

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
    }
    msg.str("");
    msg << "Setting BAS size as: " << size;
    printInfo(msg.str());

    int version = 2;
    if (argc > 5) {
        version = atoi(argv[5]);
    }
    msg.str("");
    msg << "Setting BAS connect version " << version;
    printInfo(msg.str());

    int k = 10;
    if (argc > 6) {
        k = atoi(argv[6]);
    }
    msg.str("");
    msg << "Setting number of sample steps: " << k;
    printInfo(msg.str());

    int iter = 6000;
    if (argc > 7) {
        iter = atoi(argv[7]);
    }
    msg.str("");
    msg << "Setting number of iterations: " << iter;
    printInfo(msg.str());

    int b_size = 5;
    if (argc > 8) {
        b_size = atoi(argv[8]);
    }
    msg.str("");
    msg << "Setting batch size: " << b_size;
    printInfo(msg.str());

    double l_rate = 0.01;
    if (argc > 9) {
        l_rate = atof(argv[9]);
    }
    msg.str("");
    msg << "Setting learning rate: " << l_rate;
    printInfo(msg.str());

    int f_nll = 1;
    if (argc > 10) {
        f_nll = atoi(argv[10]);
    }
    msg.str("");
    msg << "Setting frequence of NLL calculation: " << f_nll;
    printInfo(msg.str());


    Data bas(DataDistribution::BAS, size);
    int s_size = bas.get_sample_size();
    MatrixXd connectivity;

    switch (version) {
        case 1:
            connectivity = bas_connect(size);
            break;

        case 2:         // axis connectivity
            connectivity = bas_connect_2(size);
            break;

        case 3:         // convolutional
            connectivity = bas_connect_3(size);
            break;

        default:
            msg.str("");
            msg << "BAS connect v" << version << " not implemented.";
            printError(msg.str());
            cerr << "Error: Invalid connectivity version. Aborting." << endl;
            exit(1);
    }

    RBM model(s_size, s_size);
    model.connectivity(true);
    model.setConnectivity(connectivity);

    model.setRandomSeed(seed);
    model.trainSetup(SampleType::CD, k, iter, b_size, l_rate, true, f_nll);
    model.fit(bas);

    vector<double> h = model.getTrainingHistory();

    ofstream outdata;
    stringstream fname;
    fname << filePath << "nll_progress_bas" << size << "_BASconV" << version << "_k" << k;
    fname << "-run" << fileIDX << ".csv";
    cout << "Saving output as " << fname.str() << endl;
    outdata.open(fname.str()); // opens the file
    if( !outdata ) { // file couldn't be opened
        cerr << "Error: file could not be opened" << endl;
        exit(1);
    }

    outdata << "# NLL through RBM training for BAS" << size << ". CD-" << k << ", Specialist pattern version " << version << "." << endl;
    outdata << "# Seed = " << seed << ", Batch size = " << b_size << " and learning rate of " << l_rate << endl;
    if (f_nll != 1) outdata << "# NLL calculated every " << f_nll << " iterations." << endl;
    outdata << ",NLL" << endl;
    for (int i=0; i<=(float(iter)/f_nll); i++) {
        outdata << i*f_nll << "," << h.at(i) << endl;
    }
    if ((iter % f_nll) != 0) outdata << iter-1 << "," << h.back() << endl;

    model.printVariables();

    return 0;

}