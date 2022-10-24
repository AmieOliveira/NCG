/*
 * Script to train RBMs with the BAS model and 'NCG-H' training, that is, a training that
 *      optimizes (or attempts to optimize) the number of hidden units in the RBM.
 * Author: Amanda
 *
 */

#include "RBM.h"

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

    int b_size = 5;
    if (argc > 7) {
        b_size = atoi(argv[7]);
        msg.str("");
        msg << "Setting batch size: " << b_size;
        printInfo(msg.str());
    }

    double l_rate = 0.01;
    if (argc > 8) {
        l_rate = atof(argv[8]);
        msg.str("");
        msg << "Setting learning rate: " << l_rate;
        printInfo(msg.str());
    }

    double p = 1;
    if (argc > 9) {
        p = atof(argv[9]);
        msg.str("");
        msg << "Setting probability of edge in initialization: " << p;
        printInfo(msg.str());
    }

    int f_nll = 1;
    if (argc > 10) {
        f_nll = atoi(argv[10]);
        msg.str("");
        msg << "Setting frequence of NLL calculation: " << f_nll;
        printInfo(msg.str());
    }

    // Data creation
    Data bas(DataDistribution::BAS, size);
    int s_size = bas.get_sample_size();
    int H = s_size;

    // Output file names
    stringstream fname;
    fname << "bas" << size << "_ncgh-" << p << "_H" << H << "_CD-" << k << "_lr"
          << l_rate << "_mBatch" << b_size << "_iter" << iter;
    if (seed != fileIDX) { fname << "_seed" << seed; }
    fname << "_run" << fileIDX;

    string c_filename, nll_filename;

    c_filename = filePath + "connectivity_" + fname.str() + ".csv";
    nll_filename = filePath + "nll_" + fname.str() + ".csv";
    printInfo("Ouput files: " + c_filename + ", " + nll_filename);

    // RBM creation
    RBM model(s_size, H, true);

    cout << "Nominal BAS size: " << size << endl;
    cout << "SIze of a BAS sample: " << bas.get_sample_size() << endl;
    cout << "Size: " << s_size << " x " << H << endl;
    model.printVariables();

    // RBM setup and training
    model.setRandomSeed(seed);
    model.trainSetup(SampleType::CD, k, iter, b_size, l_rate, true, f_nll);
    model.optSetup(Heuristic::SGD, c_filename, p, 0);

    model.fit_H(bas);

    vector<double> h = model.getTrainingHistory();

    // Saving NLL output
    ofstream outdata;
    outdata.open(nll_filename); // opens the file
    if( !outdata ) { // file couldn't be opened
        cerr << "Error: file could not be opened" << endl;
        exit(1);
    }
    outdata << "# NLL through RBM training for BAS " << size << ". CD-" << k
            << " with optimization of the number of hidden neurons (p = " << p << ")." << endl
            << "# Seed = " << seed << ", Batch size = " << b_size
            << " and learning rate of " << l_rate << endl;
    if (f_nll != 1) outdata << "# NLL calculated every " << f_nll << " iterations." << endl;

    outdata << ",NLL" << endl;
    for (int i=0; i<=(float(iter)/f_nll); i++) {
        outdata << i*f_nll << "," << h.at(i) << endl;
    }
    if ((iter % f_nll) != 0) outdata << iter-1 << "," << h.back() << endl;
    outdata.close();

    // Final printing for results verification
    model.printVariables();

    string rbmname = filePath + fname.str() + ".rbm";
    model.save(rbmname);

    return 0;
}
