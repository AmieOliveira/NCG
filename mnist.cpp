/*
 *  Script to train RBMs with MNIST dataset. One can choose between training a traditional
 *      RBM, to train connectivity using the SGD optimizer or to train RBM's with a fixed
 *      connectivity pattern from a set of available options. Hardcodes NLL calculation.
 *  Author: Amanda
 *
 */


 // TODO: Implement SGD and fixed pattern options

#include "RBM.h"
#include "basics.h"

#include <fstream>
#include <stdlib.h>
//#include <chrono>

using namespace std;

// TODO: Add parser and logger

enum TrainingTypes {
    complete,
    sgd,
    conv,
    neighLine,
    neighSpiral,
    none,           // For inexistent types
};
TrainingTypes resolveOption(string opt) {
    if (opt == "complete") return complete;
    if (opt == "sgd") return sgd;
    if (opt == "convolution") return conv;
    if (opt == "neighborsLine") return neighLine;
    if (opt == "neighborsSpiral") return neighSpiral;
    return none;
}


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

    string trainType = "complete";
    if (argc > 4) {
        trainType = argv[4];
    }
    msg.str("");
    msg << "Setting RBM type as: " << trainType;
    printInfo(msg.str());

    float trainParam = 0;
    if (argc > 5) {
        trainParam = atof(argv[5]);
    }
    msg.str("");
    msg << "Setting RBM training parameter as: " << trainParam;
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

    int H = 500;
    if (argc > 8) {
        H = atoi(argv[8]);
    }
    msg.str("");
    msg << "Setting number of hidden neurons: " << H;
    printInfo(msg.str());

    int b_size = 5;
    if (argc > 9) {
        b_size = atoi(argv[9]);
    }
    msg.str("");
    msg << "Setting batch size: " << b_size;
    printInfo(msg.str());

    double l_rate = 0.01;
    if (argc > 10) {
        l_rate = atof(argv[10]);
    }
    msg.str("");
    msg << "Setting learning rate: " << l_rate;
    printInfo(msg.str());

    int f_nll = 10;
    if (argc > 11) {
        f_nll = atoi(argv[11]);
    }
    if (f_nll < 1) {
        printError("Cannot calculate more than one NLL per iteration");
        cerr << "Invalid f_nll value: " << f_nll << ". Aborting" << endl;
        exit(1);
    }
    msg.str("");
    msg << "Setting frequence of NLL calculation: " << f_nll;
    printInfo(msg.str());

    bool useLabels = false;
    if (argc > 12) {
        if (string(argv[12]) == "label") useLabels = true;
        if (atoi(argv[12]) == 1) useLabels = true;
    }
    if (useLabels) printInfo("RBM will be trained for classification!");


    // Data and RBM creation
    Data mnist("Datasets/bin_mnist-train.data", useLabels);
    mnist.joinLabels(useLabels);

    int X = mnist.get_sample_size();
    if (X == 0){
        printError("Could not find correct file, please check the file path");
        cerr << "No input file found" << endl;
        exit(1);
    }

    RBM model(X, H);
    model.setRandomSeed(seed);

    // Output files' base name
    stringstream filebase;
    filebase << "mnist_" << trainType;
    if (trainParam != 0) { filebase << "-" << trainParam; }
    filebase << "_H" << H << "_CD-" << k << "_lr" << l_rate << "_mBatch" << b_size << "_iter" << iter;
    if (useLabels) { filebase << "_withLabels" << seed; }
    if (seed != fileIDX) { filebase << "_seed" << seed; }
    filebase << "_run" << fileIDX;

    string rbm_fname, nll_fname, connect_fname;

    // Training
    bool doShuffle = true;  // Is there a reason I'd wish it not to be true?

    switch ( resolveOption(trainType) ) {
        case complete:
            printInfo("Training complete RBM");
            // model.connectivity(false);
            model.trainSetup(SampleType::CD, k, iter, b_size, l_rate, true, f_nll, doShuffle);
            model.fit(mnist);
            break;

        case sgd:
            if ( trainParam <= 0 ) {
                printError("Invalid training parameter");
                cerr << "Training parameter should be a number in (0,1], was given " << trainParam << endl;
                exit(1);
            }
            if ( trainParam > 1 ) {
                printWarning("Invalid training parameter, setting it to 1 instead");
                trainParam = 1;
            }
            msg.str("");
            msg << "Training connectivity with SGD (p=" << trainParam << ")";
            printInfo(msg.str());

            connect_fname = filePath + "connectivity_" + filebase.str() + ".csv";

            model.connectivity(true);
            model.trainSetup(SampleType::CD, k, iter, b_size, l_rate, true, f_nll, doShuffle);
            model.optSetup(Heuristic::SGD, connect_fname, trainParam);

            model.fit_connectivity(mnist);

            break;
        case conv:
            printInfo("Training RBM with convolutional connectivity");

            int side;

            side = int(sqrt(H));
            H = side*side;
            msg.str("");
            msg << "Since we use convolutional connectivity, changed number of hidden units to " << H;
            printWarning(msg.str());

            model.connectivity(true);
            model.setConnectivity(square_convolution( X, H ));
            cout << "Connectivity matrix:" << endl << model.getConnectivity() << endl;

            model.trainSetup(SampleType::CD, k, iter, b_size, l_rate, true, f_nll, doShuffle);
            model.fit(mnist);
            break;

        default:
            printError("Training type not recognized. Aborting operation.");
            cerr << "Training type '" << trainType << "' not implemented. Check what types are available" << endl;
            exit(1);
    }
    vector<double> h = model.getTrainingHistory();

    // Outputs
    rbm_fname = filePath + filebase.str() + ".rbm";
    nll_fname = filePath + "nll_" + filebase.str() + ".csv";

    model.save(rbm_fname);

    ofstream outdata;
    outdata.open(nll_fname);
    if( !outdata ) {
        cerr << "Error: file could not be opened" << endl;
        exit(1);
    }
    outdata << "# NLL through RBM training for MNIST data set with connectivity pattern of type " << trainType;
    if (trainParam != 0) { outdata << " (parameter " << trainParam << ")"; }
    outdata << endl;
    outdata << "# CD-" << k << ". Seed = " << seed << ", Batch size = " << b_size << " and learning rate of " << l_rate << endl;
    if (f_nll != 1) outdata << "# NLL calculated every " << f_nll << " iterations." << endl;

    outdata << ",NLL" << endl;
    for (int i=0; i<=(float(iter)/f_nll); i++) {
        outdata << i*f_nll << "," << h.at(i) << endl;
    }
    if ((iter % f_nll) != 0) outdata << iter-1 << "," << h.back() << endl;
    outdata.close();
}