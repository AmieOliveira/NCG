/*
 *  Script to train RBMs with MNIST dataset and save classification accuracy throughout
 *      training. One can choose between training a traditional RBM, to train connectivity
 *      using the SGD optimizer or to train RBM's with a fixed connectivity pattern from a
 *      set of available options. Does not calculate NLL.
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

enum TrainingTypes {
    complete,
    sgd,
    conv,
    neighLine,
    neighSpiral,
    random_connectivity,
    none,           // For inexistent types
};
TrainingTypes resolveOption(string opt) {
    if (opt == "complete") return complete;
    if (opt == "sgd") return sgd;
    if (opt == "convolution") return conv;
    if (opt == "neighborsLine") return neighLine;
    if (opt == "neighborsSpiral") return neighSpiral;
    if (opt == "random") return random_connectivity;
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

    int total_iter = 6000;
    if (argc > 7) {
        total_iter = atoi(argv[7]);
    }
    msg.str("");
    msg << "Setting number of iterations: " << total_iter;
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

    int f_acc = 10;
    if (argc > 11) {
        f_acc = atoi(argv[11]);
    }
    if (f_acc < 1) {
        printError("Cannot calculate more than one NLL per iteration");
        cerr << "Invalid f_acc value: " << f_acc << ". Aborting" << endl;
        exit(1);
    }
    msg.str("");
    msg << "Setting frequence of classification accuracy calculation: " << f_acc;
    printInfo(msg.str());


    // Data and RBM creation
    Data mnist("Datasets/bin_mnist-train.data", true);
    mnist.joinLabels(true);

    Data mnist_test("Datasets/bin_mnist-test.data", true);
    mnist_test.joinLabels(true);

    int X = mnist.get_sample_size();
    if (X == 0){
        printError("Could not find correct file, please check the file path");
        cerr << "No input file found" << endl;
        exit(1);
    }

    RBM model(X, H);
    model.setRandomSeed(seed);

    // Training variables
    bool doShuffle = true;  // Is there a reason I'd wish it not to be true?
    int nLabels = mnist.get_number_of_labels();
    int loops = ceil(total_iter/f_acc);
    double acc_train, acc_test;

    if ( int(loops * f_acc) != total_iter ) {
        total_iter = loops * f_acc;
        msg.str("");
        msg << "The given frequency of accuracy verification does not allow for exactly " << total_iter
            << " epochs of training. Will train for " << total_iter << " instead";
        printWarning(msg.str());
    }

    // Output files' base name
    stringstream filebase;
    filebase << "mnist_" << trainType;
    if (trainParam != 0) { filebase << "-" << trainParam; }
    filebase << "_H" << H << "_CD-" << k << "_lr" << l_rate << "_mBatch" << b_size << "_iter" << total_iter << "_withLabels";
    if (seed != fileIDX) { filebase << "_seed" << seed; }
    filebase << "_run" << fileIDX;

    string acc_fname = filePath + "acc_" + filebase.str() + ".csv";

    ofstream outdata;
    outdata.open(acc_fname);
    if( !outdata ) { cerr << "Error: file could not be opened" << endl; exit(1); }
    outdata << "# Classification accuracy through RBM training for MNIST data set with connectivity pattern of type " << trainType;
    if (trainParam != 0) { outdata << " (parameter " << trainParam << ")"; }
    outdata << endl;
    outdata << "# CD-" << k << ". Seed = " << seed << ", Batch size = " << b_size << " and learning rate of " << l_rate << endl;
    outdata << ",Train,Test" << endl;

    switch ( resolveOption(trainType) ) {
        case complete:
            printInfo("Training complete RBM");
            // model.connectivity(false);
            model.trainSetup(SampleType::CD, k, f_acc, b_size, l_rate, false, 0, doShuffle);

            acc_train = model.classificationStatistics(mnist, false);
            acc_test = model.classificationStatistics(mnist_test, false);

            cout << "Epoch 0:\tTrain Acc = " << acc_train
                 << " %\tTest Acc = " << acc_test << " %" << endl;
            outdata << 0 << "," << acc_train << "," << acc_test << endl;

            for (int l=1; l <= loops; l++) {
                model.fit(mnist);

                acc_train = model.classificationStatistics(mnist, false);
                acc_test = model.classificationStatistics(mnist_test, false);

                cout << "Epoch " << l * f_acc << ":\tTrain Acc = " << acc_train
                     << " %\tTest Acc = " << acc_test << " %" << endl;
                outdata << l * f_acc << "," << acc_train << "," << acc_test << endl;
            }
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

            model.connectivity(true);
            model.trainSetup(SampleType::CD, k, f_acc, b_size, l_rate, false, 0, doShuffle);
            model.optSetup(Heuristic::SGD, false, "", trainParam, nLabels);

            acc_train = model.classificationStatistics(mnist, false);
            acc_test = model.classificationStatistics(mnist_test, false);

            cout << "Epoch 0:\tTrain Acc = " << acc_train
                 << " %\tTest Acc = " << acc_test << " %" << endl;
            outdata << 0 << "," << acc_train << "," << acc_test << endl;

            for (int l=1; l <= loops; l++) {
                model.fit_connectivity(mnist);

                acc_train = model.classificationStatistics(mnist, false);
                acc_test = model.classificationStatistics(mnist_test, false);

                cout << "Epoch " << l * f_acc << ":\tTrain Acc = " << acc_train
                     << " %\tTest Acc = " << acc_test << " %" << endl;
                outdata << l * f_acc << "," << acc_train << "," << acc_test << endl;
            }
            break;

        case conv:
            printInfo("Training RBM with convolutional connectivity");

            int side, tmp;

            side = int(sqrt(H));
            tmp = H;
            H = side*side;
            if (H != tmp){
                msg.str("");
                msg << "Since we use convolutional connectivity, changed number of hidden units to " << H;
                printWarning(msg.str());
                model.setDimensions(X, H);
                model.setRandomSeed(seed);
            }

            model.connectivity(true);
            model.setConnectivity( square_convolution( X, H, nLabels ) );
            cout << "Connectivity matrix:" << endl << model.getConnectivity() << endl;

            model.trainSetup(SampleType::CD, k, f_acc, b_size, l_rate, false, 0, doShuffle);

            acc_train = model.classificationStatistics(mnist, false);
            acc_test = model.classificationStatistics(mnist_test, false);

            cout << "Epoch 0:\tTrain Acc = " << acc_train
                 << " %\tTest Acc = " << acc_test << " %" << endl;
            outdata << 0 << "," << acc_train << "," << acc_test << endl;

            for (int l=1; l <= loops; l++) {
                model.fit(mnist);

                acc_train = model.classificationStatistics(mnist, false);
                acc_test = model.classificationStatistics(mnist_test, false);

                cout << "Epoch " << l * f_acc << ":\tTrain Acc = " << acc_train
                     << " %\tTest Acc = " << acc_test << " %" << endl;
                outdata << l * f_acc << "," << acc_train << "," << acc_test << endl;
            }
            break;

        case random_connectivity: {
            if ( (trainParam <= 0) || (trainParam >= 1) ) {
                printError("Invalid training parameter");
                cerr << "Training parameter should be a number in (0,1), was given " << trainParam << endl;
                exit(1);
            }
            msg.str("");
            msg << "Training RBM with random connectivity (d=" << trainParam << ")";
            printInfo(msg.str());

            model.connectivity(true);
            model.setConnectivity( d_density_rdn( X, H, trainParam, seed+1, nLabels ) );
            cout << "Connectivity matrix:" << endl << model.getConnectivity() << endl;

            model.trainSetup(SampleType::CD, k, f_acc, b_size, l_rate, false, 0, doShuffle);

            acc_train = model.classificationStatistics(mnist, false);
            acc_test = model.classificationStatistics(mnist_test, false);

            cout << "Epoch 0:\tTrain Acc = " << acc_train
                 << " %\tTest Acc = " << acc_test << " %" << endl;
            outdata << 0 << "," << acc_train << "," << acc_test << endl;

            for (int l=1; l <= loops; l++) {
                model.fit(mnist);

                acc_train = model.classificationStatistics(mnist, false);
                acc_test = model.classificationStatistics(mnist_test, false);

                cout << "Epoch " << l * f_acc << ":\tTrain Acc = " << acc_train
                     << " %\tTest Acc = " << acc_test << " %" << endl;
                outdata << l * f_acc << "," << acc_train << "," << acc_test << endl;
            }

            break;
        }

        case neighLine: {
            printInfo("Training RBM with neighbors in line connectivity");

            if ( trainParam < 1 ) {
                printError("Invalid training parameter, should represent the number of neighbors per unit");
                cerr << "Tried to use " << trainParam << " neighbors. Should be a positive integer" << endl;
                exit(1);
            }
            int v = int(trainParam);
            if ( abs(v - trainParam) > 0 ) {
                printWarning("Training parameter has been rounded to an integer!");
            }

            model.connectivity(true);
            model.setConnectivity( v_neighbors_line_spread( X, H, trainParam, nLabels ) );
            // cout << "Connectivity matrix:" << endl << model.getConnectivity() << endl;

            model.trainSetup(SampleType::CD, k, f_acc, b_size, l_rate, false, 0, doShuffle);

            acc_train = model.classificationStatistics(mnist, false);
            acc_test = model.classificationStatistics(mnist_test, false);

            cout << "Epoch 0:\tTrain Acc = " << acc_train
                 << " %\tTest Acc = " << acc_test << " %" << endl;
            outdata << 0 << "," << acc_train << "," << acc_test << endl;


            for (int l=1; l <= loops; l++) {
                model.fit(mnist);

                acc_train = model.classificationStatistics(mnist, false);
                acc_test = model.classificationStatistics(mnist_test, false);

                cout << "Epoch " << l * f_acc << ":\tTrain Acc = " << acc_train
                     << " %\tTest Acc = " << acc_test << " %" << endl;
                outdata << l * f_acc << "," << acc_train << "," << acc_test << endl;
            }

            break;
        }

        default:
            printError("Training type not recognized. Aborting operation.");
            cerr << "Training type '" << trainType << "' not implemented. Check what types are available" << endl;
            exit(1);
    }

    // string rbm_fname = filePath + filebase.str() + ".rbm";
    // model.save(rbm_fname);
}