/*
 *  Script to train RBMs with miscelaneous datasets and save classification accuracy throughout
 *      training. One can choose between training a traditional RBM, to train connectivity
 *      using the SGD optimizer or to train RBM's with a fixed connectivity pattern from a
 *      set of available options. Does not calculate NLL.
 *  Script tested with mushrooms dataset.
 *
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

// TODO: Save connectivity / hidden neurons activation
//   I could save it like I am saving the accuracy, just using
//   the helper functions 'printConnectivity_linear' and
//   'printHiddenActivation'. I'd just need to make them public

enum TrainingTypes {
    complete,
    sgd,
    conv,
    neighLine,
    neighSpiral,
    random_connectivity,
    ncgh,
    none,           // For inexistent types
};
TrainingTypes resolveOption(string opt) {
    if (opt == "complete") return complete;
    if (opt == "sgd") return sgd;
    if (opt == "convolution") return conv;
    if (opt == "neighborsLine") return neighLine;
    if (opt == "neighborsSpiral") return neighSpiral;
    if (opt == "random") return random_connectivity;
    if (opt == "ncgh") return ncgh;
    return none;
}


int main(int argc, char **argv) {
    stringstream msg;

    int fileIDX;
    string dname;
    string dataset;

    if (argc < 4) {
        msg << "Error! Three arguments required: file Idx; dataset name; and dataset file."
            << " Arguments must be given in this order!";
        printError(msg.str());
        throw runtime_error(msg.str());
    } else {
        fileIDX = atoi(argv[1]);
        msg << "File " << fileIDX;
        printInfo(msg.str());

        dname = argv[2];
        msg.str("");
        msg << "Dataset name: " << dname;
        printInfo(msg.str());

        dataset = argv[3];
        msg.str("");
        msg << "Dataset: " << dataset;
        printInfo(msg.str());
    }

    bool hasTestSet = false;
    string dataset_test;
    if (argc > 4) {
        dataset_test = argv[4];
        msg.str("");
        if ( dataset_test != "0" ) {
            hasTestSet = true;
            msg << "Dataset (test split): " << dataset_test;
            printInfo(msg.str());
        } else {
            msg << "No test set";
            printWarning(msg.str());
        }
    }

    string filePath = "./";
    if (argc > 5) {
        filePath = argv[5];
        if ( !(filePath[filePath.length() - 1] == '/') ) filePath = filePath + "/";
    }
    msg.str("");
    msg << "File directory: " << filePath;
    printInfo(msg.str());

    unsigned seed = 18763258;
    if (argc > 6) {
        seed = atoi(argv[6]);
    }
    msg.str("");
    msg << "Setting seed as: " << seed;
    printInfo(msg.str());

    string trainType = "complete";
    if (argc > 7) {
        trainType = argv[7];
    }
    msg.str("");
    msg << "Setting RBM type as: " << trainType;
    printInfo(msg.str());

    float trainParam = 0;
    if (argc > 8) {
        trainParam = atof(argv[8]);
    }
    msg.str("");
    msg << "Setting RBM training parameter as: " << trainParam;
    printInfo(msg.str());

    int k = 10;
    if (argc > 9) {
        k = atoi(argv[9]);
    }
    msg.str("");
    msg << "Setting number of sample steps: " << k;
    printInfo(msg.str());

    int total_iter = 6000;
    if (argc > 10) {
        total_iter = atoi(argv[10]);
    }
    msg.str("");
    msg << "Setting number of iterations: " << total_iter;
    printInfo(msg.str());

    int H = 500;
    if (argc > 11) {
        H = atoi(argv[11]);
    }
    msg.str("");
    msg << "Setting number of hidden neurons: " << H;
    printInfo(msg.str());

    int b_size = 5;
    if (argc > 12) {
        b_size = atoi(argv[12]);
    }
    msg.str("");
    msg << "Setting batch size: " << b_size;
    printInfo(msg.str());

    double l_rate = 0.01;
    if (argc > 13) {
        l_rate = atof(argv[13]);
    }
    msg.str("");
    msg << "Setting learning rate: " << l_rate;
    printInfo(msg.str());

    int f_acc = 10;
    if (argc > 14) {
        f_acc = atoi(argv[14]);
    }
    if (f_acc < 1) {
        printError("Cannot calculate more than one accuracy value per iteration");
        cerr << "Invalid f_acc value: " << f_acc << ". Aborting" << endl;
        exit(1);
    }
    msg.str("");
    msg << "Setting frequence of classification accuracy calculation: " << f_acc;
    printInfo(msg.str());


    // Data and RBM creation
    Data data(dataset, true);
    data.joinLabels(true);

    Data* data_test;
    if (hasTestSet) {
        data_test = new Data(dataset_test, true);
        data_test->joinLabels(true);
    }

    int X = data.get_sample_size();
    if (X == 0){
        printError("Could not find correct file, please check the file path");
        cerr << "No input file found" << endl;
        exit(1);
    }

    RBM model(X, H);
    model.setRandomSeed(seed);

    // Training variables
    bool doShuffle = true;  // Is there a reason I'd wish it not to be true?
    int nLabels = data.get_number_of_labels();
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
    filebase << dname << "_" << trainType;
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

    string connect_fname;
    ofstream conFile;

    switch ( resolveOption(trainType) ) {
        case complete:
            printInfo("Training complete RBM");
            // model.connectivity(false);
            model.trainSetup(SampleType::CD, k, f_acc, b_size, l_rate, false, 0, doShuffle);

            acc_train = model.classificationStatistics(data, false);
            if (hasTestSet)
                acc_test = model.classificationStatistics(*data_test, false);

            cout << "Epoch 0:\tTrain Acc = " << acc_train
                 << " %\tTest Acc = " << acc_test << " %" << endl;
            outdata << 0 << "," << acc_train << "," << acc_test << endl;

            for (int l=1; l <= loops; l++) {
                model.fit(data);

                acc_train = model.classificationStatistics(data, false);
                if (hasTestSet)
                    acc_test = model.classificationStatistics(*data_test, false);

                cout << "Epoch " << l * f_acc << ":\tTrain Acc = " << acc_train;
                if (hasTestSet)
                     cout << " %\tTest Acc = " << acc_test << " %";
                cout << endl;
                outdata << l * f_acc << "," << acc_train;
                if (hasTestSet)
                    outdata << "," << acc_test;
                outdata << endl;
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

            acc_train = model.classificationStatistics(data, false);
            if (hasTestSet)
                acc_test = model.classificationStatistics(*data_test, false);

            cout << "Epoch 0:\tTrain Acc = " << acc_train;
            if (hasTestSet)
                cout << " %\tTest Acc = " << acc_test << " %";
            cout << endl;
            outdata << 0 << "," << acc_train;
            if (hasTestSet)
                outdata << "," << acc_test;
            outdata << endl;

            connect_fname = filePath + "connectivity_" + filebase.str() + ".csv";
            conFile.open(connect_fname);
            if( !conFile ) { cerr << "Error: file could not be opened" << endl; exit(1); }
            conFile << "# Connectivity patterns through RBM training for MNIST data set (";
            conFile << trainType << " with p = " << trainParam << ")" << endl;
            conFile << "# CD-" << k << ". Seed = " << seed << ", Batch size = " << b_size << " and learning rate of " << l_rate << endl;
            conFile << "0," << model.printConnectivity_linear() << endl;

            for (int l=1; l <= loops; l++) {
                model.fit_connectivity(data);

                acc_train = model.classificationStatistics(data, false);
                if (hasTestSet) acc_test = model.classificationStatistics(*data_test, false);

                cout << "Epoch " << l * f_acc << ":\tTrain Acc = " << acc_train;
                if (hasTestSet) cout << " %\tTest Acc = " << acc_test << " %";
                cout << endl;
                outdata << l * f_acc << "," << acc_train;
                if (hasTestSet) outdata << "," << acc_test;
                outdata << endl;

                conFile << l * f_acc << "," << model.printConnectivity_linear() << endl;
            }
            break;

        case ncgh:
            if ( trainParam < 1 ) {
                printError("Invalid training parameter");
                cerr << "Training parameter should be an integer greater than 0, was given " << trainParam << endl;
                exit(1);
            }
            msg.str("");
            msg << "Training hidden neurons activation (initializing with " << trainParam << " units)";
            printInfo(msg.str());

            model.hidden_activation(true);
            model.trainSetup(SampleType::CD, k, f_acc, b_size, l_rate, false, 0, doShuffle);
            model.optSetup(Heuristic::SGD, false, "", trainParam, nLabels);

            acc_train = model.classificationStatistics(data, false);
            if (hasTestSet)
                acc_test = model.classificationStatistics(*data_test, false);

            cout << "Epoch 0:\tTrain Acc = " << acc_train;
            if (hasTestSet)
                cout << " %\tTest Acc = " << acc_test << " %";
            cout << endl;
            outdata << 0 << "," << acc_train;
            if (hasTestSet)
                outdata << "," << acc_test;
            outdata << endl;

            connect_fname = filePath + "connectivity_" + filebase.str() + ".csv";
            conFile.open(connect_fname);
            if( !conFile ) { cerr << "Error: file could not be opened" << endl; exit(1); }
            conFile << "# Hidden units activation through RBM training for MNIST data set (";
            conFile << trainType << " with p = " << trainParam << ")" << endl;
            conFile << "# CD-" << k << ". Seed = " << seed << ", Batch size = " << b_size << " and learning rate of " << l_rate << endl;
            conFile << "0," << model.printHiddenActivation() << endl;

            for (int l=1; l <= loops; l++) {
                model.fit_H(data);

                acc_train = model.classificationStatistics(data, false);
                if (hasTestSet) acc_test = model.classificationStatistics(*data_test, false);

                cout << "Epoch " << l * f_acc << ":\tTrain Acc = " << acc_train;
                if (hasTestSet) cout << " %\tTest Acc = " << acc_test << " %";
                cout << endl;
                outdata << l * f_acc << "," << acc_train;
                if (hasTestSet) outdata << "," << acc_test;
                outdata << endl;

                conFile << l * f_acc << "," << model.printHiddenActivation() << endl;
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

            acc_train = model.classificationStatistics(data, false);
            if (hasTestSet) acc_test = model.classificationStatistics(*data_test, false);

            cout << "Epoch 0:\tTrain Acc = " << acc_train;
            if (hasTestSet) cout << " %\tTest Acc = " << acc_test << " %";
            cout << endl;
            outdata << 0 << "," << acc_train;
            if (hasTestSet) outdata << "," << acc_test;
            outdata << endl;

            for (int l=1; l <= loops; l++) {
                model.fit(data);

                acc_train = model.classificationStatistics(data, false);
                if (hasTestSet) acc_test = model.classificationStatistics(*data_test, false);

                cout << "Epoch " << l * f_acc << ":\tTrain Acc = " << acc_train;
                if (hasTestSet) cout << " %\tTest Acc = " << acc_test << " %";
                cout << endl;
                outdata << l * f_acc << "," << acc_train;
                if (hasTestSet) outdata << "," << acc_test;
                outdata << endl;
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

            acc_train = model.classificationStatistics(data, false);
            if (hasTestSet) acc_test = model.classificationStatistics(*data_test, false);

            cout << "Epoch 0:\tTrain Acc = " << acc_train;
            if (hasTestSet) cout << " %\tTest Acc = " << acc_test << " %";
            cout << endl;
            outdata << 0 << "," << acc_train;
            if (hasTestSet) outdata << "," << acc_test;
            outdata << endl;

            for (int l=1; l <= loops; l++) {
                model.fit(data);

                acc_train = model.classificationStatistics(data, false);
                if (hasTestSet) acc_test = model.classificationStatistics(*data_test, false);

                cout << "Epoch " << l * f_acc << ":\tTrain Acc = " << acc_train;
                if (hasTestSet) cout << " %\tTest Acc = " << acc_test << " %";
                cout << endl;
                outdata << l * f_acc << "," << acc_train;
                if (hasTestSet) outdata << "," << acc_test;
                outdata << endl;
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

            acc_train = model.classificationStatistics(data, false);
            if (hasTestSet) acc_test = model.classificationStatistics(*data_test, false);

            cout << "Epoch 0:\tTrain Acc = " << acc_train;
            if (hasTestSet) cout << " %\tTest Acc = " << acc_test << " %";
            cout << endl;
            outdata << 0 << "," << acc_train;
            if (hasTestSet) outdata << "," << acc_test;
            outdata << endl;


            for (int l=1; l <= loops; l++) {
                model.fit(data);

                acc_train = model.classificationStatistics(data, false);
                if (hasTestSet) acc_test = model.classificationStatistics(*data_test, false);

                cout << "Epoch " << l * f_acc << ":\tTrain Acc = " << acc_train;
                if (hasTestSet) cout << " %\tTest Acc = " << acc_test << " %";
                cout << endl;
                outdata << l * f_acc << "," << acc_train;
                if (hasTestSet) outdata << "," << acc_test;
                outdata << endl;
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