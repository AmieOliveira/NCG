#include "Data.h"
#include "RBM.h"
#include <stdlib.h>
#include <fstream>
#include <cmath>
#include <limits>

using namespace std;

int main(int argc, char **argv) {
    Data mnistTrain("Datasets/bin_mnist-train.data", true);
    cout << "Using " << mnistTrain.get_number_of_samples() << " training samples." << endl;

    Data mnistTest("Datasets/bin_mnist-test.data", true);
    cout << "Using " << mnistTest.get_number_of_samples() << " test samples." << endl;

    RBM model;

    string rbmType = "complete";
    if (argc > 1) {
        rbmType = argv[1];
    }

    int H = 500;
    if (argc > 2) {
        H = atoi(argv[2]);
    }

    int k = 1;
    if (argc > 3) {
        k = atoi(argv[3]);
    }

    int iter = 100;
    if (argc > 4) {
        iter = atoi(argv[4]);
    }

    double lr = 0.1;
    if (argc > 5) {
        lr = atof(argv[5]);
    }

    int bSize = 50;
    if (argc > 6) {
        bSize = atoi(argv[6]);
    }

    int repeat = 5;
    if (argc > 7) {
        repeat = atoi(argv[7]);
    }

    vector<double> trainResults;
    vector<double> testResults;

    for (int i=0; i<repeat; i++) {
        stringstream fname;
        fname << "Training Outputs/MNIST Classificacao/mnist_" << rbmType << "_H" << H << "_CD-" << k << "_lr" << lr
              << "_mBatch" << bSize << "_iter" << iter << "_withLabels_run" << i << ".rbm";
        try {
            model.load(fname.str());
        } catch (runtime_error) {
            continue;
        }

        cout << "FILE: '" << fname.str() << "'" << endl;
        cout << "TRAIN SET" << endl;
        trainResults.push_back( model.classificationStatistics(mnistTrain) );
        cout << endl << "TEST SET" << endl;
        testResults.push_back( model.classificationStatistics(mnistTest) );
        cout << endl << endl;
    }

    double sumT = 0;

    cout << "Train accuracies: \t";
    for (auto acc: trainResults) {
        sumT += acc;
        cout << acc << "\t";
    }
    cout << endl << "Mean train accuracy: " << sumT/repeat << endl;

    sumT = 0;
    cout << "Test accuracies: \t";
    for (auto acc: testResults) {
        sumT += acc;
        cout << acc << "\t";
    }
    cout << endl << "Mean test accuracy: " << sumT/repeat << endl;

    // TODO: Do I want to calculate standard deviation?

    return 0;
}