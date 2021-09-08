//
// Created by Amanda Oliveira on 04/05/21.
//

#include "RBM.h"

#include <stdlib.h>
//#include <chrono>

using namespace std;

// FIXME: Change the name of this file to "tester"?
//        Posso ir fazendo arquivos como o completeGraph para de fato serem usados...

// TODO: Add parser and logger

// Funções de teste (auxiliares)
void testVariables(int argc, char **argv){
    if (argc < 2) {
        cout << "ERROR:\tYou have entered too few arguments: only " << argc << " found.\n"
                                                                               "      \tFirst argument should be X and second H (redo if done wrong)"
             << endl;
        exit(1);
    }

    int X = atoi(argv[1]);
    int H = atoi(argv[2]);    // Numbers in ASCII characters begin at 48

    if (X == 0) {
        cout << "ERROR:\tCannot have no visible units!\n"
                "      \tExpecting a positive number, but received " << argv[1] << endl;
        exit(1);
    }
    if (H == 0) {
        cout << "ERROR:\tCannot have no hidden units!\n"
                "      \tExpecting a positive number, but received " << argv[2] << endl;
        exit(1);
    }

    cout << "INFO:\tYou have set the RBM to have " << X <<
         " visible units and " << H << " hidden ones." << endl;


    RBM rbm(X, H);
    //rbm.setDimensions(X,H);
    rbm.printVariables();

    VectorXd vec(X); //rbm.getVisibleUnits();
    vec = VectorXd::Constant(X, 1);

    int ok_flag = rbm.setVisibleUnits(vec);
    if (ok_flag != 0) {
        cout << "Problem occured when trying to set RBM visible "
                "units (" << ok_flag << "). Abborting." << endl;
        exit(1);
    }

    MatrixXd mat = MatrixXd::Random(H,X);
    ok_flag = rbm.setWeights(mat);
    if (ok_flag != 0) {
        cout << "Problem occured when trying to set RBM weight "
                "matrix (" << ok_flag << "). Abborting." << endl;
        exit(1);
    }
    rbm.printVariables();

    mat = MatrixXd::Constant(H,X, 1);
    mat(0,0) = 0;
    mat(int(H/2),X-1) = 0;
    rbm.connectivity(false);
    rbm.connectivity(true);
    ok_flag = rbm.setConnectivity(mat);
    if (ok_flag != 0) {
        cout << "Problem occured when trying to set RBM connectivity "
                "pattern (" << ok_flag << "). Abborting." << endl;
        exit(1);
    }
    rbm.printVariables();

    VectorXd vec2(2, 1);
    ok_flag = rbm.setVisibleUnits(vec2);
    if (ok_flag != 0) {
        cout << "Problem occured when trying to set RBM visible "
                "units (" << ok_flag << "). Abborting." << endl;
        exit(1); // Should raise this error and quit without executing further things
    }
    rbm.printVariables();
}

void testSampling(int argc, char **argv){
    stringstream msg;

    if (argc < 3) {
        msg.str("");
        msg << "You have entered too few arguments: only " << argc-1
            << " found.\n\tShould give X and H values as argument "
               "(redo if done wrong)";
        printError(msg.str());
        exit(1);
    }

    int X = atoi(argv[1]);
    int H = atoi(argv[2]);    // Numbers in ASCII characters begin at 48

    if (X == 0) {
        msg.str("");
        msg << "Cannot have no visible units!\n\t\tExpecting a positive "
               "number, but received " << argv[1];
        printError(msg.str());
        exit(1);
    }
    if (H == 0) {
        msg.str("");
        msg << "Cannot have no hidden units!\n\t\tExpecting a positive "
               "number, but received " << argv[2];
        printError(msg.str());
        exit(1);
    }

    msg.str("");
    msg << "You have set the RBM to have " << X << " visible units and "
        << H << " hidden ones.";
    printInfo(msg.str());

    RBM rbm(X, H);

    int check;
    //check = rbm.setWeights(MatrixXd::Random(H,X));
    //if (check != 0){exit(1);}
    VectorXd vec = VectorXd::Constant(H,1);
    check = rbm.setHiddenUnits(vec);
    if (check != 0){exit(1);}

    //MatrixXd mat = MatrixXd::Constant(H,X,1);
    //mat(0,0) = 0; mat(1,X-1) = 0; mat(H-1,X-2) = 0;
    //rbm.connectivity(true);
    //rbm.setConnectivity(mat);

    unsigned seed = 98;
    if (argc >= 4) {
        seed = atoi(argv[3]);
        msg.str("");
        msg << "Setting seed as: " << seed;
        printInfo(msg.str());
    }
    rbm.setRandomSeed(seed);
    rbm.startWeights();
    rbm.sampleXH();

    rbm.printVariables();

    rbm.printVariables();
}

void testRandomGenerator(){
    RBM rbm1(2, 2);

    unsigned seed = 61309258;

    cout << "Testing Mersenne Twister" << endl;
    rbm1.setRandomSeed(seed);
    rbm1.generatorTest();

    cout << "Testing generator sampling in 2x2 RBM: " << endl;
    rbm1.validateSample(seed, 1000);

    cout << "Testing generator sampling in 2x3 RBM: " << endl;
    RBM rbm2(2, 3);
    rbm2.validateSample(seed, 1000);

    cout << "Testing generator sampling in 4x3 RBM: " << endl;
    RBM rbm3(4, 3);
    rbm3.validateSample(seed, 1000);
}

void testaDataCreation(int size, int n){
    Data bas(DataDistribution::BAS, size, n);

    for (int i = 0; i < n; i++) {
        cout << "------------" << endl;
        RowVectorXd vec = bas.get_sample(i);
        for (int l = 0; l < size; ++l) {
            cout << vec.segment(l*size, size) << endl;
        }
    }
    cout << "------------" << endl;

    cout << "Relative frequency of first variable: "
         << bas.marginal_relativeFrequence(0) << endl;

    vector<Data> sets = bas.separateTrainTestSets(0.6);
    cout << "Train set with " << sets[0].get_number_of_samples()
         << " samples and test set with " << sets[1].get_number_of_samples()
         << endl;

    RBM model(bas.get_sample_size(), bas.get_sample_size());
    model.setRandomSeed(18763258);

    double nll = model.negativeLogLikelihood(bas);
    cout << "NLL before training: " << nll << endl;

    model.trainSetup();
    model.fit(bas);

    model.printVariables();

    nll = model.negativeLogLikelihood(bas);
    cout << "NLL after training: " << nll << endl;
}

void checkNormalizationConstant(){
    MatrixXd data = MatrixXd::Zero(2, 1); data(1,0) = 1;
    Data bas(data);
    RBM model(2, 2);

    VectorXd b(2); b << 1, 1; model.setHiddenBiases(b);
    VectorXd d(2); d << 1, 1; model.setVisibleBiases(d);
    MatrixXd W = MatrixXd::Identity(2, 2); model.setWeights(W);

    double nll = model.negativeLogLikelihood(bas);
    cout << "NLL: " << nll << endl;
    model.printVariables();
}

void testSaveLoad() {
    int basSize = 2;
    int size = basSize*basSize;

    Data bas(DataDistribution::BAS, basSize);

    RBM model(size, size+1);
    model.setRandomSeed(0);
    model.trainSetup(SampleType::CD, 1, 100, 5, 0.01, false);
    model.fit(bas);
    model.printVariables();
    model.save("test.rbm");

    RBM m2;
    m2.load("test.rbm");
    m2.printVariables();
}

void checkNormConstantEstimation(){
    // Data bas(DataDistribution::BAS, 2);
    // RBM model(4, 4);
    // model.setRandomSeed(0);
    // model.trainSetup(SampleType::CD, 1, 1000, 5, 0.01, false);
    // model.fit(bas);
    // model.printVariables();

    RBM model;
    model.load("Training Outputs/Teste MNIST/bas4_CD-1_lr0.01_mBatch5_iter2500_seed0.rbm");

    model.setRandomSeed(6324);
    model.sampleXH();   // Just to mix a bit the x and h units

    double exact = log( model.normalizationConstant_effX() );
    cout << "Exact value: " << exact << endl;
    cout << endl;

    for (int i=0; i<10; i++) {
        model.setRandomSeed(2000 + 3*i);
        double estimated = log( model.normalizationConstant_MCestimation(10000) );
        cout << "Estimated value: " << estimated << endl;
    }
}

void testDataShuffle() {
    Data bas(DataDistribution::BAS, 2);
    int size = bas.get_number_of_samples();

    cout << "Original samples:" << endl;
    for (int i; i < size; i++) {
        cout << bas.get_sample(i).transpose() << endl;
    }

    bas.shuffle(7335);

    cout << "Shuffled samples:" << endl;
    for (int i; i < size; i++) {
        cout << bas.get_sample(i).transpose() << endl;
    }
}


int main(int argc, char **argv) {

    Data mnistTrain("Datasets/bin_mnist-train.data", false);
    cout << "Using " << mnistTrain.get_number_of_samples() << " training samples." << endl;

    Data mnistTest("Datasets/bin_mnist-test.data", false);
    cout << "Using " << mnistTest.get_number_of_samples() << " test samples." << endl;

    int size = mnistTrain.get_sample_size();

    int k = 10;
    int iter = 2;
    int b_size = 10;
    double l_rate = 0.01;
    double p = 1;
    unsigned seed = 1382;  // 8924
    bool shuffleData = true;

    // Traditional RBM
    RBM model(size, 500, false);
    model.setRandomSeed(seed);
    // model.trainSetup(SampleType::CD, k, iter, b_size, l_rate, false, 0, shuffleData);
    // model.fit(mnistTrain);
    // model.save("mnist-partial_H500_CD-10_lr0.01_mBatch10_iter2_shuffle_seed0.rbm");


    int repeat = 5;

    // Auxiliar variables
    VectorXd nll(repeat);
    VectorXd nllTest(repeat);
    double meanTrain, meanTest, sumTrain, sumTest;

    // // RBM model complete
    // model.load("Training Outputs/Teste MNIST/mnist_complete_H500_CD-1_lr0.01_mBatch50_iter20_run0.rbm");
    // model.setRandomSeed(seed);
    //
    // for (int r=0; r<repeat; r++) {
    //     nll(r) = model.negativeLogLikelihood(mnist);
    //     cout << "Estimated value: " << nll(r) << endl;
    // }
    // cout << "\t Complete: Mean of " << nll.mean() << endl;

    // RBM model Complete
    model.load("Training Outputs/Teste MNIST/mnist_complete_H500_CD-1_lr0.01_mBatch50_iter20_run4.rbm");
    model.setRandomSeed(seed);

    for (int r=0; r<repeat; r++) {
        nll(r) = model.negativeLogLikelihood(mnistTrain);
        nllTest(r) = model.negativeLogLikelihood(mnistTest);
        cout << "Estimated value: " << nll(r) << " for train set and " << nllTest(r) << " for test set" << endl;
    }

    meanTrain = nll.mean();
    meanTest = nllTest.mean();
    cout << "\t Complete (run 4): Train mean of " << meanTrain << " and test mean of " << meanTest << endl;

    sumTrain = 0;
    sumTest = 0;
    for (int r=0; r<repeat; r++) {
        sumTrain += (nll(r) - meanTrain)*(nll(r) - meanTrain);
        sumTest += (nllTest(r) - meanTest)*(nllTest(r) - meanTest);
    }
    sumTrain = sqrt(sumTrain/repeat);
    sumTest = sqrt(sumTest/repeat);
    cout << "\t Complete (run 4): Train standard deviation of " << sumTrain << " and test one of " << sumTest << endl;


    // RBM model Convolutional
    model.load("Training Outputs/Teste MNIST/mnist_convolution_H500_CD-1_lr0.01_mBatch50_iter20_run4.rbm");
    model.setRandomSeed(seed);

    for (int r=0; r<repeat; r++) {
        nll(r) = model.negativeLogLikelihood(mnistTrain);
        nllTest(r) = model.negativeLogLikelihood(mnistTest);
        cout << "Estimated value: " << nll(r) << " for train set and " << nllTest(r) << " for test set" << endl;
    }

    meanTrain = nll.mean();
    meanTest = nllTest.mean();
    cout << "\t Convolution (run 4): Train mean of " << meanTrain << " and test mean of " << meanTest << endl;

    sumTrain = 0;
    sumTest = 0;
    for (int r=0; r<repeat; r++) {
        sumTrain += (nll(r) - meanTrain)*(nll(r) - meanTrain);
        sumTest += (nllTest(r) - meanTest)*(nllTest(r) - meanTest);
    }
    sumTrain = sqrt(sumTrain/repeat);
    sumTest = sqrt(sumTest/repeat);
    cout << "\t Convolution (run 4): Train standard deviation of " << sumTrain << " and test one of " << sumTest << endl;


    // RBM model SGD
    model.load("Training Outputs/Teste MNIST/mnist_sgd-1_H500_CD-1_lr0.01_mBatch50_iter20_run4.rbm");
    model.setRandomSeed(seed);

    for (int r=0; r<repeat; r++) {
        nll(r) = model.negativeLogLikelihood(mnistTrain);
        nllTest(r) = model.negativeLogLikelihood(mnistTest);
        cout << "Estimated value: " << nll(r) << " for train set and " << nllTest(r) << " for test set" << endl;
    }

    meanTrain = nll.mean();
    meanTest = nllTest.mean();
    cout << "\t SGD (run 4): Train mean of " << meanTrain << " and test mean of " << meanTest << endl;

    sumTrain = 0;
    sumTest = 0;
    for (int r=0; r<repeat; r++) {
        sumTrain += (nll(r) - meanTrain)*(nll(r) - meanTrain);
        sumTest += (nllTest(r) - meanTest)*(nllTest(r) - meanTest);
    }
    sumTrain = sqrt(sumTrain/repeat);
    sumTest = sqrt(sumTest/repeat);
    cout << "\t SGD (run 4): Train standard deviation of " << sumTrain << " and test one of " << sumTest << endl;

    return 0;
}