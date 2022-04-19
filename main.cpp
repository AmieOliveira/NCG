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

void testRBMprediction() {
    Data mnist("Datasets/bin_mnist-test.data", true);  // Data mnist("Datasets/bin_mnist-train.data", true);
    cout << "Using " << mnist.get_number_of_samples() << " training samples." << endl;
    mnist.joinLabels(false);

    RBM model;
    model.load("Training Outputs/Teste MNIST/mnist_complete_H500_CD-1_lr0.01_mBatch50_iter20_withLabels_run4.rbm");
    model.setRandomSeed(76244);

    VectorXd pred;
    int predLabel;
    int totSamples = 20;
    double accuracy = 0;
    int init = 0; // 8734;

    for (int idx = init; idx < init + totSamples; idx++) {
        cout << "Will predict " << idx+1 << "th sample. Label = " << mnist.get_label(idx) << endl;
        cout << "\tShould get vector: " << mnist.get_label_vector(idx).transpose() << endl;
        pred = model.complete_pattern( mnist.get_sample(idx), 4 );
        cout << "\tPredicted vector:  ";
        for (int i = mnist.get_sample_size(); i < mnist.get_sample_size() + mnist.get_number_of_labels(); i++)
            cout << " " << pred(i);
        cout << endl;

        predLabel = model.predict( mnist.get_sample(idx), mnist.get_number_of_labels() );
        cout << "Predicted label: " << predLabel << endl;
        cout << endl;

        if ( predLabel == mnist.get_label(idx) ) accuracy++;
    }

    cout << endl << "Total accuracy: " << 100 * double(accuracy/totSamples) << "%" << endl;


    model.classificationStatistics(mnist, true);
}


int main(int argc, char **argv) {
    // RBM model(784,15);
    //
    // // model.load("Training Outputs/Teste MNIST/bas4_CD-1_lr0.01_mBatch5_iter2500_seed0.rbm");
    // // model.load("Training Outputs/Teste MNIST/mnist_complete_H16_CD-1_lr0.01_mBatch50_iter1000_run0.rbm");
    // model.load("Training Outputs/Teste MNIST/mnist_complete_H16_CD-1_lr0.01_mBatch50_iter1000_run0-LAND.rbm");
    //
    // // MatrixXd w = MatrixXd::Constant(15,784,1);
    // // w(0,1) = -1; w(1,0) = -1; w(1,2) = 2;
    // // w(3,7) = .3; w(0,9) = -1; w(2,8) = -.5; w(4,3) = .7;
    // // model.setWeights(w);
    // //
    // // VectorXd b(15);
    // // b(0) = 1; b(1) = 2; b(2) = 1.23; b(3) = -0.1; b(4) = -1;
    // // model.setHiddenBiases(b);
    // //
    // // VectorXd d(784);
    // // d(0) = 1; d(1) = 2; d(2) = 3; d(3) = 2; d(4) = 1; d(5) = -1; d(6) = -2; d(7) = .7; d(8) = 1; d(9) = -.2;
    // // model.setVisibleBiases(d);
    //
    // // model.printVariables();
    //
    // MatrixXd datMat = MatrixXd::Constant(784, 1, 1);
    // Data dat(datMat);
    //
    // double nll; // = model.negativeLogLikelihood(dat, None);
    // nll = model.negativeLogLikelihood(dat, None_H);
    //
    // // model.save("test.rbm");

    int size = 4;
    int X = size*size, H = 10;

    RBM model(X,H);
    model.connectivity(true);
    model.setConnectivity( d_density_rdn(H, X, 0.25, 1374) );
    model.printVariables();

    return 0;
}