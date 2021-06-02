//
// Created by Amanda Oliveira on 04/05/21.
//      Para compilar: "g++ -I /usr/local/include/eigen3/ main.cpp RBM.cpp -o main.exe"
//

#include "RBM.h"

#include <fstream>
#include <stdlib.h>
//#include <chrono>

using namespace std;

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

/*
void testRandomGenerator(){
    typedef std::chrono::high_resolution_clock myclock;
    myclock::time_point beginning = myclock::now();

    unsigned seed1 = 2000051;
    std::mt19937 generator (seed1);
    std::cout << "Your seed produced: " << generator() << " and " << generator() << std::endl;

    myclock::duration d = myclock::now() - beginning;
    unsigned seed2 = d.count();
    generator.seed (seed2);
    std::cout << "A time seed produced: " << generator() << " and " << generator() << std::endl;

    cout << endl << "Considering the (0,1) distribution: " << endl;
    std::uniform_real_distribution<double> dis(0.0, 1.0);
    std::cout << "A time seed produced: " << dis(generator) << " and " << dis(generator) << std::endl;
    generator.seed (seed1);
    std::cout << "Your seed produced: " << dis(generator) << " - " << dis(generator) << " - "
              << dis(generator) << " - " << dis(generator) << " - " << dis(generator)
              << " - " << dis(generator) << " - " << dis(generator) << std::endl;
}
void testRandomGenOnRBM(){
    stringstream msg;

    if (argc < 2) {
        msg.str("You have entered too few arguments: only ");
        msg << argc << " found.\n\tFirst argument should be X and second H (redo if done wrong)";
        printError(msg.str());
        exit(1);
    }

    int X = atoi(argv[1]);
    int H = atoi(argv[2]);    // Numbers in ASCII characters begin at 48

    if (X == 0) {
        msg.str("Cannot have no visible units!\n\t\tExpecting a positive number, but received ");
        msg << argv[1];
        printError(msg.str());
        exit(1);
    }
    if (H == 0) {
        msg.str("Cannot have no hidden units!\n\t\tExpecting a positive number, but received ");
        msg << argv[2];
        printError(msg.str());
        exit(1);
    }

    msg.str("You have set the RBM to have ");
    msg << X << " visible units and " << H << " hidden ones.";
    printInfo(msg.str());

    RBM rbm(X, H);

    //double rnd = rbm.getRandomNumber(); // Should give error!

    rbm.setRandomSeed(X+H);
    cout << "Random numbers: " << rbm.getRandomNumber() << " - "
         << rbm.getRandomNumber() << " - " << rbm.getRandomNumber() << endl;
}
*/

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


int main(int argc, char **argv) {
    //testVariables(argc,argv);
    //testRandomGenerator();
    //testRandomGenOnRBM();
    //testSampling();
    //testRandomGenerator();
    //testaDataCreation(4,32);
    //checkNormalizationConstant();

    stringstream msg;

    unsigned seed = 18763258;
    if (argc >= 2) {
        seed = atoi(argv[1]);
        msg.str("");
        msg << "Setting seed as: " << seed;
        printInfo(msg.str());
    }

    int size = 4;
    if (argc >= 3) {
        size = atoi(argv[2]);
        msg.str("");
        msg << "Setting BAS size as: " << size;
        printInfo(msg.str());
    }

    int iter = 1000;
    if (argc >= 4) {
        iter = atoi(argv[3]);
        msg.str("");
        msg << "Setting number of iterations: " << iter;
        printInfo(msg.str());
    }

    Data bas(DataDistribution::BAS, size);

    for (int i = 0; i < bas.get_number_of_samples(); i++) {
        cout << "------------" << endl;
        RowVectorXd vec = bas.get_sample(i);
        for (int l = 0; l < size; ++l) {
            cout << vec.segment(l*size, size) << endl;
        }
    }
    cout << "------------" << endl;

    RBM model(bas.get_sample_size(), bas.get_sample_size());
    model.setRandomSeed(seed);
    model.trainSetup(SampleType::CD, 1, iter, 5, 0.1, true);
    model.fit(bas);

    vector<double> h = model.getTrainingHistory();

    ofstream outdata;
    outdata.open("nll_progress.csv"); // opens the file
    if( !outdata ) { // file couldn't be opened
        cerr << "Error: file could not be opened" << endl;
        exit(1);
    }

    outdata << "# NLL through RBM training for BAS" << size << endl;
    outdata << "NLL" << endl;
    for (auto i: h)
        outdata << i << endl;
    outdata.close();

    return 0;
}