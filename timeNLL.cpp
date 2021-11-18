//
//  Script to compare output and execution time of different NLL estimators
//

#include "RBM.h"

#include <stdlib.h>
#include <chrono>

using namespace std;



int main(int argc, char **argv) {
    // TODO: Add arguments so that I don't have to have all specifications hardcoded


    int k = 1;
    int iter = 200;
    int b_size = 50;  // 5
    double l_rate = 0.01;
    int H = 16;
    double p = 1;
    unsigned seed = 1;  // 232140824 9217 71263 92174
    bool shuffleData = true;

    int repeat = 1;
    int n_seeds = 1;

    //// BAS Dataset
    //Data bas(BASnoRep, 4);
    //Data bas(seed, BASnoRep, 28, 1000);
    Data bas("Datasets/bin_mnist-test.data", false); // bas = bas.separateTrainTestSets(0.1).at(0);
    // Data bas(BAS, 4);

    // Traditional RBM
    RBM model;

    model.load("Training Outputs/Redes Treinadas/H500_nets/mnist_sgd-1_H500_CD-1_lr0.01_mBatch50_iter200_run0.rbm");

    cout << "Training model for " << iter << " epochs" << endl;

    for (int s = 0; s < n_seeds; s++) {
        cout << "Experiment " << s+1 << " of " << n_seeds << endl;

        // model.setDimensions(bas.get_sample_size(), H, false);
        // model.setRandomSeed(seed + s);
        // model.trainSetup(SampleType::CD, k, iter, b_size, l_rate, false, 1, shuffleData);
        // // cout << "Before training: " << model.negativeLogLikelihood(bas) << endl;
        //
        // model.fit(bas);
        // // model.connectivity(true); model.optSetup(Heuristic::SGD, p); model.fit_connectivity(bas);

        auto t1 = chrono::steady_clock::now();
        // cout << "Exact value (through H): " << model.negativeLogLikelihood(bas, ZEstimation::None_H) << endl;
        auto t2 = chrono::steady_clock::now();
        // cout << "Truncation estimation (no verification): " << model.negativeLogLikelihood(bas, ZEstimation::Trunc) << endl;
        // auto t3 = chrono::steady_clock::now();

        // cout << "Time elapsed for total calculation was " << chrono::duration_cast<chrono::milliseconds>(t2 - t1).count()/1000
        //      << " ms and for truncated Z was " << chrono::duration_cast<chrono::milliseconds>(t3 - t2).count()/1000 << " s" << endl;
        // // cout << "\t\t(In nanoseconds truncation calculated " <<  chrono::duration_cast<chrono::microseconds>(t3 - t2).count() << ")" << endl;

        vector<double> vals;
        vector<double> times;

        double logZ_AIS[] = {717.842878};
        double time_logZ[] = {177.524293};

        double sumAIS = 0, sumTimes = 0, tmp, tmpT;

        for (int r=1; r <= repeat; r++) {
            model.setRandomSeed(seed + s + r);

            t1 = chrono::steady_clock::now();
            tmp = model.negativeLogLikelihood(bas, logZ_AIS[r-1]);
            t2 = chrono::steady_clock::now();
            // tmpT = chrono::duration_cast<chrono::milliseconds>(t2 - t1).count();
            tmpT = time_logZ[r-1] + chrono::duration_cast<chrono::milliseconds>(t2 - t1).count()/1000;

            sumAIS += tmp;
            sumTimes += tmpT;
            vals.push_back(tmp);
            times.push_back(tmpT);
            cout << "AIS estimation " << r << ": " << tmp << " (took " << tmpT << " s)" << endl;
        }
        sumAIS = sumAIS/repeat;
        sumTimes = sumTimes/repeat;
        double std = 0, stdT = 0;
        for (auto v: vals) { std += pow(sumAIS - v, 2); }
        for (auto t: times) { stdT += pow(sumTimes - t, 2); }
        std = sqrt(std/repeat);
        stdT = sqrt(stdT/repeat);
        cout << "\tAIS mean: " << sumAIS << " ± " << std << "; time to calculate is " << sumTimes << " ± " << stdT << " s" << endl;
        cout << endl;
    }
}