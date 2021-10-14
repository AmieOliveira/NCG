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
    int iter = 900;
    int b_size = 50;  // 5
    double l_rate = 0.1;
    // double p = 1;
    unsigned seed = 71263;  // 232140824 9217
    bool shuffleData = true;

    int repeat = 5;
    int n_seeds = 1;

    //// BAS Dataset
    Data bas(BASnoRep, 4);
    //Data bas(seed, BASnoRep, 28, 1000);
    //Data bas("Datasets/bin_mnist-test.data", false); bas = bas.separateTrainTestSets(0.1).at(0);

    // Traditional RBM
    RBM model;

    cout << "Training model for " << iter << " epochs" << endl;

    for (int s = 0; s < n_seeds; s++) {
        cout << "Experiment " << s+1 << " of " << n_seeds << endl;

        model.setDimensions(bas.get_sample_size(), 500, false);
        model.setRandomSeed(seed + s);
        model.trainSetup(SampleType::CD, k, iter, b_size, l_rate, false, 1, shuffleData);
        // cout << "Before training: " << model.negativeLogLikelihood(bas) << endl;

        model.fit(bas);
        // model.optSetup(Heuristic::SGD, 1); model.fit_connectivity(bas);

        auto t1 = chrono::steady_clock::now();
        cout << "Exact value: " << model.negativeLogLikelihood(bas, ZEstimation::None) << endl;
        auto t2 = chrono::steady_clock::now();
        cout << "Truncation estimation (no verification): " << model.negativeLogLikelihood(bas, ZEstimation::Trunc) << endl;
        auto t3 = chrono::steady_clock::now();

        cout << "Time elapsed for total calculation was " << chrono::duration_cast<chrono::milliseconds>(t2 - t1).count()
             << " ms and for truncated Z was " << chrono::duration_cast<chrono::milliseconds>(t3 - t2).count() << " ms" << endl;

        vector<double> vals;
        vector<double> times;

        double sumMC = 0, sumTimes = 0, tmp, tmpT;
        for (int r=1; r <= repeat; r++) {
            model.setRandomSeed(seed + s + r);

            t1 = chrono::steady_clock::now();
            tmp = model.negativeLogLikelihood(bas, ZEstimation::MC);
            t2 = chrono::steady_clock::now();
            tmpT = chrono::duration_cast<chrono::milliseconds>(t2 - t1).count();

            sumMC += tmp;
            sumTimes += tmpT;
            vals.push_back(tmp);
            times.push_back(tmpT);
            cout << "MC estimation " << r << ": " << tmp << " (took " << tmpT << " ms)" << endl;
        }
        sumMC = sumMC/repeat;
        sumTimes = sumTimes/repeat;
        double std = 0, stdT = 0;
        for (auto v: vals) { std += pow(sumMC - v, 2); }
        for (auto t: times) { stdT += pow(sumMC - t, 2); }
        std = sqrt(std/repeat);
        stdT = sqrt(stdT/repeat);
        cout << "\tMC mean: " << sumMC << " ± " << std << "; time to calculate is " << sumTimes << " ± " << stdT << " ms" << endl;

        double sumAIS = 0;
        sumTimes = 0;
        for (int r=1; r <= repeat; r++) {
            model.setRandomSeed(seed + s + r);

            t1 = chrono::steady_clock::now();
            tmp = model.negativeLogLikelihood(bas, ZEstimation::AIS);
            t2 = chrono::steady_clock::now();
            tmpT = chrono::duration_cast<chrono::milliseconds>(t2 - t1).count();

            sumAIS += tmp;
            sumTimes += tmpT;
            vals.at(r-1) = tmp;
            times.at(r-1) = tmpT;
            cout << "AIS estimation " << r << ": " << tmp << " (took " << tmpT << " ms)" << endl;
        }
        sumAIS = sumAIS/repeat;
        sumTimes = sumTimes/repeat;
        std = 0; stdT = 0;
        for (auto v: vals) { std += pow(sumAIS - v, 2); }
        for (auto t: times) { stdT += pow(sumMC - t, 2); }
        std = sqrt(std/repeat);
        stdT = sqrt(stdT/repeat);
        cout << "\tAIS mean: " << sumAIS << " ± " << std << "; time to calculate is " << sumTimes << " ± " << stdT << " ms" << endl;
        cout << endl;
    }
}