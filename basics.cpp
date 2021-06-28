//
// Helper functions
//

#include "basics.h"

// Log messages
void printError(string msg){
    cout << BOLDRED << "ERROR:\t" << RED << msg << RESET << endl;
}

void printWarning(string msg){
    cout << BOLDYELLOW << "WARNING:\t" << YELLOW << msg << RESET << endl;
}

void printInfo(string msg){
    cout << BOLDGREEN << "INFO:\t" << GREEN << msg << RESET << endl;
}

// Connectivity Matrixes
Eigen::MatrixXd v_neighbors(int nRows, int nCols, int nNeighs) {
    if (nNeighs <= 0) {
        string msg = "Invalid number of neighbors, set 1 or higher";
        printError(msg);
        throw;
    }
    Eigen::MatrixXd ret = Eigen::MatrixXd::Identity(nRows, nCols);

    if (nNeighs == 1) {
        return ret;
    }

    int i, j, n;
    for (i=0; i<nRows; i++) {
        j=i;
        n=nNeighs-1;
        while (n > 0) {
            j++;
            if (j >= nCols) j=j-nCols;
            ret(i,j) = 1;
            n--;
        }
    }

    return ret;
}

Eigen::MatrixXd bas_connect(int basSize) {
    if (basSize < 1) {
        string msg = "Invalid size, should be a positive non null number";
        printError(msg);
        throw;
    } else if (basSize == 1) {
        string msg = "BAS 1x1 is a very trivial case. Check if you've "
                     "estipulated the size correctly";
        printWarning(msg);

        return Eigen::MatrixXd::Constant(1, 1, 1);
    }

    int size = basSize*basSize;

    Eigen::MatrixXd ret = Eigen::MatrixXd::Zero(size, size);

    int unitR = 0, unitC = basSize;

    for (int a = 0; a < basSize; a++) {
        for (int b = 0; b < basSize; b++) {
            // Add unit to correspond to row (a=row, b=col)
            ret(unitR, basSize*a + b) = 1;
            // Add unit to correspond to column (a=col, b=row)
            ret(unitC, basSize*b + a) = 1;
        }
        unitR++;
        unitC++;
    }

    if (basSize == 2) {
        string msg = "BAS 2x2 has 4 variables and four rows/columns. "
                     "Cannot have a unit to connect all.";
        printWarning(msg);

        return ret;
    }

    for (int u = 0; u < size; u++) {
        ret(unitC, u) = 1;
    }
    return ret;
}

// TODO: bas_connect considering possibility of not square matrix
//       After all, the resulting matrix will seldomly be square
//       (only for 2x2, and even then it suppresses a desired hidden unit)
//Eigen::MatrixXd bas_connect(int size, )

Eigen::MatrixXd bas_connect_2(int basSize) {
    if (basSize < 1) {
        string msg = "Invalid size, should be a positive non null number";
        printError(msg);
        throw;
    } else if (basSize == 1) {
        string msg = "BAS 1x1 is a very trivial case. Check if you've "
                     "estipulated the size correctly";
        printWarning(msg);

        return Eigen::MatrixXd::Constant(1, 1, 1);
    }

    int size = basSize*basSize;

    Eigen::MatrixXd ret = Eigen::MatrixXd::Zero(size, size);

    int row = 0, col = 0;
    for (int j = 0; j < size; j++) {
        row = (int) j/basSize;
        col = j % basSize;

        for (int a = 0; a < basSize; a++) {
            ret(j, basSize*row + a) = 1;  // Adding row
            ret(j, basSize*a + col) = 1;  // Adding column
        }
    }

    return ret;
}


// Randomizing Connectivity
Mixer::Mixer(unsigned s) {
    seed = s;
    generator.seed(seed);
}
Mixer::Mixer() {
    random_device rd;
    seed = rd();
    generator.seed(seed);
}

Eigen::MatrixXd Mixer::mix_neighbors(Eigen::MatrixXd regPattern, int iter) {
    // This function considers that the given pattern is regular, that is,
    //      all vertices of a layer have the same degree
    if (iter == 0) {
        string msg = "Given argument for zero iterations, no modifications made!";
        printWarning(msg);
        return regPattern;
    }

    int H = regPattern.rows();
    int X = regPattern.cols();
    vector<int> mapping[H];
    int edges_h = 0;
    for (int j=0; j<X; j++) {
        if (regPattern(0,j) == 1) {
            edges_h++;
            mapping[0].push_back(j);
        }

        for (int i=1; i<H; i++) {
            if (regPattern(i,j) == 1) mapping[i].push_back(j);
        }
    }
    //cout << "Matrix " << H << "x" << X << ". " << edges_h << " neighbors per h." << endl;

    uniform_int_distribution<int> rdnH(0, H-1);
    uniform_int_distribution<int> rdn_edge(0, edges_h-1);

    Eigen::MatrixXd ret = regPattern;

    int i1, i2, j1, j2;
    int jIdx1, jIdx2;
    bool same;

    for (int it=0; it < iter; it++) {
        i1 = rdnH(generator);
        jIdx1 = rdn_edge(generator);
        j1 = mapping[i1].at(jIdx1);

        same = true;

        while (same) {
            i2 = rdnH(generator);
            jIdx2 = rdn_edge(generator);
            j2 = mapping[i2].at(jIdx2);
            same = ((i1 == i2) && (j1 == j2));
        }


        ret(i1,j1) = 0;
        ret(i2,j2) = 0;

        if ( (ret(i1,j2) == 1) || (ret(i2,j1) == 1) ) {
            // Invalid switch!
            ret(i1,j1) = 1;
            ret(i2,j2) = 1;
            it--;
            continue;
        }
        ret(i1,j2) = 1;
        ret(i2,j1) = 1;

        mapping[i1].at(jIdx1) = j2;
        mapping[i2].at(jIdx2) = j1;

        // cout << "Old: (" << i1 << "," << j1 << "), ("  << i2 << "," << j2 << ")." << endl;
        // cout << "New: (" << i1 << "," << j2 << "), ("  << i2 << "," << j1 << ")." << endl;
        // cout << ret << endl;
    }

    return ret;

}

// TODO: Mixing when the pattern is not regular
//       (will have to sample h_i according to degrees...)

/*
void plotVectorPython(vector<double>) {
    printInfo("Calling python to plot vector");
    Py_Initialize();

    PyObject *pName, *pModule, *pDict, *pFunc, *pArgs, *pValue;
    pName = PyString_FromString("pyprinter.py");
    pModule = PyImport_Import(pName);
    pDict = PyModule_GetDict(pModule);
    pFunc = PyDict_GetItemString(pDict, "test");
    pArgs = PyTuple_New(2);
    pValue = PyInt_FromLong(2);
    PyTuple_SetItem(pArgs, 0, pValue);

    PyObject* pResult = PyObject_CallObject(pFunc, pArgs);
    if (pResult == NULL)
        cout << "Calling the test method failed.\n";

    long result = PyInt_AsLong(pResult);
    Py_Finalize();
}
*/