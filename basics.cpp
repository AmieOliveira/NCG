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
    // "Crux connectivity". Has 2*(basSize)-1 neighbors, and connects all units
    // on the same row and/or column as a central point
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

int wraparound(int x, int lim) {
    if (x < 0) { return lim + x; }
    if (x >= lim) { return x - lim; }
    return x;
}

Eigen::MatrixXd bas_connect_3(int basSize) {
    // "Convolutional connectivity". Has 9 neighbors, and connects
    // units to a square around a central point
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

    Eigen::MatrixXd ret = Eigen::MatrixXd::Identity(size, size);

    int c, r;
    for (int i=0; i<size; i++) {
        r = int( i/basSize );
        c = i % basSize;

        // (i, i) já está adicionado
        ret( i, basSize*r + wraparound(c+1, basSize) ) = 1;
        ret( i, basSize*r + wraparound(c-1, basSize) ) = 1;
        ret( i, wraparound(i + basSize, size) ) = 1;
        ret( i, basSize * wraparound(r+1, basSize) + wraparound(c+1, basSize) ) = 1;
        ret( i, basSize * wraparound(r+1, basSize) + wraparound(c-1, basSize) ) = 1;
        ret( i, wraparound(i - basSize, size) ) = 1;
        ret( i, basSize * wraparound(r-1, basSize) + wraparound(c+1, basSize) ) = 1;
        ret( i, basSize * wraparound(r-1, basSize) + wraparound(c-1, basSize) ) = 1;
    }

    return ret;
}

Eigen::MatrixXd v_neighbors_spiral(int nRows, int nCols, int nNeighs) {
    if (nNeighs <= 0) {
        string msg = "Invalid number of neighbors, set 1 or higher";
        printError(msg);
        throw;
    }
    Eigen::MatrixXd ret = Eigen::MatrixXd::Identity(nRows, nCols);

    if (nNeighs == 1) {
        return ret;
    }

    int basSize = int(sqrt(nCols));

    for (int i=0; i<nRows; i++) {
        int l;
        int g_r, g_c;
        int v_r, v_c;
        int r = int(i/basSize);
        int c = i % basSize;

        // Values initialization
        if (basSize % 2 == 0) {
            // Has an even-sided square
            l = 2;
            v_r = 0;
            v_c = -1;
            g_r = r;
            g_c = wraparound(c - 1, basSize);
        } else {
            // Has odd sized square
            l = 1;
            v_r = 0;
            v_c = 0;
            g_r = r;
            g_c = c;
        }

        for (int v = 2; v <= nNeighs; v++) {
            // cout << "Current position (" << r << "," << c << "), movement (" << v_r
            //      << "," << v_c << ") next goal (" << g_r << ", " << g_c << ")" << endl;

            // Add next neighbor
            r = wraparound(r + v_r, basSize);
            c = wraparound(c + v_c, basSize);

            if ( ret(i, basSize*r + c) == 1 ) {
                // Finished lap! Begin new one!
                l += 2;
                r = wraparound( r+1, basSize );
                v_r = 0;
                v_c = -1;
                g_r = r;
                g_c = wraparound( c - int(l/2), basSize );

                // cout << "Adding neighbor: (" << i << ", " << basSize*r + c << ")" << endl;
                ret(i, basSize*r + c) = 1;
                continue;
            }

            // cout << "Adding neighbor: (" << i << ", " << basSize*r + c << ")" << endl;
            ret(i, basSize*r + c) = 1;

            if ( (r == g_r) && (c == g_c) ) {
                // Turn
                int tmp = v_r;
                v_r = v_c;
                v_c = - tmp;

                g_r = wraparound( r + (v_r*(l-1)), basSize );
                g_c = wraparound( c + (v_c*(l-1)), basSize );
            }
        }

    }

    return ret;
}

Eigen::MatrixXd bas_connect_4(int basSize) {
    // "Stairs connectivity". Has 2*(basSize) neighbors, and tracks all rows and all
    // columns in the pattern
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

    Eigen::MatrixXd ret = Eigen::MatrixXd::Identity(size, size);

    int c, r;
    for (int i=0; i<size; i++) {
        r = int( i/basSize );
        c = i % basSize;

        // (i, i) já está adicionado
        c = wraparound(c+1, basSize);
        ret( i, basSize*r + c ) = 1;

        for (int l=1; l<basSize; l++) {
            r = wraparound(r+1, basSize);
            ret( i, basSize*r + c ) = 1;
            c = wraparound(c+1, basSize);
            ret( i, basSize*r + c ) = 1;
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