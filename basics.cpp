//
// Created by Amanda Oliveira on 24/05/21.
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