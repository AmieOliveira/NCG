//
// Created by Amanda Oliveira on 24/05/21.
//

#ifndef TESE_CÓDIGO_BASICS_H
#define TESE_CÓDIGO_BASICS_H

namespace colors {
    // Colors
    #define RESET   "\033[0m"
    #define RED     "\033[31m"
    #define GREEN   "\033[32m"
    #define YELLOW  "\033[33m"
    #define BOLDRED     "\033[1m\033[31m"
    #define BOLDGREEN   "\033[1m\033[32m"
    #define BOLDYELLOW  "\033[1m\033[33m"
}

#include <iostream>
#include <string>
using namespace std;
using namespace colors;

// Log messages
void printError(string msg);
void printWarning(string msg);
void printInfo(string msg);

#endif //TESE_CÓDIGO_BASICS_H
