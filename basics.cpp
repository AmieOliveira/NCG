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