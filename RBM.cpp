//
// Created by Amanda Oliveira on 04/05/21.
//

#include "RBM.h"

// Constructors
RBM::RBM() {
    initialized = false;
}
RBM::RBM(int X, int H) {
    initialized = true;
    patterns = false;
    hasSeed = false;

    xSize = X;
    hSize = H;

    x = VectorXd::Zero(X);
    h = VectorXd::Zero(H);

    // According to reference: initializing with small
    //      random weights and zero bias
    d = VectorXd::Zero(X);
    b = VectorXd::Zero(H);

    W = MatrixXd::Zero(H,X); // TODO: Criar função inicializadora
    A = MatrixXd::Constant(H,X,1);

    //C = W; // TODO: Inicializar como zeros aqui
}
RBM::RBM(int X, int H, bool use_pattern) {
    RBM(X,H);
    patterns = use_pattern;
}


// Connectivity (de)ativation
void RBM::connectivity(bool activate) {
    if (activate == patterns) {
        printWarning("Tried to set connectivity activation to a status it already had");
    }
    patterns = activate;
    if (patterns){
        C = W.cwiseProduct(A);
    }
}


// Set dimensions
void RBM::setDimensions(int X, int H) {
    RBM(X, H);
}
void RBM::setDimensions(int X, int H, bool use_pattern) {
    RBM(X, H, use_pattern);
}


// Set and get variables
int RBM::setVisibleUnits(VectorXd vec) {
    if (!initialized){
        string errorMessage = "Tried to set a vector that has no dimension!\n\t"
                              "You need to set the RBM dimensions before "
                              "assigning values!";
        printError(errorMessage);
        return 1;
    }

    if (vec.size() == x.size()) {
        x = vec;
        return 0;
    } else {
        string errorMessage = "Tried to set visible units with wrong size! "
                              "Cancelled operation.";
        printError(errorMessage);
        return 2;
    }
}
// void setVisibleUnits(int* vec){}
VectorXd RBM::getVisibleUnits() {
    return x;
}

int RBM::setHiddenUnits(VectorXd vec) {
    if (!initialized){
        string errorMessage = "Tried to set a vector that has no dimension!\n\t"
                              "You need to set the RBM dimensions before "
                              "assigning values!";
        printError(errorMessage);
        return 1;
    }

    if (vec.size() == h.size()) {
        h = vec;
        return 0;
    } else {
        string errorMessage = "Tried to set hidden units with wrong size! "
                              "Cancelled operation.";
        printError(errorMessage);
        return 2;
    }
}
VectorXd RBM::getHiddenUnits() {
    return h;
}

VectorXd RBM::getVisibleBiases() {
    return d;
}

VectorXd RBM::getHiddenBiases() {
    return b;
}

void RBM::startBiases() {
    // Implementado para teste. Atribui valores aleatórios entre -1 e
    // +1 para cada viés

    if (!initialized){
        string errorMessage = "Tried to set a matrix that has no dimension!\n\t"
                              "You need to set the RBM dimensions before "
                              "assigning values!";
        printError(errorMessage);
        return;
    }
    if (!hasSeed){
        string errorMessage;
        errorMessage = "Tried to sample vector without random seed!";
        printError(errorMessage);
        throw runtime_error(errorMessage);
    }

    double rdn, bias;
    for (int i = 0; i < hSize; i++){
        rdn = (*p_dis)(generator);
        bias = 2*rdn - 1;
        b(i) = bias;
    }
    for (int j = 0; j < xSize; ++j) {
        rdn = (*p_dis)(generator);
        bias = 2*rdn - 1;
        d(j) = bias;
    }
}

MatrixXd RBM::getWeights() {
    return W;
}
int RBM::setWeights(MatrixXd mat) {
    if (!initialized){
        string errorMessage = "Tried to set a matrix that has no dimension!\n\t"
                              "You need to set the RBM dimensions before "
                              "assigning values!";
        printError(errorMessage);
        return 1;
    }

    if ((mat.size() == W.size()) && (mat.rows() == W.rows())) {
        W = mat;
        if (patterns){
            C = W.cwiseProduct(A);
        } //else {C = W;} TODO: Verify
        return 0;
    } else {
        string errorMessage = "Tried to set weight matrix with wrong size! "
                              "Cancelled operation.";
        printError(errorMessage);
        return 2;
    }
}
void RBM::startWeigths() {
    // Implementação inicial: estou inicializando com pesos aleatórios
    // distribuídos uniformemente entre -1 e +1

    if (!initialized){
        string errorMessage = "Tried to set a matrix that has no dimension!\n\t"
                              "You need to set the RBM dimensions before "
                              "assigning values!";
        printError(errorMessage);
        return;
    }
    if (!hasSeed){
        string errorMessage;
        errorMessage = "Tried to sample vector without random seed!";
        printError(errorMessage);
        throw runtime_error(errorMessage);
    }

    double rdn, weight;
    for (int i = 0; i < hSize; i++){
        for (int j = 0; j < xSize; ++j) {
            rdn = (*p_dis)(generator);
            weight = 2*rdn - 1;
            W(i,j) = weight;
        }
    }
}

MatrixXd RBM::getConnectivity() {
    return A;
}
int RBM::setConnectivity(MatrixXd mat) {
    if (!initialized){
        string errorMessage = "Tried to set a matrix that has no dimension!\n\t"
                              "You need to set the RBM dimensions before "
                              "assigning values!";
        printError(errorMessage);
        return 1;
    }
    if (!patterns) {
        string errorMessage = "Tried to set connectivity matrix, but it is not active "
                              "for this RBM!";
        printError(errorMessage);
        return 1;
    }

    if ((mat.size() == A.size()) && (mat.rows() == A.rows())) {
        A = mat;
        C = W.cwiseProduct(A);
        return 0;
    } else {
        string errorMessage = "Tried to set connectivity matrix with wrong size! "
                              "Cancelled operation.";
        printError(errorMessage);
        return 2;
    }
}


// RBM probabilities
VectorXd RBM::getProbabilities_x() {
    if (!initialized){
        string errorMessage;
        errorMessage = "Tried to sample vector that has no dimension!\n\t"
                       "You need to set the RBM dimensions before "
                       "sampling values!";
        printError(errorMessage);
        throw runtime_error("Tried to sample vector that has no dimension!");
    }

    MatrixXd* p_W;
    if (patterns) {
        p_W = &C;
    } else {
        p_W = &W;
    }

    VectorXd output(xSize);
    RowVectorXd vAux = h.transpose()*(*p_W);

    for (int j=0; j<xSize; j++){
        output(j) = 1.0/( 1 + exp( - d(j) -  vAux(j) ) );
    }

    return output;
}
VectorXd RBM::getProbabilities_h() {
    if (!initialized){
        string errorMessage;
        errorMessage = "Tried to sample vector that has no dimension!\n\t"
                       "You need to set the RBM dimensions before "
                       "sampling values!";
        printError(errorMessage);
        throw runtime_error("Tried to sample vector that has no dimension!");
    }

    MatrixXd* p_W;
    if (patterns) {
        p_W = &C;
    } else {
        p_W = &W;
    }

    VectorXd output(hSize);
    VectorXd vAux = (*p_W)*x;

    for (int i=0; i<hSize; i++){
        output(i) = 1.0/( 1 + exp( - b(i) -  vAux(i) ) );
    }

    return output;
}


// Random generator functions
void RBM::setRandomSeed(unsigned int seed) {
    hasSeed = true;
    generator.seed(seed);
    p_dis = new uniform_real_distribution<double>(0.0, 1.0);
}

// Sampling methods
VectorXd RBM::sampleXfromH() {
    // NOTE: Will not check if has been initialized and/or has seed,
    //   because this method should be called only by the RBM itself
    //   and other functions should have performed the necessary
    //   checks

    /*if (!initialized){
        string errorMessage;
        errorMessage = "Tried to sample vector that has no dimension!\n\t"
                       "You need to set the RBM dimensions before "
                       "sampling values!";
        printError(errorMessage);
        throw runtime_error("Tried to sample vector that has no dimension!");
    }
    if (!hasSeed){
        string errorMessage;
        errorMessage = "Tried to sample vector without random seed!";
        printError(errorMessage);
        throw runtime_error(errorMessage);
    }*/
    //cout << "Sampling x!" << endl;

    MatrixXd* p_W;
    if (patterns) {
        p_W = &C;
    } else {
        p_W = &W;
    }

    VectorXd output(xSize);
    RowVectorXd vAux = h.transpose()*(*p_W);

    double prob, moeda;
    for (int j=0; j<xSize; j++){
        prob = 1.0/( 1 + exp( - d(j) -  vAux(j) ) );
        moeda = (*p_dis)(generator);
        //cout << "Probabilidade: " << prob << ", numero aleatorio: " << moeda << endl;

        if (moeda < prob)
            output(j) = 1;
        else
            output(j) = 0;
    }
    //cout << output.transpose() << endl;
    return output;
}
VectorXd RBM::sampleHfromX() {
    // NOTE: Will not check if has been initialized and/or has seed,
    //   because this method should be called only by the RBM itself
    //   and other functions should have performed the necessary
    //   checks

    /*if (!initialized){
        string errorMessage;
        errorMessage = "Tried to sample vector that has no dimension!\n\t"
                       "You need to set the RBM dimensions before "
                       "sampling values!";
        printError(errorMessage);
        throw runtime_error("Tried to sample vector that has no dimension!");
    }
    if (!hasSeed){
        string errorMessage;
        errorMessage = "Tried to sample vector without random seed!";
        printError(errorMessage);
        throw runtime_error(errorMessage);
    }*/
    //cout << "Sampling h!" << endl;

    MatrixXd* p_W;
    if (patterns) {
        p_W = &C;
    } else {
        p_W = &W;
    }

    VectorXd output(hSize);
    VectorXd vAux = (*p_W)*x;

    double prob, moeda;
    for (int i=0; i<hSize; i++){
        prob = 1.0/( 1 + exp( - b(i) -  vAux(i) ) );
        moeda = (*p_dis)(generator);
        //cout << "Probabilidade: " << prob << ", numero aleatorio: " << moeda << endl;

        if (moeda < prob)
            output(i) = 1;
        else
            output(i) = 0;
    }

    return output;
}

vector<VectorXd> RBM::sampleXtilde(SampleType sType,
                                   int k, //int b_size,
                                   vector<VectorXd> vecs) {
    // NOTE: Will not check if has been initialized and/or has seed,
    //   because this method should be called only as part of the
    //   RBM training, and that one should already have performed
    //   the necessary checks

    //MatrixXd* p_W;
    //if (patterns) {
    //    p_W = &C;
    //} else {
    //    p_W = &W;
    //}
    vector<VectorXd> ret;

    switch (sType) {
        case SampleType::CD:
            //cout << "Beginning Contrastive Divergence!" << endl;
            for (vector<VectorXd>::iterator x_0 = vecs.begin();
                 x_0 != vecs.end();
                 ++x_0) {
                x = *x_0;
                //cout << "x0 = " << x.transpose() << endl;

                for (int t=0; t<k; t++){
                    h = sampleHfromX();
                    x = sampleXfromH();

                    //cout << "x^" << t+1 << " = " << x.transpose() << endl;
                }
                ret.push_back(x);
            }
            break;

        default:
            string errorMessage = "Sample type not implemented";
            printError(errorMessage);
            throw runtime_error(errorMessage);

        // TODO: Outros tipos de treinamento
        // case SampleType::PCD:
    }
    return ret;
}


// Training methods
void RBM::fit(){
    // Verificar se teve setup, ou, no caso de eu nao implementar setup,
    // se foi inicializado, etc.
    // FIXME: Implementado somente o teste da sampleXtilde!
    SampleType stype = CD;
    VectorXd x1, x2;

    x1 = VectorXd::Zero(xSize);
    x1(0) = 1;
    x2 = VectorXd::Constant(xSize, 1);
    x2(xSize-1) = 0;

    vector<VectorXd> vecIn, vecOut;
    vecIn.push_back(x1);

    cout << "Input: ";
    for (auto i: vecIn)
        cout << "[" << i.transpose() << "] ";
    cout << endl;
    vecOut = sampleXtilde(stype, 4, vecIn);
    cout << "Output: ";
    for (auto i: vecOut)
        cout << "[" << i.transpose() << "] ";
    cout << endl;

    vecIn.push_back(x2);

    cout << "Input: ";
    for (auto i: vecIn)
        cout << "[" << i.transpose() << "] ";
    cout << endl;
    vecOut = sampleXtilde(stype, 4, vecIn);
    cout << "Output: ";
    for (auto i: vecOut)
        cout << "[" << i.transpose() << "] ";
    cout << endl;
}

// Test Functions
void RBM::printVariables() {
    cout << "RBM variables:\t(" << xSize << " visible units and " << hSize << " hidden units)" << endl;
    cout << "----------------------------------------------------------------" << endl;
    cout << "\t\tVisible Units: " << x.transpose() << endl;
    cout << "\t\tHidden Units: " << h.transpose() << endl;
    cout << "\t\tVisible Biases: " << d.transpose() << endl;
    cout << "\t\tHidden Biases: " << b.transpose() << endl;
    cout << "\t\tWeights: " << endl << W << endl;
    cout << "\t\tConnectivity (" << patterns << "): " << endl << A << endl;
    cout << "\t\tMatrix C: " << endl << C << endl;
    cout << "----------------------------------------------------------------" << endl;
    cout << endl;
}

/* double RBM::getRandomNumber() {
    if (!hasSeed){
        string errorMessage = "Tried to get random number with no random seed set!";
        printError(errorMessage);
        throw runtime_error(errorMessage);
    }
    return (*p_dis)(generator);
} */

void RBM::sampleXH() {
    VectorXd vec = sampleXfromH();
    cout << "x sampled: " << vec.transpose() << endl;
    vec = sampleHfromX();
    cout << "h sampled: " << vec.transpose() << endl;
}

void RBM::generatorTest() {
    if (!initialized){
        string errorMessage;
        errorMessage = "Tried to sample vector that has no dimension!\n\t"
                       "You need to set the RBM dimensions before "
                       "sampling values!";
        printError(errorMessage);
        throw runtime_error("Tried to sample vector that has no dimension!");
    }
    if (!hasSeed){
        string errorMessage;
        errorMessage = "Tried to sample vector without random seed!";
        printError(errorMessage);
        throw runtime_error(errorMessage);
    }

    int repeat = 1000;

    double sumOfValues = 0;
    double value;
    double pseudoHist[10] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    int idx;

    for (int n = 0; n < repeat; ++n) {
        // Use dis to transform the random unsigned int generated by gen into a
        // double in [1, 2). Each call to dis(gen) generates a new random double
        value = (*p_dis)(generator);
        //std::cout << value << ' ';
        sumOfValues = sumOfValues + value;

        idx = int(value*10);
        pseudoHist[idx] += 1;
    }
    std::cout << '\n';
    std::cout << "Mean = " << sumOfValues/repeat << std::endl;
    std::cout << "Distribution: | ";
    for (int i = 0; i < 10; ++i) {
        std::cout << pseudoHist[i] << " | ";
    }
    std::cout << std::endl;
}

void RBM::validateSample(unsigned seed, int rep) {
    // Função para a validação do gerador aleatório

    if (!initialized){
        string errorMessage;
        errorMessage = "Tried to sample vector that has no dimension!\n\t"
                       "You need to set the RBM dimensions before "
                       "sampling values!";
        printError(errorMessage);
        throw runtime_error("Tried to sample vector that has no dimension!");
    }

    // Setup RBM
    setRandomSeed(seed);
    startWeigths();
    startBiases();
    for (int k = 0; k < 5; ++k) {
        // Mix a bit x and h, to have non trivial probabilities
        h << sampleHfromX();
        x << sampleXfromH();
    }
    printVariables();

    // Auxiliary variables
    VectorXd probX = getProbabilities_x();
    VectorXd probH = getProbabilities_h();

    VectorXd freqX = VectorXd::Zero(xSize);
    VectorXd freqH = VectorXd::Zero(hSize);

    for (int k = 0; k < rep; ++k) {
        freqX = freqX + sampleXfromH();
        freqH = freqH + sampleHfromX();
    }
    freqX = freqX/rep;
    freqH = freqH/rep;

    cout << endl << "Random Generator Analysis: " << endl;
    cout << "----------------------------------------------------" << endl;
    cout << "\tProbabilities of x: " << probX.transpose() << endl;
    cout << "\tFrequency of x:     " << freqX.transpose() << endl;
    cout << "\tError (p - f):      " << (probX - freqX).transpose() << endl;
    cout << "----------------------------------------------------" << endl;
    cout << "\tProbabilities of h: " << probH.transpose() << endl;
    cout << "\tFrequency of h:     " << freqH.transpose() << endl;
    cout << "\tError (p - f):      " << (probH - freqH).transpose() << endl;
    cout << "----------------------------------------------------" << endl << endl;

}