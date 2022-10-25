//
// RBM implementation
//

#include "RBM.h"

// Constructors
RBM::RBM() {
    initialized = false;
}
RBM::RBM(int X, int H) {
    initialized = true;
    patterns = false;
    optim_H = false;
    initializer(X, H);
}
RBM::RBM(int X, int H, bool use_pattern) {
    initialized = true;
    patterns = use_pattern;
    optim_H = use_pattern;
    initializer(X,H);
}

void RBM::initializer(int X, int H) {
    hasSeed = false;
    trainReady = false;
    optReady = false;
    isTrained = false;

    xSize = X;
    hSize = H;

    x = VectorXd::Zero(X);
    h = VectorXd::Zero(H);

    // According to reference: initializing with small
    //      random weights and zero bias
    d = VectorXd::Zero(X);
    b = VectorXd::Zero(H);

    W = MatrixXd::Zero(H,X);
    A = MatrixXd::Constant(H,X,1);

    s = VectorXd::Constant(H,1);

    if (optim_H) {
        // v = b.cwiseProduct(s);
        v = b;
        p_b = &v;
    }
    else p_b = &b;

    if (patterns) {
        // C = W.cwiseProduct(A);
        // NOTE: No need to compute the product since A is a matrix of ones!
        C = W;
        p_W = &C;
    }
    else p_W = &W;

    auxH = VectorXd::Zero(H);
    auxX = RowVectorXd::Zero(X);
}

// Helpers
void RBM::inhibitA() {
    // Applies s (inhibittor of hidden units) to A (connectivity matrix)
    // FIXME: verificar eficiencia!

    // A = A.cwiseProduct( s * VectorXd::Constant(xSize, 1).transpose() );
    for (int i=0; i < hSize; i++) {
        for (int j=0; j < xSize; j++) {
            A(i,j) = s(i);
        }
    }
}

// Connectivity (de)ativation
void RBM::connectivity(bool activate) {
    if (activate == patterns) {
        printWarning("Tried to set connectivity activation to a status it already had");
        return;
    }
    patterns = activate;
    if (patterns){
        C = W.cwiseProduct(A);
        p_W = &C;
    } else {
        p_W = &W;
    }
}

void RBM::hidden_activation(bool active) {
    if (active == optim_H) {
        printWarning("Tried to set hidden neurons activation to a status it already had");
        return;
    }
    optim_H = active;
    if (optim_H) {
        v = b.cwiseProduct(s);
        p_b = &v;

        if (!patterns) {
            patterns = true;
            inhibitA();
            C = W.cwiseProduct(A);
            p_W = &C;
        }
    } else {
        p_b = &b;
    }
}


// Set dimensions
void RBM::setDimensions(int X, int H) {
    if (!initialized) {
        initialized = true;
        optim_H = false;
        patterns = false;
    }
    initializer(X, H);
}
void RBM::setDimensions(int X, int H, bool use_pattern) {
    if (!initialized) initialized = true;
    optim_H = use_pattern;
    patterns = use_pattern;
    initializer(X, H);
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
int RBM::setVisibleBiases(VectorXd vec) {
    if (!initialized){
        string errorMessage = "Tried to set a vector that has no dimension!\n\t"
                              "You need to set the RBM dimensions before "
                              "assigning values!";
        printError(errorMessage);
        return 1;
    }

    if (vec.size() == d.size()) {
        d = vec;
        return 0;
    } else {
        string errorMessage = "Tried to set visible biases with wrong size! "
                              "Cancelled operation.";
        printError(errorMessage);
        return 2;
    }
}

VectorXd RBM::getHiddenBiases() {
    return *p_b;
}
int RBM::setHiddenBiases(VectorXd vec) {
    if (!initialized){
        string errorMessage = "Tried to set a vector that has no dimension!\n\t"
                              "You need to set the RBM dimensions before "
                              "assigning values!";
        printError(errorMessage);
        return 1;
    }

    if (vec.size() == b.size()) {
        b = vec;
        if (optim_H) {
            v = b.cwiseProduct(s);
        }

        return 0;
    } else {
        string errorMessage = "Tried to set hidden biases with wrong size! "
                              "Cancelled operation.";
        printError(errorMessage);
        return 2;
    }
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

    if (optim_H) {
        v = b.cwiseProduct(s);
    }
}

MatrixXd RBM::getWeights() {
    return *p_W;
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
void RBM::startWeights() {
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
    if (patterns) {C = W.cwiseProduct(A);}
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

void RBM::startConnectivity(double p) {
    // Initializes the connectivity pattern randomly according to the
    //      probability p of any given edge existing
    if (!initialized){
        string errorMessage = "Tried to set a matrix that has no dimension!\n\t"
                              "You need to set the RBM dimensions before "
                              "assigning values!";
        printError(errorMessage);
        exit(1);
    }
    if (!patterns) {
        string errorMessage = "Tried to set connectivity matrix, but it is not active "
                              "for this RBM!";
        printError(errorMessage);
        exit(1);
    }
    if ( (p < 0) || (p > 1) ) {
        string errorMessage = "Tried to set connectivity matrix, "
                              "with invalid probability value!";
        printError(errorMessage);
        cerr << "Probability p = " << p << " is not valid. Please assign a value " <<
                "between 0 and 1" << endl;
        exit(1);
    }


    double moeda;

    for (int i=0; i<hSize; i++) {
        for (int j=0; j < xSize-nLabels; j++) {
            moeda = (*p_dis)(generator);

            if (moeda < p) A(i,j) = 1;
            else A(i,j) = 0;
        }
    }

    // cout << "Connecttivity matrix, initialated with p = " << p << endl;
    // cout << A << endl;
    // cout << "Percentage of connections activated: " << A.sum()/(xSize*hSize) << endl;
}

VectorXd RBM::getActiveHiddenUnits() {
    return s;
}

int RBM::setActiveHiddenUnits(VectorXd vec) {
    if (!initialized){
        string errorMessage = "Tried to set a vector that has no dimension!\n\t"
                              "You need to set the RBM dimensions before "
                              "assigning values!";
        printError(errorMessage);
        return 1;
    }
    if (!optim_H) {
        string errorMessage = "Tried to set hidden units activation, but this feature is "
                              "inactive for this RBM!";
        printError(errorMessage);
        return 1;
    }

    if (vec.size() == s.size()) {
        s = vec;
        v = b.cwiseProduct(s);
        inhibitA();
        C = W.cwiseProduct(A);
        return 0;
    } else {
        string errorMessage = "Tried to set hidden units activation with wrong size! "
                              "Cancelled operation.";
        printError(errorMessage);
        return 2;
    }
}

void RBM::startActiveHiddenUnits(double p) {
    // Initialize the hidden units activation with a probability p of each unit being active
    if (!initialized){
        string errorMessage = "Tried to set a vector that has no dimension!\n\t"
                              "You need to set the RBM dimensions before "
                              "assigning values!";
        printError(errorMessage);
        exit(1);
    }
    if (!optim_H) {
        string errorMessage = "Tried to set hidden units activation, but this feature is "
                              "inactive for this RBM!";
        printError(errorMessage);
        exit(1);
    }
    if ( (p < 0) || (p > 1) ) {
        string errorMessage = "Tried to set hidden units activation, with invalid "
                              "probability value!";
        printError(errorMessage);
        cerr << "Probability p = " << p << " is not valid. Please assign a value " <<
                "between 0 and 1" << endl;
        exit(1);
    }

    double moeda;

    for (int i=0; i<hSize; i++) {
        moeda = (*p_dis)(generator);

        if (moeda < p) s(i) = 1;
        else s(i) = 0;
    }
}

MatrixXd RBM::getNonInhibitedWeights() {
    return W;
}
VectorXd RBM::getNonInhibitedBiases() {
    return b;
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

    VectorXd output(xSize);
    auxX = h.transpose() * (*p_W);

    for (int j=0; j<xSize; j++){
        output(j) = 1.0/( 1 + exp( - d(j) -  auxX(j) ) );
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

    VectorXd output(hSize);
    auxH = (*p_W)*x;

    for (int i=0; i<hSize; i++){
        output(i) = 1.0/( 1 + exp( - (*p_b)(i) -  auxH(i) ) );
    }

    return output;
}

void RBM::getProbabilities_h(VectorXd & output) {
    if (!initialized){
        string errorMessage;
        errorMessage = "Tried to sample vector that has no dimension!\n\t"
                       "You need to set the RBM dimensions before "
                       "sampling values!";
        printError(errorMessage);
        throw runtime_error("Tried to sample vector that has no dimension!");
    }

    auxH = (*p_W) * x;

    for (int i=0; i<hSize; i++){
        output(i) = 1.0/( 1 + exp( - (*p_b)(i) -  auxH(i) ) );
    }
}
void RBM::getProbabilities_h(VectorXd & output, VectorXd & x_vec) {
    if (!initialized){
        string errorMessage;
        errorMessage = "Tried to sample vector that has no dimension!\n\t"
                       "You need to set the RBM dimensions before "
                       "sampling values!";
        printError(errorMessage);
        throw runtime_error("Tried to sample vector that has no dimension!");
    }

    auxH = (*p_W) * x_vec;

    for (int i=0; i<hSize; i++){
        output(i) = 1.0/( 1 + exp( - (*p_b)(i) -  auxH(i) ) );
    }
}


// Random generator functions
void RBM::setRandomSeed(unsigned int seed) {
    hasSeed = true;
    generator.seed(seed);
    p_dis = new uniform_real_distribution<double>(0.0, 1.0);
}

// Sampling methods
void RBM::sample_x() {
    // NOTE: Will not check if has been initialized and/or has seed,
    //   because this method should be called only by the RBM itself
    //   and other functions should have performed the necessary
    //   checks

    //cout << "Sampling x!" << endl;

    auxX = h.transpose()*(*p_W);
    // NOTE: Não uso getProbabilities porque aproveito o loop

    double prob, moeda;
    for (int j=0; j<xSize; j++){
        prob = 1.0/( 1 + exp( - d(j) -  auxX(j) ) );
        moeda = (*p_dis)(generator);
        //cout << "Probabilidade: " << prob << ", numero aleatorio: " << moeda << endl;

        if (moeda < prob)
            x(j) = 1;
        else
            x(j) = 0;
    }
}

void RBM::sample_x(VectorXd & h_vec) {
    // NOTE: Not checking if has been initialized and/or has seed,
    //   because this method should be called only by the RBM itself
    //   and other functions should have performed the necessary
    //   checks

    //cout << "Sampling x!" << endl;

    auxX = h_vec.transpose()*(*p_W);
    // NOTE: Não uso getProbabilities porque aproveito o loop

    double prob, moeda;
    for (int j=0; j<xSize; j++){
        prob = 1.0/( 1 + exp( - d(j) -  auxX(j) ) );
        moeda = (*p_dis)(generator);
        //cout << "Probabilidade: " << prob << ", numero aleatorio: " << moeda << endl;

        if (moeda < prob)
            x(j) = 1;
        else
            x(j) = 0;
    }
}

VectorXd RBM::sample_xout() {
    // NOTE: Will not check if has been initialized and/or has seed,
    //   because this method should be called only by the RBM itself
    //   and other functions should have performed the necessary
    //   checks

    //cout << "Sampling x!" << endl;

    VectorXd out(xSize);
    auxX = h.transpose()*(*p_W);
    // NOTE: Não uso getProbabilities porque aproveito o loop

    double prob, moeda;
    for (int j=0; j<xSize; j++){
        prob = 1.0/( 1 + exp( - d(j) -  auxX(j) ) );
        moeda = (*p_dis)(generator);
        //cout << "Probabilidade: " << prob << ", numero aleatorio: " << moeda << endl;

        if (moeda < prob)
            out(j) = 1;
        else
            out(j) = 0;
    }

    return out;
}

void RBM::sample_h() {
    // NOTE: Will not check if has been initialized and/or has seed,
    //   because this method should be called only by the RBM itself
    //   and other functions should have performed the necessary
    //   checks

    auxH = (*p_W)*x;
    // NOTE: Não uso getProbabilities porque aproveito o loop

    double prob, moeda;
    for (int i=0; i<hSize; i++){
        prob = 1.0/( 1 + exp( - (*p_b)(i) -  auxH(i) ) );
        moeda = (*p_dis)(generator);
        // cout << "Probabilidade: " << prob << ", numero aleatorio: " << moeda << endl;

        if (moeda < prob)
            h(i) = 1;
        else
            h(i) = 0;
    }
}

void RBM::sample_h(VectorXd & x_vec) {
    // NOTE: Will not check if has been initialized and/or has seed,
    //   because this method should be called only by the RBM itself
    //   and other functions should have performed the necessary
    //   checks

    //cout << "Sampling h!" << endl;

    auxH = (*p_W)*x_vec;
    // NOTE: Não uso getProbabilities porque aproveito o loop

    double prob, moeda;
    for (int i=0; i<hSize; i++){
        prob = 1.0/( 1 + exp( - (*p_b)(i) -  auxH(i) ) );
        moeda = (*p_dis)(generator);
        //cout << "Probabilidade: " << prob << ", numero aleatorio: " << moeda << endl;

        if (moeda < prob)
            h(i) = 1;
        else
            h(i) = 0;
    }
}

VectorXd RBM::sample_hout() {
    // NOTE: Will not check if has been initialized and/or has seed,
    //   because this method should be called only by the RBM itself
    //   and other functions should have performed the necessary
    //   checks

    //cout << "Sampling h!" << endl;

    VectorXd out(hSize);
    auxH = (*p_W)*x;
    // NOTE: Não uso getProbabilities porque aproveito o loop

    double prob, moeda;
    for (int i=0; i<hSize; i++){
        prob = 1.0/( 1 + exp( - (*p_b)(i) -  auxH(i) ) );
        moeda = (*p_dis)(generator);
        //cout << "Probabilidade: " << prob << ", numero aleatorio: " << moeda << endl;

        if (moeda < prob)
            out(i) = 1;
        else
            out(i) = 0;
    }

    return out;
}

void RBM::sampleXtilde( SampleType sType, int k ) {
    if (!initialized){
        string errorMessage;
        errorMessage = "Cannot sample without RBM dimensions!";
        printError(errorMessage);
        throw runtime_error(errorMessage);
    }
    if (!hasSeed){
        string errorMessage;
        errorMessage = "Cannot sample from RBM without random seed!\n\t"
                       "Use 'setRandomSeed' before proceeding";
        printError(errorMessage);
        throw runtime_error(errorMessage);
    }

    switch (sType) {
        case SampleType::CD:
        // case SampleType::PCD:
            //cout << "Beginning Contrastive Divergence!" << endl;
            //cout << "x0 = " << x.transpose() << endl;

            for (int t=0; t<k; t++){
                sample_h();
                sample_x();

                //cout << "x^" << t+1 << " = " << x.transpose() << endl;
            }
            break;

        default:
            string errorMessage = "Sample type not implemented";
            printError(errorMessage);
            throw runtime_error(errorMessage);

        // TODO: Outros tipos de treinamento
    }
}

void RBM::sampleXtilde( SampleType sType, int k, VectorXd & x_vec ) {
    if (!initialized){
        string errorMessage;
        errorMessage = "Cannot sample without RBM dimensions!";
        printError(errorMessage);
        throw runtime_error(errorMessage);
    }
    if (!hasSeed){
        string errorMessage;
        errorMessage = "Cannot sample from RBM without random seed!\n\t"
                       "Use 'setRandomSeed' before proceeding";
        printError(errorMessage);
        throw runtime_error(errorMessage);
    }

    switch (sType) {
        case SampleType::CD:
        // case SampleType::PCD:
            //cout << "Beginning Contrastive Divergence!" << endl;
            //cout << "x0 = " << x.transpose() << endl;

            sample_h(x_vec);
            sample_x();

            for (int t=1; t<k; t++){
                sample_h();
                sample_x();
            }
            break;

        default:
            string errorMessage = "Sample type not implemented";
            printError(errorMessage);
            throw runtime_error(errorMessage);

        // TODO: Outros tipos de treinamento
    }
}

void RBM::sample_x_label(int l_size) {
    double prob, moeda, aux;

    for (int j = xSize - l_size; j < xSize; j++) {
        aux = h.transpose() * (*p_W).col(j);

        prob = 1.0/( 1 + exp( - d(j) -  aux ) );
        moeda = (*p_dis)(generator);

        if (moeda < prob)
            x(j) = 1;
        else
            x(j) = 0;
    }
}


// Training methods
void RBM::trainSetup() {
    trainSetup(false);
}
void RBM::trainSetup(bool NLL) {
    trainSetup(CD, 1, 1000, 5, 0.01, NLL, 1, false);
}
void RBM::trainSetup(SampleType sampleType, int k, int iterations,
                     int batchSize, double learnRate, bool NLL) {
    trainSetup(sampleType, k, iterations, batchSize, learnRate, NLL, 1, false);
}

void RBM::trainSetup(SampleType sampleType, int k, int iterations,
                     int batchSize, double learnRate, bool NLL, int period) {
    trainSetup(sampleType, k, iterations, batchSize, learnRate, NLL, period, false);
}

void RBM::trainSetup(SampleType sampleType, int k, int iterations,
                     int batchSize, double learnRate, bool NLL,
                     int period, bool doShuffle) {
    if (!initialized){
        string errorMessage;
        errorMessage = "Cannot train without RBM dimensions!";
        printError(errorMessage);
        throw runtime_error(errorMessage);
    }
    if (!hasSeed){
        string errorMessage;
        errorMessage = "Cannot train machine without random seed!\n\t"
                       "Use 'setRandomSeed' before proceeding";
        printError(errorMessage);
        throw runtime_error(errorMessage);
    }

    trainReady = true;

    stype = sampleType;     // training method
    k_steps = k;            // gibbs sampling steps
    n_iter = iterations;    // number of iterations over data
    b_size = batchSize;     // batch size
    l_rate = learnRate;     // learning rate
    calcNLL = NLL;          // flag to calculate NLL over iterations (or not)
    freqNLL = period;       // Rate of NLL calculation
    shuffle = doShuffle;    // flag to shuffle data order through iterations

    if ( calcNLL && (xSize > MAXSIZE_EXACTPROBABILITY) && (hSize > MAXSIZE_EXACTPROBABILITY) ) {
        printWarning("Training set to calculate NLL, but dimensions are too big. "
                     "An approximation will be made instead");
    }

    startWeights();
    // TODO: Adicionar outras inicializações como parâmetro opcional??
}

void RBM::fit(Data & trainData){
    if (!trainReady){
        string errorMessage;
        errorMessage = "Cannot train machine without setting up "
                       "training features!\n\tUse 'trainSetup' "
                       "before proceeding";
        printError(errorMessage);
        throw runtime_error(errorMessage);
    }
    if (shuffle) { trainData.setRandomSeed(generator()); }

    printInfo("------- TRAINING DATA -------");

    int n_batches = ceil(trainData.get_number_of_samples()/float(b_size));
    MatrixXd W_gradient(hSize, xSize);
    VectorXd b_gradient(hSize), d_gradient(xSize);
    VectorXd h_hat(hSize);
    int actualSize;
    double nll_val;

    if (calcNLL) {
        history.push_back(negativeLogLikelihood(trainData)); // Before training
    }

    int it, bIdx, sample;

    for (it = 0; it < n_iter; ++it) {
        // cout << "Iteration " << it+1 << " of " << n_iter << endl;

        if ( (it > 0) && shuffle ) {
            trainData.shuffle();
        }

        actualSize = b_size;

        for (bIdx = 0; bIdx < n_batches; ++bIdx) {
            // cout << "\tBatch " << bIdx+1 << endl;

            int initS = bIdx * b_size;

            for (sample = initS; sample < (bIdx+1) * b_size; ++sample) {
                if ( sample >= trainData.get_number_of_samples() ) break;
                // cout << "\t\ts = " << s << endl;

                VectorXd & xt = trainData.get_sample(sample);
                getProbabilities_h(h_hat, xt);

                if (sample == initS) {
                    W_gradient = h_hat * xt.transpose();
                    b_gradient = h_hat;
                    d_gradient = xt;
                } else {
                    W_gradient += h_hat * xt.transpose();
                    b_gradient += h_hat;
                    d_gradient += xt;
                }

                sampleXtilde(stype, k_steps, xt); // Changes x value
                getProbabilities_h(h_hat);

                W_gradient -= h_hat*x.transpose();
                b_gradient -= h_hat;
                d_gradient -= x;
            }

            if (bIdx == n_batches-1) {
                actualSize = trainData.get_number_of_samples() - bIdx * b_size;
            }

            W_gradient = W_gradient/actualSize;
            if (patterns) W_gradient = W_gradient.cwiseProduct(A);
            W = W + l_rate * W_gradient;
            if (patterns) {C = W.cwiseProduct(A);}
            // Ao atualizar W, temos que atualizar C também!

            b_gradient = b_gradient/actualSize;
            if (optim_H) b_gradient = b_gradient.cwiseProduct(s);
            b = b + l_rate*b_gradient;
            if (optim_H) v = b.cwiseProduct(s);

            d_gradient = d_gradient/actualSize;
            d = d + l_rate*d_gradient;

            // cout << "\tUpdated weights" << endl;

        }

        if (calcNLL) {
            if ( ((it+1) % freqNLL == 0) || (it == n_iter-1) ) {
                nll_val = negativeLogLikelihood(trainData);
                history.push_back(nll_val);
                cout << "Iteration " << it+1 << ": NLL = " << nll_val << endl;
            }
        }
    }

    isTrained = true;
    printInfo("Finished RBM training");
}


void RBM::fit_connectivity(Data & trainData) {
    if ( !patterns ){
        printError("Cannot optimize patterns if we have a classical RBM");
        cerr << "Cannot optimize patterns if we have a classical RBM, "
             << "please set connectivity to true before proceeding" << endl;
        exit(1);
    }
    if ( !trainReady ) {
        printError("Cannot optmize connectivity without setting up training parameters");
        cerr << "Please set up training parameters in 'trainSetup' before attempting to train connectivity" << endl;
        exit(1);
    }
    if ( !optReady ) {
        printError("Cannot optmize connectivity without setting up the optimization parameters");
        cerr << "Please set up optimization parameters in 'optSetp' before attempting to train connectivity" << endl;
        exit(1);
    }

    // FIXME: Tirar a flag opt_type (e trazer o treino pra ca)
    //    Não vou mais implementar alternativas de treinamento, à princípio. Esse já é o NCG
    switch (opt_type) {
        case Heuristic::SGD:
            optimizer_SGD(trainData);
            break;
        default:
            printError("Not yet implemented!");
            cerr << "Optimization via " << opt_type << " heuristic is not implemented" << endl;
    }
    printInfo("Finished RBM training");
}


void RBM::fit_H(Data & trainData) {
    // This function trains the RBM, optimizing the number of neurons H. After training, a
    //      smaller RBM with less hidden neurons may be achieved ('reduced_rbm' method)

    if ( !optim_H ){
        printError("Cannot optimize units activation with a classical RBM");
        cerr << "Cannot optimize patterns if we have a classical RBM, please "
             << "use 'rbm.hidden_activation(true)' before proceeding" << endl;
        exit(1);
    }
    if ( !trainReady ) {
        printError("Cannot optmize connectivity without setting up training parameters");
        cerr << "Please set up training parameters in 'trainSetup' before attempting to train connectivity" << endl;
        exit(1);
    }
    if ( !optReady ) {
        printError("Cannot optmize connectivity without setting up the optimization parameters");
        cerr << "Please set up optimization parameters in 'optSetp' before attempting to train connectivity" << endl;
        exit(1);
    }

    // TODO: Quero colocar outros parametros de otimizacao especificos deste treinamento?
    // TODO: Verificar flags!

    printInfo("------- TRAINING DATA: Optimizing H -------");

    int n_batches = ceil(trainData.get_number_of_samples()/float(b_size));
    MatrixXd W_gradient(hSize, xSize);
    VectorXd b_gradient(hSize), d_gradient(xSize);
    VectorXd s_gradient(hSize);
    MatrixXd matAux(hSize, xSize);
    VectorXd h_hat(hSize);
    int actualSize;
    double nll_val;

    int Xdata = xSize - nLabels;

    if (shuffle) { trainData.setRandomSeed(generator()); }

    VectorXd s_(hSize);          // Continuous version of h
    // s_ = VectorXd::Constant(hSize, 0.5);
    for (int i=0; i<hSize; i++) {
        s_(i) = (*p_dis)(generator)/2;
        if ( s(i) == 1 ) s_(i) += 0.5;
    }
    // FIXME: Preciso ver se quero inicializar assim mesmo...

    ofstream output;
    output.open(connect_out);

    if (calcNLL) {
        history.push_back(negativeLogLikelihood(trainData)); // Before training
    }
    if (saveConnectivity) {
        output << "# Hidden units activation throughout training" << endl;
        output << "# NCG-H optimization with threshold " << limiar_h
               << ". CD-" << k_steps << endl;
        output << "# Batch size = " << b_size << ", learning rate of " << l_rate
               << "(h' using " << l_rate_h << ")" << endl;
        output << "# s initialized with p = " << a_prob << endl;
        output << "0," << printHiddenActivation() << endl;
    }

    int it, bIdx, sample;

    for (it = 0; it < n_iter; ++it) {
        // cout << "Iteration " << it+1 << " of " << n_iter << endl;

        if ( (it > 0) && shuffle ) {
            trainData.shuffle();
        }

        actualSize = b_size;

        for (bIdx = 0; bIdx < n_batches; ++bIdx) {
            int initS = bIdx * b_size;

            for (sample = initS; sample < (bIdx+1) * b_size; ++sample) {
                if ( sample >= trainData.get_number_of_samples() ) break;

                VectorXd & xt = trainData.get_sample(sample);
                getProbabilities_h(h_hat, xt);

                matAux = h_hat * xt.transpose();    // FIXME: Is this faster than calculating twice?

                if (sample == initS) {
                    W_gradient = matAux;
                    b_gradient = h_hat;
                    d_gradient = xt;
                    s_gradient = h_hat.cwiseProduct(W * xt + b);
                } else {
                    W_gradient += matAux;
                    b_gradient += h_hat;
                    d_gradient += xt;
                    s_gradient += h_hat.cwiseProduct(W * xt + b);
                }

                sampleXtilde(stype, k_steps, xt);  // Changes x value
                getProbabilities_h(h_hat);
                matAux = h_hat * x.transpose();

                W_gradient -= matAux;
                b_gradient -= h_hat;
                d_gradient -= x;
                s_gradient -= h_hat.cwiseProduct(W * x + b);
            }

            if (bIdx == n_batches-1) {
                actualSize = trainData.get_number_of_samples() - bIdx * b_size;
            }

            W_gradient = W_gradient/actualSize;
            W_gradient = W_gradient.cwiseProduct(A);
            W = W + l_rate*W_gradient;

            s_gradient = s_gradient/actualSize;
            s_ = s_ + l_rate_h * s_gradient;
            // cout << "Batch " << bIdx << endl;
            // cout << "Helper (h') " << h_.transpose() << endl;
            // cout << "Activtion   " << s.transpose() << endl;
            // cout << "Gradient    " << s_gradient.transpose() << endl;
            for (int i=0; i < hSize; i++) {
                if ( s_(i) < limiar_h ) {
                    s(i) = 0;
                    if ( s_(i) < 0 ) s_(i) = 0;
                } else
                if ( h(i) > limiar_h ) {
                    s(i) = 1;
                    if ( s_(i) > 1 ) s_(i) = 1;
                }
            }

            inhibitA();
            C = W.cwiseProduct(A);

            b_gradient = b_gradient/actualSize;
            b = b + l_rate*b_gradient;
            v = b.cwiseProduct(s);

            d_gradient = d_gradient/actualSize;
            d = d + l_rate*d_gradient;
        }

        if (calcNLL) {
            if ( ((it+1) % freqNLL == 0) || (it == n_iter-1) ) {
                nll_val = negativeLogLikelihood(trainData);
                history.push_back(nll_val);
                cout << "Iteration " << it+1 << ": NLL = " << nll_val << endl;
            }
        }
        if (saveConnectivity) {
            output << it+1 << "," << printHiddenActivation() << endl;
        }
    }

    isTrained = true;
    printInfo("Finished RBM training");
}

/*
void RBM::fit_conn_size(Data & trainData) {
    // This function trains the RBM, optimizing both the connectivity A, and the number of
    //      neurons H. After training, a smaller RBM with less hidden neurons may be achieved,
    //      by applying the 'reduced_rbm' method.
}
*/


void RBM::optSetup(){
    optSetup(SGD, true, "connectivity", 1, 0);
}

void RBM::optSetup(Heuristic method, string connFileName, double p){
    optSetup(SGD, true, connFileName, p, 0);
}

void RBM::optSetup(Heuristic method, string connFileName, double p, int labels){
    optSetup(SGD, true, connFileName, p, 0);
}

void RBM::optSetup(Heuristic method, double p){
    optSetup(SGD, false, "", p, 0);
}

void RBM::optSetup(Heuristic method, double p, int labels){
    optSetup(SGD, false, "", p, 0);
}

void RBM::optSetup(Heuristic method, bool saveConn, string connFileName, double p, int labels){
    opt_type = method;
    saveConnectivity = saveConn;
    connect_out = connFileName;
    a_prob = p;
    nLabels = labels;
    startConnectivity(p);
    startActiveHiddenUnits(p);

    // NCG parameters
    limiar_A = 0.5;
    l_rate_A = 5*l_rate;

    limiar_h = 0.5;
    l_rate_h = l_rate;
    // TODO: Change this so it is not hardcoded
    // Talvez fazer uma função setThreshold, só fica bem chato ter que dar três setups diferentes

    optReady = true;
}

string RBM::printConnectivity_linear() {
    stringstream ret;
    const char* separator = "";
    for (int i=0; i<hSize; i++) {
        for (int j=0; j<xSize; j++) {
            ret << separator << A(i,j);
            separator = ",";
        }
    }
    return ret.str();
}

string RBM::printHiddenActivation() {
    stringstream ret;
    const char* separator = "";
    for (int i=0; i<hSize; i++) {
        ret << separator << s(i);
        separator = ",";
    }
    return ret.str();
}

void RBM::optimizer_SGD(Data & trainData) {
    printInfo("------- TRAINING DATA: Optimizing A -------");

    int n_batches = ceil(trainData.get_number_of_samples()/float(b_size));
    MatrixXd W_gradient(hSize, xSize);
    VectorXd b_gradient(hSize), d_gradient(xSize);
    MatrixXd A_gradient(hSize, xSize);
    MatrixXd matAux(hSize, xSize);
    VectorXd h_hat(hSize);
    int actualSize;
    double nll_val;

    int Xdata = xSize - nLabels;

    if (shuffle) { trainData.setRandomSeed(generator()); }

    MatrixXd A_(hSize, xSize);    // Continuous version of A
    for (int i=0; i<hSize; i++) {
        for (int j=0; j<Xdata; j++) {
            A_(i,j) = (*p_dis)(generator)/2;
            if (A(i,j) == 1) A_(i,j) += 0.5;
            // A_ is initialized uniformly between 0 and 0.5 or
            // between 0.5 and 1, according to A's values
        }
    }

    //stringstream mat_dump;
    //mat_dump << connect_out << "_SGD_lim" << limiar_A << "_CD-" << k;

    ofstream output;
    output.open(connect_out);

    if (calcNLL) {
        history.push_back(negativeLogLikelihood(trainData)); // Before training
    }
    if (saveConnectivity) {
        output << "# Connectivity patterns throughout training" << endl;
        output << "# NCG optimization (version 1) with threshold " << limiar_A << ". CD-" << k_steps << endl;
        output << "# Batch size = " << b_size << ", learning rate of " << l_rate << "(" << l_rate_A
               << " for the connectivity)" << endl;
        output << "# A initialized with p = " << a_prob << endl;
        output << "0," << printConnectivity_linear() << endl;
    }

    int it, bIdx, sample;

    for (it = 0; it < n_iter; ++it) {
        // cout << "Iteration " << it+1 << " of " << n_iter << endl;

        if ( (it > 0) && shuffle ) {
            trainData.shuffle();
        }

        actualSize = b_size;

        for (bIdx = 0; bIdx < n_batches; ++bIdx) {
            int initS = bIdx * b_size;

            for (sample = initS; sample < (bIdx+1) * b_size; ++sample) {
                if ( sample >= trainData.get_number_of_samples() ) break;

                VectorXd & xt = trainData.get_sample(sample);
                getProbabilities_h(h_hat, xt);

                matAux = h_hat * xt.transpose(); // FIXME: Is this faster than calculating twice?

                if (sample == initS) {
                    W_gradient = matAux;
                    b_gradient = h_hat;
                    d_gradient = xt;
                    A_gradient = W.cwiseProduct( matAux );
                } else {
                    W_gradient += matAux;
                    b_gradient += h_hat;
                    d_gradient += xt;
                    A_gradient += W.cwiseProduct( matAux );
                }

                sampleXtilde(stype, k_steps, xt);  // Changes x value
                getProbabilities_h(h_hat);
                matAux = h_hat * x.transpose();

                W_gradient -= matAux;
                b_gradient -= h_hat;
                d_gradient -= x;
                A_gradient -= W.cwiseProduct( matAux );
            }

            if (bIdx == n_batches-1) {
                actualSize = trainData.get_number_of_samples() - bIdx * b_size;
            }

            W_gradient = W_gradient/actualSize;
            W_gradient = W_gradient.cwiseProduct(A);
            W = W + l_rate*W_gradient;

            A_gradient = A_gradient/actualSize;
            A_ = A_ + (l_rate_A) * A_gradient;
            for (int i=0; i<hSize; i++) {
                for (int j=0; j<Xdata; j++) {
                    if ( A_(i,j) < limiar_A ) {
                        A(i,j) = 0;

                        if ( A_(i,j) < 0 ) A_(i,j) = 0;
                    }
                    else if ( A_(i,j) > limiar_A ) {
                        A(i,j) = 1;

                        if ( A_(i,j) > 1 ) A_(i,j) = 1;
                    }
                }
            }   // NOTE: Talvez criar uma função para binarizar a matriz A

            // cout << "Maximum and minimum coefficients of A's gradient: "
            //      << A_gradient.maxCoeff() << ", " << A_gradient.minCoeff() << endl;

            // // Print for debugging
            // cout << "Continuous connectivity: " << endl;
            // cout << A_ << endl;

            C = W.cwiseProduct(A);

            b_gradient = b_gradient/actualSize;
            // if (optim_H) b_gradient = b_gradient.cwiseProduct(s); // FIXME: necessário??
            b = b + l_rate*b_gradient;
            if (optim_H) v = b.cwiseProduct(s);

            d_gradient = d_gradient/actualSize;
            d = d + l_rate*d_gradient;
        }

        if (calcNLL) {
            if ( ((it+1) % freqNLL == 0) || (it == n_iter-1) ) {
                nll_val = negativeLogLikelihood(trainData);
                history.push_back(nll_val);
                cout << "Iteration " << it+1 << ": NLL = " << nll_val << endl;
            }
        }
        if (saveConnectivity) {
            output << it+1 << "," << printConnectivity_linear() << endl;
        }
    }

    isTrained = true;
}

double RBM::negativeLogLikelihood(Data & data) {
    //return negativeLogLikelihood(data, ZEstimation::Trunc);
    if (xSize > MAXSIZE_EXACTPROBABILITY) {
        if (hSize <= MAXSIZE_EXACTPROBABILITY) {
            return negativeLogLikelihood(data, ZEstimation::None_H);
        } else {
            if (isTrained) {    // Only raise the warning after training, so as not to pollute training log
                printWarning("Will provide an approximation of the NLL, since the RBM is too big for exact calculation");
            }
            return negativeLogLikelihood(data, ZEstimation::AIS);
            // NOTE: You must have either matlab or orcad to estimate the AIS (choose which in 'ais_estimator.sh')
            //      You can select another way of calculating the NLL if you want (though most are not very good)
            //      Check possibilities from ZEstimation
        }
    } else {
        return negativeLogLikelihood(data, ZEstimation::None);
    }
}

// Evaluation methods
double RBM::negativeLogLikelihood(Data & data, ZEstimation method) {
    if (!initialized){
        string errorMessage;
        errorMessage = "Cannot calculate NLL without initializing RBM!";
        printError(errorMessage);
        throw runtime_error(errorMessage);
    }

    //cout << "Calculating NLL" << endl;

    long double total = 0;
    int N = data.get_number_of_samples();

    for (int idx = 0; idx < N; ++idx) {
        total += freeEnergy( data.get_sample(idx) );
    }
    total = total/N;
    // cout << "Parcial da NLL: " << total << endl;

    switch (method) {
        case None:
            if (xSize > MAXSIZE_EXACTPROBABILITY) {
                printWarning("Attempting to calculate exact NLL for a RBM that is too big. Script may never finish.");
            }
            total += log( normalizationConstant_effX() );
            break;

        case None_H:
            if (hSize > MAXSIZE_EXACTPROBABILITY) {
                printWarning("Attempting to calculate exact NLL for a RBM that is too big. Script may never finish.");
            }
            // double Z;
            // Z = logl( normalizationConstant_effH() );
            // cout << "\nLog of the normalization constant: " << Z << endl;
            // total += Z;
            total += log( normalizationConstant_effH() );
            break;

        case MC:
            if (!hasSeed){
                string errorMessage;
                errorMessage = "Cannot estimate NLL without random seed!";
                printError(errorMessage);
                throw runtime_error(errorMessage);
            }
            int idx;
            idx = int( N * (*p_dis)(generator) );

            x = data.get_sample(idx);
            total += logl( normalizationConstant_MCestimation( 1000 ) );  // TODO: Fine-tune parameter
            break;

        case AIS:
            {
            // if (!hasSeed){
            //     string errorMessage;
            //     errorMessage = "Cannot estimate NLL without random seed!";
            //     printError(errorMessage);
            //     throw runtime_error(errorMessage);
            // }
            // total += logl( normalizationConstant_AISestimation( 100 ) );
            int pid = getpid();
            string pidStr = to_string(pid);

            string rbmName = "tmp_" + pidStr + ".rbm";
            save(rbmName);

            string command = "./ais_estimator.sh " + pidStr;
            system(command.c_str());

            fstream input;
            string ncfilename = "lnZ_" + pidStr + ".txt";
            input.open(ncfilename.c_str(), ios::in);
            if( !input ) {
                printError("Could not estimate normalization constant");
                cerr << "File '" << ncfilename << "' could not be opened" << endl;
                throw runtime_error("Something wrong with AIS (matlab code)");
            }
            string line;
            input >> line;
            input.close();

            total += atof(line.c_str());
            }
            break;

        case Trunc:
            total += log( normalizationConstant_trunc(data) );
            break;

        case TruncRep:
            total += log( normalizationConstant_truncRep(data) );
            break;

        default:
            printError("Normalization constant method not implemented!");
            cerr << "There is no available implementation of method '" << method
                 << "'. Please choose another one."<< endl;
            exit(1);
    }

    return total;
}

double RBM::negativeLogLikelihood(Data & data, double logZ) {
    if (!initialized){
        string errorMessage;
        errorMessage = "Cannot calculate NLL without initializing RBM!";
        printError(errorMessage);
        throw runtime_error(errorMessage);
    }

    long double total = 0;
    int N = data.get_number_of_samples();

    for (int idx = 0; idx < N; ++idx) {
        total += freeEnergy( data.get_sample(idx) );
    }
    total = (total/N) + logZ;

    return total;
}

vector<double> RBM::getTrainingHistory() {
    if (!isTrained) {
        string errorMessage;
        errorMessage = "RBM has not been trained, and therefore has no training history";
        printError(errorMessage);
    }
    if (!calcNLL) {
        string errorMessage;
        errorMessage = "RBM is not set to calculate NLL throughout training, no data available.";
        printWarning(errorMessage);
    }
    return history;
}

// Energy methods
double RBM::energy() {
    double ret = - x.transpose()*d;
    ret -= h.transpose()*(*p_b);
    ret -= (h.transpose()*(*p_W))*x;
    //cout << "Energy: " << ret << endl;

    return ret;
}

double RBM::freeEnergy() {
    double ret = - x.transpose()*d;
    auxH = (*p_W) * x;
    for (int i = 0; i < hSize; ++i) {
        ret -= log( 1 + exp( auxH(i) + (*p_b)(i) ) );
    }
    //cout << "Free energy: " << ret << endl;

    return ret;
}

double RBM::freeEnergy(VectorXd & x_vec) {
    double ret = - x_vec.transpose()*d;
    auxH = (*p_W) * x_vec;
    for (int i = 0; i < hSize; ++i) {
        ret -= log( 1 + exp( auxH(i) + (*p_b)(i) ) );
    }
    //cout << "Free energy: " << ret << endl;

    return ret;
}

double RBM::normalizationConstant() {
    // TODO: Warning for intractable calculation?
    // I am already putting in the NLL. Can this be used solo?
    return partialZ(xSize+hSize);
}
double RBM::partialZ(int n) {
    if (n == 0) {
        //cout << "current x: " << x.transpose() << "; and h: " << h.transpose() << endl;
        return exp( -energy() );
    }
    double ret = partialZ(n-1);
    if (n > hSize) {
        x(n-hSize-1) = abs(1 - x(n-hSize-1));
    } else {
        h(n - 1) = abs(1 - h(n - 1));
    }
    ret += partialZ(n-1);

    return ret;
}

double RBM::normalizationConstant_effX() {
    return partialZ_effX(xSize);
}
double RBM::partialZ_effX(int n) {
    if (n == 0) {
        //cout << "current x: " << x.transpose() << "(F = " << freeEnergy() << ")" << endl;
        double ret = exp(x.transpose()*d);
        auxH = (*p_W)* x + (*p_b);
        for (int i=0; i < hSize; i++) { ret *= 1 + exp( auxH(i) ); }

        return ret;
    }
    double ret = partialZ_effX(n-1);
    x(n-1) = abs(1 - x(n-1));
    ret += partialZ_effX(n-1);

    return ret;
}


long double RBM::normalizationConstant_effH() {
    return partialZ_effH(hSize);
}
long double RBM::partialZ_effH(int n) {
    if (n == 0) {
        long double ret = exp(h.transpose()*(*p_b));
        auxX = h.transpose() * (*p_W) + d.transpose();
        for (int j=0; j < xSize; j++) { ret *= 1 + exp( auxX(j) ); }
        return ret;
    }
    long double ret = partialZ_effH(n-1);
    h(n-1) = abs(1 - h(n-1));
    ret += partialZ_effH(n-1);

    return ret;
}


long double RBM::normalizationConstant_MCestimation(int n_samples) {
    // Estimates the normalization constant (partition function) of the RBM
    int steps = int(xSize * log(2)); // FIXME: It does not seem to be big enough

    // int check = 0;

    long double soma = 0;
    for (int s=0; s < n_samples; s++) {
        for (int r=0; r < steps; r++) {
            sample_h();
            sample_x();
        }
        soma += expl(freeEnergy());
        // cout << "Soma = " << soma << endl;
    }

    return pow(2, xSize) * n_samples / soma;
}


long double RBM::normalizationConstant_trunc(Data & data) {
    // NOTE: This method version assumes there are no duplicated samples in 'data'
    //  This makes the method faster, although could lead to over-estimation of the
    //  normalization constant if there are duplicated samples
    long double estimativa = 0;

    int N = data.get_number_of_samples();
    int size = data.get_sample_size();

    for (int s = 0; s < N; ++s) {

        estimativa += addSampleAndNeighbors( data.get_sample(s) );
    }

    return estimativa;
}


long double RBM::normalizationConstant_truncRep(Data & data) {
    long double estimativa = 0;

    int N = data.get_number_of_samples();

    for (int s = 0; s < N; ++s) {
        if ( data.isDuplicated(s) ) continue;  // Skips sample if it's been computed

        estimativa += addSampleAndNeighbors( data.get_sample(s) );
    }

    return estimativa;
}


long double RBM::addSampleAndNeighbors(VectorXd & vec) {
    x = vec;

    long double partialSum = expl( - freeEnergy() );

    // Troca do primeiro elemento
    x(0) = abs(1 - x(0));
    partialSum += expl( - freeEnergy() );

    // Troca dos outros elementos (primeiro destroca o anterior)
    for (int e = 1; e < xSize; e++) {
        x(e-1) = abs(1 - x(e-1));
        x(e) = abs(1 - x(e));
        partialSum += expl( - freeEnergy() );
    }

    return partialSum;
}


VectorXd RBM::complete_pattern(VectorXd & sample, int repeat) {
    if ( !initialized ) {
        printError("Cannot predict label without a trained RBM!");
        exit(1);
    }
    if ( !hasSeed ) {
        printError("Cannot predict label without random seed!");
        cerr << "RBM needs random seed to predict labels. Use 'setRandomSeed' before proceeding" << endl;
        exit(1);
    }

    int n_units = xSize - sample.size();

    x << sample, VectorXd::Constant(n_units, 0.5);

    for (int r=0; r < repeat; r++) {
        sample_h();
        sample_x_label(n_units);
    }

    return x;
}


int RBM::predict(VectorXd & sample, int n_labels) {
    if ( !initialized ) {
        printError("Cannot predict label without a trained RBM!");
        exit(1);
    }
    // FIXME: Preciso dar n_labels? É dedutível... Mas se eu for usar Data como entrada, vou ter esse valor

    int sampleSize = xSize - n_labels;
    if (sampleSize != sample.size()) {
        printError("Size of the data given does not match RBM size! (or the specified number of labels is wrong)");
        exit(1);
    }

    x << sample, VectorXd::Constant(n_labels, 0.5);

    int maxL;
    double maxProb = 0;

    auxH = (*p_W)*x + b;
    for (int i=0; i<hSize; i++){
        h(i) = 1.0/( 1 + exp( - auxH(i) ) );
    }
    auxX = h.transpose() * (*p_W) + d.transpose();
    for (int j = sampleSize; j < xSize; j++){
        if ( 1.0/( 1 + exp( - auxX(j) ) ) > maxProb ) {
            maxProb = 1.0/( 1 + exp( - auxX(j) ) );
            maxL = j - sampleSize;
        }
    }

    return maxL;
}

double RBM::classificationStatistics(Data & data, bool printExtras) {
    if ( !initialized ) {
        printError("Cannot predict label without a trained RBM!");
        exit(1);
    }

    data.joinLabels(false);
    int total = data.get_number_of_samples();
    int labels = data.get_number_of_labels();
    int sampleSize = data.get_sample_size();

    if (sampleSize + labels != xSize) {
        printError("Size of the data given does not match RBM size!");
        exit(1);
    }

    int results[labels][2];
    memset( results, 0, sizeof(results) );

    int maxL;
    double maxProb;

    int correct;

    for (int s = 0; s < total; s++) {
        // Classify the sample
        x << data.get_sample(s), VectorXd::Constant(labels, 0.5);

        maxProb = 0;

        auxH = (*p_W)*x + (*p_b);
        for (int i=0; i<hSize; i++){
            h(i) = 1.0/( 1 + exp( - auxH(i) ) );
        }
        auxX = h.transpose() * (*p_W) + d.transpose();
        for (int j = sampleSize; j < xSize; j++){
            if ( 1.0/( 1 + exp( - auxX(j) ) ) > maxProb ) {
                maxProb = 1.0/( 1 + exp( - auxX(j) ) );
                maxL = j - sampleSize;
            }
        }

        correct = data.get_label(s);
        results[ correct ][ int(correct == maxL) ]++;
        // cout << "Label = " << correct << ", prediction = " << maxL
        //      << ". Correct? " << int(correct == maxL) << endl;
    }

    int rights = 0;
    double lTotal;

    for (int l=0; l < labels; l++) {
        lTotal = results[l][1] + results[l][0];
        if (printExtras) {
            cout << "Label " << l << ": " << results[l][1] * 100 / lTotal
                 << "% correct and (total of " << lTotal << " samples)" << endl;
        }
        rights += results[l][1];
    }
    if (printExtras) cout << "\t Total Accuracy: " << float(rights)*100/total << " %" << endl;
    data.joinLabels(true);

    return double(rights)*100/total;
}


// Saving methods
void RBM::save(string filename) {
    if (!initialized){
        string errorMessage;
        errorMessage = "Cannot save an RBM without dimensions!";
        printError(errorMessage);
        throw runtime_error(errorMessage);
    }


    ofstream output;
    output.open(filename);

    output << "# RBM parameters" << endl;
    output << "# Contains sizes (X and H), weights and biases (visible, then hidden). "
           << "Does not save connectivity separated" << endl;

    if (!isTrained) {
        printWarning("Saving RBM that has not been trained, you may want to reconsider this.");
    } else {
        output << "# CD-" << k_steps << ", learning rate " << l_rate << ". " << n_iter
               << " iterations (epochs), batches of " << b_size << endl;
    }

    output << xSize << " " << hSize << endl;
    output << (*p_W) << endl;
    output << d.transpose() << endl;
    output << (*p_b).transpose() << endl;

    printInfo("Saved RBM into '" + filename + "'");
}

void RBM::load(string filename) {
    // NOTE: Note that loading an RBM reinitializes it, so previous configurations may be lost
    initialized = true;
    patterns = false;    // In the way it is loading, that is the case
    optim_H = false;
    calcNLL = false;

    fstream input;
    input.open(filename.c_str(), ios::in);
    if( !input ) {
        printError("Could not load RBM!");
        cerr << "File '" << filename << "' could not be opened" << endl;
        throw runtime_error("Load failed");
    }

    string line;

    int idx = 0;
    while (getline(input, line)) {
        if (line.substr(0,1) != "#") break;
    }

    stringstream ss(line);

    int X, H;
    ss >> X;
    ss >> H;

    initializer(X, H);

    for (int i = 0; i < hSize; i++) {
        getline(input, line);
        ss.clear();
        ss.str(line);

        for (int j = 0; j < xSize; j++) ss >> W(i,j);
    }

    getline(input, line);
    ss.clear();
    ss.str(line);

    for (int j = 0; j < xSize; j++) ss >> d(j);

    getline(input, line);
    ss.clear();
    ss.str(line);

    for (int i = 0; i < hSize; i++) ss >> b(i);

    printInfo("Loaded RBM from '" + filename + "'");
}

RBM RBM::reduced_rbm() {
    if ( !initialized ){
        string errorMessage;
        errorMessage = "Cannot output RBM with no dimensions!";
        printError(errorMessage);
        cerr << errorMessage << " You need to initialize and train the RBM" << endl;
        exit(1);
    }
    if ( !isTrained ){
        printError("In this method, one cannot obtain a pruned RBM without training first!");
        cerr << "You need to train the RBM (with H optimization enabled) before using this method" << endl;
        exit(1);
    }
    if ( !optim_H ){
        printError("Cannot obtain a pruned RBM if pruning is not active!");
        return *this;
    }

    int newH = s.sum();

    RBM ret(xSize, newH);

    ret.setVisibleBiases(d);

    VectorXd newB(newH);
    MatrixXd newW(newH, xSize);

    int idx = 0;
    for (int i=0; i < hSize; i++) {
        if ( idx >= newH ) break;
        if ( s(i) == 1 ) {
            newB(idx) = v(i);
            for (int j=0; j < xSize; j++) {
                newW(idx,j) = C(i,j);
            }
            idx++;
        }
    }

    ret.setHiddenBiases(newB);
    ret.setWeights(newW);

    return ret;
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
    cout << "\t\tHidden Units activation: " << s.transpose() << endl;
    cout << "\t\tVector v: " << v.transpose() << endl;
    cout << "\t\tConnectivity (" << patterns << "): " << endl << A << endl;
    cout << "\t\tMatrix C: " << endl << C << endl;
    cout << "----------------------------------------------------------------" << endl;
    cout << endl;
}


void RBM::sampleXH() {
    if ( !initialized ) {
        printError("Cannot sample from non initialized RBM");
        exit(1);
    }
    if ( !hasSeed ) {
        printError("Cannot sample without random seed");
        cerr << "Please set RBM random seed before trying to sample from it" << endl;
        exit(1);
    }
    sample_x();
    cout << "x sampled: " << x.transpose() << endl;
    sample_h();
    cout << "h sampled: " << h.transpose() << endl;
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
    startWeights();
    startBiases();
    for (int k = 0; k < 5; ++k) {
        // Mix a bit x and h, to have non trivial probabilities
        sample_h();
        sample_x();
    }
    printVariables();

    // Auxiliary variables
    VectorXd probX = getProbabilities_x();
    VectorXd probH = getProbabilities_h();

    VectorXd freqX = VectorXd::Zero(xSize);
    VectorXd freqH = VectorXd::Zero(hSize);

    for (int k = 0; k < rep; ++k) {
        freqX += sample_xout();
        freqH += sample_hout();
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