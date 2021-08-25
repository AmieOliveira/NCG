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
    initializer(X, H);
}
RBM::RBM(int X, int H, bool use_pattern) {
    initialized = true;
    patterns = use_pattern;
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

    if (patterns) {
        C = W.cwiseProduct(A);
        p_W = &C;
    }
    else p_W = &W;
}

// Connectivity (de)ativation
void RBM::connectivity(bool activate) {
    if (activate == patterns) {
        printWarning("Tried to set connectivity activation to a status it already had");
    }
    patterns = activate;
    if (patterns){
        C = W.cwiseProduct(A);
        p_W = &C;
    } else {
        p_W = &W;
    }
}


// Set dimensions
void RBM::setDimensions(int X, int H) {
    if (!initialized) {
        initialized = true;
        patterns = false;
    }
    initializer(X, H);
}
void RBM::setDimensions(int X, int H, bool use_pattern) {
    if (!initialized) initialized = true;
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
    return b;
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
        for (int j=0; j<xSize; j++) {
            moeda = (*p_dis)(generator);

            if (moeda < p) A(i,j) = 1;
            else A(i,j) = 0;
        }
    }

    // cout << "Connecttivity matrix, initialated with p = " << p << endl;
    // cout << A << endl;
    // cout << "Percentage of connections activated: " << A.sum()/(xSize*hSize) << endl;
}

MatrixXd RBM::getConnectivityWeights() {
    return *p_W;
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
VectorXd RBM::sample_x() {
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

    VectorXd output(xSize);
    RowVectorXd vAux = h.transpose()*(*p_W);
    // NOTE: Não uso getProbabilities porque aproveito o loop

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
VectorXd RBM::sample_h() {
    // NOTE: Will not check if has been initialized and/or has seed,
    //   because this method should be called only by the RBM itself
    //   and other functions should have performed the necessary
    //   checks

    //cout << "Sampling h!" << endl;

    VectorXd output(hSize);
    VectorXd vAux = (*p_W)*x;
    // NOTE: Não uso getProbabilities porque aproveito o loop

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

    vector<VectorXd> ret;

    switch (sType) {
        case SampleType::CD:
        // case SampleType::PCD:
            //cout << "Beginning Contrastive Divergence!" << endl;
            for (vector<VectorXd>::iterator x_0 = vecs.begin();
                 x_0 != vecs.end();
                 ++x_0) {
                x = *x_0;
                //cout << "x0 = " << x.transpose() << endl;

                for (int t=0; t<k; t++){
                    h = sample_h();
                    x = sample_x();

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
    }
    return ret;
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

    if ( calcNLL && (xSize > MAXSIZE_EXACTPROBABILITY) ) {
        printWarning("Training set to calculate NLL, but dimensions are too big. "
                     "An approximation will be made instead");
    }

    startWeights();
    // TODO: Adicionar outras inicializações como parâmetro opcional??
}

void RBM::fit(Data trainData){
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
    // TODO: Print training setup?

    int n_batches = ceil(trainData.get_number_of_samples()/float(b_size));
    vector<VectorXd> batch, sampled;
    MatrixXd W_gradient(hSize, xSize);
    VectorXd b_gradient(hSize), d_gradient(xSize);
    VectorXd h_hat(hSize);
    int actualSize;
    double nll_val;

    if (calcNLL) {
        history.push_back(negativeLogLikelihood(trainData)); // Before training
    }

    for (int it = 0; it < n_iter; ++it) {
        // cout << "Iteration " << it+1 << " of " << n_iter << endl;

        if ( (it > 0) && shuffle ) {
            trainData.shuffle();
        }

        actualSize = b_size;

        for (int bIdx = 0; bIdx < n_batches; ++bIdx) {
            //cout << "Batch " << bIdx+1 << endl;

            batch = trainData.get_batch(bIdx, b_size);
            sampled = sampleXtilde(stype, k_steps, batch);

            W_gradient = MatrixXd::Zero(hSize, xSize);
            b_gradient = VectorXd::Zero(hSize);
            d_gradient = VectorXd::Zero(xSize);

            for (int s = 0; s < batch.size(); ++s) {
                //cout << "x0 = " << batch.at(s).transpose() << endl;
                //cout << "xk = " << sampled.at(s).transpose() << endl;

                x = batch.at(s);
                h_hat = getProbabilities_h(); // Note that x must be set before!
                //cout << "h^ de xt: " << h_hat.transpose() << endl;
                W_gradient += h_hat*x.transpose();
                b_gradient += h_hat;
                d_gradient += x;

                x = sampled.at(s);
                h_hat = getProbabilities_h(); // Note that x must be set before!
                //cout << "h^ de x~: " << h_hat.transpose() << endl;
                W_gradient -= h_hat*x.transpose();
                b_gradient -= h_hat;
                d_gradient -= x;
            }

            if (bIdx == n_batches-1) actualSize = batch.size();

            W_gradient = W_gradient/actualSize;
            if (patterns) W_gradient = W_gradient.cwiseProduct(A);
            W = W + l_rate*W_gradient;
            if (patterns) {C = W.cwiseProduct(A);}
            // Ao atualizar W, temos que atualizar C também!

            b_gradient = b_gradient/actualSize;
            b = b + l_rate*b_gradient;

            d_gradient = d_gradient/actualSize;
            d = d + l_rate*d_gradient;

            // cout << "New W:" << endl << W << endl;
            // cout << "New b:" << b.transpose() << endl;
            // cout << "New d:" << d.transpose() << endl;

            //for (auto x0: batch) {
            //    cout << "x0 = " << x0.transpose() << endl;
            //    xk = sampleXtilde(stype, k_steps, x0);
            //    cout << "xk = " << xk.transpose() << endl;
            //}
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

    /*
    vecOut = sampleXtilde(stype, 4, vecIn);
    cout << "Output: ";
    for (auto i: vecOut)
        cout << "[" << i.transpose() << "] ";
    cout << endl;
     */
}


void RBM::fit_connectivity(Data trainData) {
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

    switch (opt_type) {
        case Heuristic::SGD:
            optimizer_SGD(trainData);
            break;
        default:
            printError("Not yet implemented!");
            cerr << "Optimization via " << opt_type << " heuristic is not implemented" << endl;
    }
}


void RBM::optSetup(){
    optSetup(SGD, "connectivity", 1);
}

void RBM::optSetup(Heuristic method, string connFileName, double p){
    opt_type = method;
    connect_out = connFileName;
    a_prob = p;
    startConnectivity(p);

    // SGD parameters
    limiar = 0.5;
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

void RBM::optimizer_SGD(Data trainData) {
    printInfo("------- TRAINING DATA: Optimizing A -------");

    int n_batches = ceil(trainData.get_number_of_samples()/float(b_size));
    vector<VectorXd> batch, sampled;
    MatrixXd W_gradient(hSize, xSize);
    VectorXd b_gradient(hSize), d_gradient(xSize);
    MatrixXd A_gradient(hSize, xSize);
    VectorXd h_hat(hSize);
    int actualSize;
    double nll_val;

    MatrixXd A_(hSize, xSize);    // Continuous version of A
    for (int i=0; i<hSize; i++) {
        for (int j=0; j<xSize; j++) {
            A_(i,j) = (*p_dis)(generator)/2;
            if (A(i,j) == 1) A_(i,j) += 0.5;
            // A_ is initialized uniformly between 0 and 0.5 or
            // between 0.5 and 1, according to A's values
        }
    }

    //stringstream mat_dump;
    //mat_dump << connect_out << "_SGD_lim" << limiar << "_CD-" << k;
    // TODO: Rever esse nome (o que eu quero e não quero por? E vai dar certo criar o nome aqui dentro?)

    ofstream output;
    output.open(connect_out);

    if (calcNLL) {
        history.push_back(negativeLogLikelihood(trainData)); // Before training
    }
    output << "# Connectivity patterns throughout training" << endl;
    output << "# SGD optimization (version 1) with threshold " << limiar << ". CD-" << k_steps << endl;
    output << "# Batch size = " << b_size << ", learning rate of " << l_rate << endl;
    output << "# A initialized with p = " << a_prob << endl;
    output << "0," << printConnectivity_linear() << endl;

    for (int it = 0; it < n_iter; ++it) {
        // TODO: pra mim faz sentido dar shuffle, mas isso é mesmo uma boa?
        //if (it != 0) trainData.shuffleOrder();

        cout << "Iteration " << it+1 << " of " << n_iter << endl;

        actualSize = b_size;

        for (int bIdx = 0; bIdx < n_batches; ++bIdx) {

            batch = trainData.get_batch(bIdx, b_size);
            sampled = sampleXtilde(stype, k_steps, batch);

            W_gradient = MatrixXd::Zero(hSize, xSize);
            b_gradient = VectorXd::Zero(hSize);
            d_gradient = VectorXd::Zero(xSize);

            A_gradient = MatrixXd::Zero(hSize, xSize);

            for (int s = 0; s < batch.size(); ++s) {

                x = batch.at(s);
                h_hat = getProbabilities_h();
                W_gradient += h_hat*x.transpose();
                b_gradient += h_hat;
                d_gradient += x;
                A_gradient += W.cwiseProduct( h_hat*x.transpose() );

                x = sampled.at(s);
                h_hat = getProbabilities_h();
                W_gradient -= h_hat*x.transpose();
                b_gradient -= h_hat;
                d_gradient -= x;
                A_gradient -= W.cwiseProduct( h_hat*x.transpose() );
            }

            if (bIdx == n_batches-1) actualSize = batch.size();

            W_gradient = W_gradient/actualSize;
            W_gradient = W_gradient.cwiseProduct(A);
            W = W + l_rate*W_gradient;

            A_gradient = A_gradient/actualSize;
            A_ = A_ + l_rate * A_gradient;
            for (int i=0; i<hSize; i++) {
                for (int j=0; j<xSize; j++) {
                    if ( A_(i,j) < limiar ) {
                        A(i,j) = 0;

                        if ( A_(i,j) < 0 ) A_(i,j) = 0;
                    }
                    else if ( A_(i,j) > limiar ) {
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
            b = b + l_rate*b_gradient;

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
        output << it+1 << "," << printConnectivity_linear() << endl;
    }

    isTrained = true;
}


// Evaluation methods
double RBM::negativeLogLikelihood(Data data) {
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
        x = data.get_sample(idx);
        total = total + freeEnergy();
    }
    total = total/N;

    //cout << "Partial: " << total << endl;
    if (xSize > MAXSIZE_EXACTPROBABILITY) {
        if (isTrained) {    // Only raise the warning after training, so as not to pollute training log
            printWarning("NLL provided is an approximation, since the RBM is too big for exact calculation");
        }
        if (!hasSeed){
            string errorMessage;
            errorMessage = "Cannot estimate NLL without random seed!";
            printError(errorMessage);
            throw runtime_error(errorMessage);
        }
        total += log( normalizationConstant_MCestimation( 1000 ) );
        // TODO: Change number of samples (make adaptable?)
    } else {
        //total += log( normalizationConstant() );
        total += log( normalizationConstant_effX() );
    }
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
    ret -= h.transpose()*b;
    ret -= (h.transpose()*(*p_W))*x;
    //cout << "Energy: " << ret << endl;

    return ret;
}

double RBM::freeEnergy() {
    double ret = - x.transpose()*d;
    VectorXd vAux = (*p_W)*x;
    for (int i = 0; i < hSize; ++i) {
        ret -= log( 1 + exp( vAux(i) + b(i) ) );
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
    // TODO: Warning for intractable calculation?

    double ret = partialZ_effX(xSize);
    return ret;
}
double RBM::partialZ_effX(int n) {
    if (n == 0) {
        //cout << "current x: " << x.transpose() << "(F = " << freeEnergy() << ")" << endl;
        return exp( -freeEnergy() );
    }
    double ret = partialZ_effX(n-1);
    x(n-1) = abs(1 - x(n-1));
    ret += partialZ_effX(n-1);

    return ret;
}


long double RBM::normalizationConstant_MCestimation(int n_samples) {
    // Estimates the normalization constant (partition function) of the RBM
    // NOTE: I am hardcoding one step between samples. (teorema ergódigo)

    long double soma = 0;
    for (int s=0; s < n_samples; s++) {
        h = sample_h();
        x = sample_x();

        soma += exp(freeEnergy());

        // cout << "Partial result: " << soma << endl;
    }
    // cout << "Normalization constant: " << pow(2, xSize) * n_samples / soma << endl;

    return pow(2, xSize) * n_samples / soma;
}


/*********************
double RBM::normalizationConstant_AISestimation() {
    // Reference: Salakhutdinov & Murray (2008), On the quantitative analysis of deep belief networks

    // FIXME: Estimation has unidentified bug

    // TODO: In theory I'd give some other RBM, with tuned d biases
    //        So far, the prior RBM has all weights and biases null
    // Reference "An Infinite Restricted Boltzmann Machine" uses same visible biases as model
    RBM prior(xSize, hSize);
    prior.setRandomSeed(generator());
    // NOTE: Code implemented assuming RBM has same shape as current one (but does not assume zero biases)
    // prior.setVisibleBiases(d);

    // FIXME: These are preliminary parameters, should change them
    int n_runs = 10;
    int n_betas = 10000;
    int transitionRepeat = 1;
    double betas[n_betas];

    for (int k=0; k < n_betas; k++) {
        betas[k] = double(k)/(n_betas - 1);
        // cout << "Beta_" << k << ": " << betas[k] << endl;
    }

    // Importance weights
    vector<double> weights;

    VectorXd x_k;

    RowVectorXd va_B;  // va_A;
    VectorXd hA(hSize);
    VectorXd dA, bA;
    double prob, moeda, part_w, w_r;

    bA = prior.getHiddenBiases();
    dA = prior.getVisibleBiases();
    // WA = prior.getConnectivityWeights();

    for (int r=0; r < n_runs; r++) {
        // Do an AIS run
        x_k = prior.sample_x();  // NOTE: Since weights are null, x does not depend on h, and I only need this step

        x = x_k;
        prior.setVisibleUnits( x_k );

        // First importance weight ratio
        part_w = 1;
        va_B = ((*p_W) * x_k) + b;
        // va_A = 0;  // va_A = WA * x_k;

        for (int i=0; i<hSize; i++) { // Somando as 4 parcelas softmax
            part_w *= ( 1 + exp( (1 - betas[1])*bA(i) ) );
            part_w *= ( 1 + exp( betas[1]*va_B(i) ) );
            // part_w *= ( 1 + exp( (1 - betas[1])*(va_A(i) + bA(i)) ) );
            part_w /= ( 1 + exp( betas[0]*va_B(i) ) );
            part_w /= ( 1 + exp( (1 - betas[0])*bA(i) ) );
            // part_w /= ( 1 + exp( (1 - betas[0])*(va_A(i) + bA(i)) ) );
        }

        w_r = exp( ( (betas[1] - betas[0]) * (x_k.transpose() * d )(0) ) +
                   ( (betas[0] - betas[1]) * (x_k.transpose() * dA)(0) )  ) * part_w;

        for (int k = 1; k < n_betas; k++) {
            // Sample x_k from x_{k-1}
            // FIXME: If I keep the RBM with some or all biases null, I can optimize calculations here

            int tr = 0;
            do {
                // Part 1: Sample h
                va_B = (*p_W) * x;
                // va_A = 0;  // WA * prior.getVisibleUnits();

                for (int i=0; i<hSize; i++){
                    prob = 1.0/( 1 + exp( - betas[k] * ( b(i) + va_B(i) ) ) );
                    moeda = (*p_dis)(generator);

                    if (moeda < prob)
                        h(i) = 1;
                    else
                        h(i) = 0;

                    prob = 1.0/( 1 + exp( - (1 - betas[k]) * bA(i) ) );
                    moeda = (*p_dis)(generator);

                    if (moeda < prob)
                        hA(i) = 1;
                    else
                        hA(i) = 0;
                }
                prior.setHiddenUnits(hA);  // Acho que não é necessário deixar isso
                // FIXME: Preciso de hA para alguma coisa? Parece que posso retirar dos calculos!

                // Part 2: Sample x
                va_B = h.transpose() * (*p_W);
                // va_A = 0;  // hA.transpose() * WA;

                for (int j=0; j<xSize; j++){
                    prob = 1.0/( 1 + exp( -( betas[k]*( d(j) + va_B(j) ) + (1-betas[k])*( dA(j) ) ) ) );
                    moeda = (*p_dis)(generator);

                    if (moeda < prob)
                        x_k(j) = 1;
                    else
                        x_k(j) = 0;
                }

                x = x_k;
                prior.setVisibleUnits( x_k );

                tr++;
            } while (tr < transitionRepeat);

            // Calculate current ratio for the importance weight
            part_w = 1;
            va_B = ((*p_W) * x_k) + b;
            // va_A = 0;  // WA * x_k;

            for (int i=0; i<hSize; i++) { // Somando as 4 parcelas softmax
                part_w *= ( 1 + exp( betas[k+1] * va_B(i) ) );
                part_w *= ( 1 + exp( (1 - betas[k+1]) * bA(i) ) );
                part_w /= ( 1 + exp( betas[k] * va_B(i) ) );
                part_w /= ( 1 + exp( (1 - betas[k]) * bA(i) ) );
            }
            w_r *= exp( ((betas[k+1] - betas[k]) * (x_k.transpose() * d ))(0) +
                        ((betas[k] - betas[k+1]) * (x_k.transpose() * dA))(0)  ) * part_w;
        }

        // cout << "Importance weight for run " << r << ": " << w_r << endl;

        weights.push_back(w_r);
    }

    double ZA = 1;
    for (int i=0; i<hSize; i++) {
        ZA *= 1 + exp(bA(i));
    }
    for (int j=0; j<xSize; j++) {
        ZA *= 1 + exp(dA(j));
    }
    // cout << "Normalization constant of auxiliar RBM: " << ZA << endl;

    double ratio_estimation = 0;
    for (auto wr: weights) {
        ratio_estimation += wr;
    }
    ratio_estimation /= n_runs;

    double variance = 0;
    for (auto wr: weights) {
        variance += (ratio_estimation - wr)*(ratio_estimation - wr);
    }
    variance /= n_runs;     // Empirical variance of wr
    variance /= n_runs;     // Approximated variance of the ratio estimation

    double Z = ratio_estimation * ZA;
    double sdZ = sqrt(variance) * ZA;

    cout << "Ratio: " << ratio_estimation << " +- " << variance << endl;
    cout << "Z^: " << Z << " +- " << sdZ << endl;
    cout << "ln( Z^ +- s ) = [" << log(Z - sdZ) << ", " << log(Z + sdZ) << "]" << endl;

    return Z;
}
*********************/

// Saving methods
void RBM::save(string filename) {
    if (!initialized){
        string errorMessage;
        errorMessage = "Cannot train without RBM dimensions!";
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
    output << b.transpose() << endl;

    printInfo("Saved RBM into '" + filename + "'");
}

void RBM::load(string filename) {
    // NOTE: Note that loading an RBM reinitializes it, so previous configurations may be lost
    initialized = true;
    patterns = false;    // In the way it is loading, that is the case
    calcNLL = false;

    fstream input;
    input.open(filename.c_str(), ios::in);

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


void RBM::sampleXH() {
    VectorXd vec = sample_x();
    cout << "x sampled: " << vec.transpose() << endl;
    vec = sample_h();
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
    startWeights();
    startBiases();
    for (int k = 0; k < 5; ++k) {
        // Mix a bit x and h, to have non trivial probabilities
        h << sample_h();
        x << sample_x();
    }
    printVariables();

    // Auxiliary variables
    VectorXd probX = getProbabilities_x();
    VectorXd probH = getProbabilities_h();

    VectorXd freqX = VectorXd::Zero(xSize);
    VectorXd freqH = VectorXd::Zero(hSize);

    for (int k = 0; k < rep; ++k) {
        freqX = freqX + sample_x();
        freqH = freqH + sample_h();
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