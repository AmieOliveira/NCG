//
// Created by Amanda Oliveira on 20/05/21.
//

#ifndef TESE_CÓDIGO_DATA_H
#define TESE_CÓDIGO_DATA_H

#include "Eigen/Dense"

// IDEIA: Posso criar um DataGenerator, para gerar dados de BAS,
//        por exemplo, mas de outras distribuições tb. Ai serviria
//        como input para Data (ou Data criaria um DataGenerator
//        se eu criasse com um construtor específico...)

class Data {
    int _size;          // Size of a data sample
    int _n;             // Number of samples
    MatrixXd _data;     // Data

public:
    // Constructors
    Data(string data_path);
    Data(MatrixXd mat);

    // Data statistics
    int get_number_of_samples();
    VectorXd get_sample(int idx);
    // Função de obter conjunto de amostras (slice)
    // Separação em pacotes direto? treino e teste...
    // Frequência relativa de cada marginal (e de todas)
};


#endif //TESE_CÓDIGO_DATA_H
