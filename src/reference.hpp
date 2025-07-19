#ifndef REFERENCE_H
#define REFERENCE_H

#include <vector>

#include "bitmask.hpp"
#include "encoder.hpp"

class Reference {
public: 
    static void initialize_labels(std::istream & labels);
    static void normalize_labels(float loss_normalizer);
    //labels for each row of the dataset, according to the reference model
    static std::vector<float> numeric_labels;   // (N,)
};

#endif
