#include "reference.hpp"

std::vector<float> Reference::numeric_labels;

void Reference::initialize_labels(std::istream & labels){
    //read labels
    Encoder encoder(labels);
    Reference::numeric_labels = encoder.read_numerical_targets();
};

void Reference::normalize_labels(float loss_normalizer) {
    std::vector<float> &labels = Reference::numeric_labels;
    
    for (int i = 0; i < labels.size(); i++) {
        labels[i] = labels[i] / loss_normalizer;
    }

};
