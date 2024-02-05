#ifndef DATASET_H
#define DATASET_H

#include <iostream>
#include <sstream>
#include <math.h>
#include <map>
#include <vector>
#include <tuple>
#include <assert.h>
#include <vector>
#include <numeric>
#include <unordered_map>
#include <tbb/concurrent_hash_map.h>
#include <tbb/scalable_allocator.h>
// #include <ckmeans/dynamic_prog.cpp>
#include <ckmeans/Ckmeans.1d.dp.h>

#include <json/json.hpp>
#include <csv/csv.h>

class Dataset;

#include "bitmask.hpp"
#include "configuration.hpp"
#include "encoder.hpp"
#include "index.hpp"
//#include "state.hpp" // FIREWOLF: Circular References: Moved to cpp.
#include "tile.hpp"

using json = nlohmann::json;

// Contain the dataset and any preprocessed values
class Dataset {
public:
    // The encoder used in converting between non-binary and binary
    Encoder encoder;

    Dataset(void);
    // @param data_source: byte stream of csv format which will be automatically encoded into a binary dataset
    // @note see encoder documentation for data source formatting preconditions and encoding semantics
    Dataset(std::istream & data_source);
    ~Dataset(void);

    // @modifies loads data from data stream
    void load(std::istream & data_source);
    
    // @modifies resets dataset to initial state
    void clear(void);

    // @returns the sample size of the data set
    unsigned int size(void) const;
    // @returns the physical number of rows needed to represent the data set
    unsigned int height(void) const; 
    // @returns the number of binary non-target features used to represent the data set
    unsigned int width(void) const;
    // @returns the number of unique target values in the dataset
    unsigned int depth(void) const; 

    // @param capture_set: The indicator for each equivalent groups are contained by this problem
    // @param id: Index of the local state entry used when a column buffer is needed
    // @modifies info: The alkaike information critierion of this set w.r.t the target distribution
    // @modifies potential: The maximum reduction in loss if all equivalent classes are relabelled (without considering complexity penalty)
    // @modifies min_obj: The minimal objective incurred if all equivalent classes are optimally labelled without considering complexity penalty
    // @modifies max_loss: The loss incurred if the capture set is left unsplit and the best single label is chosen
    // @modifies target_index: The label to choose if left unsplit
    void summary(Bitmask const & capture_set, float & info, float & potential, float & min_obj, float & max_loss, unsigned int & target_index, unsigned int id) const;

    // @param feature_index: the index of the binary feature to use bisect the set
    // @param positive: if true, modifies set to reflect the part of the bisection that responds positive to the binary feature
    //                  if false, the other part of the bisection is used
    // @param set: indicates the captured set of samples to be bisected
    // @modifies set: the captured set will be overwritten to reflect the subset extracted from the bisection
    //                this can be either the positive or negative subset depending on the positive argument
    void subset(unsigned int feature_index, bool positive, Bitmask & set) const;
    // Convenient alias for performing both negative and positive tests
    void subset(unsigned int feature_index, Bitmask & negative, Bitmask & positive) const;

    // @param set: The indicator for each equivalent groups are contained by this problem
    // @param buffer: a buffer used for bitwise operations
    // @param i: feature index for pairwise comparison
    // @param j: other feature index for pairwise comparison
    // @return distance: The maximum change in objective value if feature i is swapped for j or vice versa
    float distance(Bitmask const & set, unsigned int i, unsigned int j, unsigned int id) const;

    void tile(Bitmask const & filter, Bitmask const & selector, Tile & tile_output, std::vector< int > & order, unsigned int id) const;

    void target_value(Bitmask capture_set, std::string & prediction_value) const;

    mutable int summary_calls = 0;
    mutable int compute_kmeans_calls = 0;
    mutable int summary_calls_has_gap = 0;
    mutable double cum_percent = 0;

private:
    static bool index_comparator(const std::pair< unsigned int, unsigned int > & left, const std::pair< unsigned int, unsigned int > & right);

    // The dimensions of the dataset
    //  Dim-0 = Number of samples
    //  Dim-1 = Number of binary features
    //  Dim-2 = Number of classes
    std::tuple< unsigned int, unsigned int, unsigned int > shape;
    unsigned int _size; // shortcut for number of samples

    // std::vector< Bitmask > columns; // Binary representation of columns
    // std::vector< std::vector< float > > distributions; // Class distributions for each row

    // N := Number of datapoints in the original dataset
    // E := Number of equivalent points (clusters) in the original dataset 
    std::vector< Bitmask > features; // Binary representation of columns
    std::vector< float > targets; // Float vector of size N
    std::vector< float > clustered_targets; // Float vector of size E
    std::vector< float > cluster_loss; // Float vector of size E
    std::vector< int > clustered_targets_mapping; // Size N to index in E (value ranging from 0 - (E-1))
    std::vector< Bitmask > rows; // Binary representation of rows
    std::vector< Bitmask > feature_rows; // Binary representation of rows
    // Bitmask majority; // Binary representation of columns
    std::vector<float> weights; // Float vector of size N, weights of sample
    float loss_normalizer;
    // Index index; // Index for calculating summaries
    // Index distance_index; // Index for calculating feature distances

    void construct_bitmasks(std::istream & input_stream);
//    void construct_cost_matrix(void);
//    void parse_cost_matrix(std::istream & input_stream);
//    void aggregate_cost_matrix(void);
    void construct_clusters(void);

    void construct_ordering(void);
    
    void normalize_data(void);
    
    float compute_kmeans_lower_bound(Bitmask capture_set) const;
    float compute_equivalent_points_lower_bound(Bitmask capture_set) const;

    float ssq_loss(Bitmask capture_set) const;
    float ssq_loss(std::vector< int > capture_set_idx, float & sum_weights) const;

    float sabs_loss(Bitmask capture_set) const;
    float sabs_loss(std::vector< int > capture_set_idx, float & sum_weights) const;

    float compute_loss(Bitmask capture_set) const;
    float compute_loss(std::vector< int > capture_set_idx, float & sum_weights) const;
};

#endif