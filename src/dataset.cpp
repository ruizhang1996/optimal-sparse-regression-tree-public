#include "dataset.hpp"

Dataset::Dataset(void) {}
Dataset::~Dataset(void) {}

Dataset::Dataset(std::istream & data_source) { load(data_source); }

// Loads the binary-encoded data set into precomputed form:
// Step 1: Build bitmasks for each column and row of the dataset, allowing fast parallel operations
// Step 2: Build the cost matrix. Either from values in an input file or a specified mode.
// Step 3: Compute columnar aggregations of the cost matrix to speed up some calculations from K^2 to K
// Step 4: Data set shape is stored
//   The overall shape of the data set is stored for indexing later
void Dataset::load(std::istream & data_source) {
    // Step 1: Construct all rows, features, and targets in binary form
    construct_bitmasks(data_source);


    // Step 2: Initialize the cost matrix
    // construct_cost_matrix();

    // Step 3: Build the majority and minority costs based on the cost matrix
    // aggregate_cost_matrix();
    
    // construct_ordering();
    normalize_data();

    // Step 4: Build the majority bitmask indicating whether a point is in the majority group
    construct_majority();
    
    if (Configuration::verbose) {
        std::cout << "Dataset Dimensions: " << height() << " x " << width() << " x " << depth() << std::endl;
    }
    return;
}

void Dataset::clear(void) {
    this -> features.clear();
    this -> targets.clear();
    this -> rows.clear();
    this -> feature_rows.clear();
    this -> target_rows.clear();
    // this -> costs.clear();
    this -> match_costs.clear();
    this -> mismatch_costs.clear();
    this -> max_costs.clear();
    this -> min_costs.clear();
    this -> diff_costs.clear();
    this -> majority = Bitmask();
}

void Dataset::construct_bitmasks(std::istream & data_source) {
    this -> encoder = Encoder(data_source);
    std::vector< Bitmask > rows = this -> encoder.read_binary_rows();
    unsigned int number_of_samples = this -> encoder.samples(); // Number of samples in the dataset
    unsigned int number_of_rows = 0; // Number of samples after compressions
    unsigned int number_of_binary_features = this -> encoder.binary_features(); // Number of source features
    unsigned int number_of_binary_targets = this -> encoder.binary_targets(); // Number of target features
    this -> _size = number_of_samples;

    this -> rows = this -> encoder.read_binary_rows();

    this -> features.resize(number_of_binary_features, number_of_samples);
    this -> feature_rows.resize(number_of_samples, number_of_binary_features);
    this -> targets.resize(number_of_binary_targets, number_of_samples);
    this -> target_rows.resize(number_of_samples, number_of_binary_targets);
    
    this -> targets = encoder.read_numerical_targets();
    this -> target_rows = encoder.read_numerical_targets();


    for (unsigned int i = 0; i < number_of_samples; ++i) {
        for (unsigned int j = 0; j < number_of_binary_features; ++j) {
            this -> features[j].set(i, bool(rows[i][j]));
            this -> feature_rows[i].set(j, bool(rows[i][j]));
        }
    }
    this -> shape = std::tuple< int, int, int >(this -> rows.size(), this -> features.size(), this -> targets.size());
};

void Dataset::construct_majority(void) {
    std::vector< Bitmask > keys(height(), width());
    for (unsigned int i = 0; i < height(); ++i) {
        for (unsigned int j = 0; j < width(); ++j) {
            keys[i].set(j, bool(this -> rows[i][j]));
        }
    }
    
    // Step 1: Construct a map from the binary features to their clusters,
    // indicated by their indices in capture set
    std::unordered_map< Bitmask, std::vector< int > > clusters;
    for (int i = 0; i < height(); ++i) {
        Bitmask const & key = keys.at(i);
        clusters[key].emplace_back(i);
    }
    
    // Step 2: Convert clusters map into an array by taking the mean of each
    // cluster, initialize unsorted order, and initialize data index to cluster
    // index mapping
    std::vector< double > clustered_targets;
    std::vector< int > cluster_order;
    std::vector< int > clustered_targets_mapping(size());
    int cluster_idx = 0;
    for (auto it = clusters.begin(); it != clusters.end(); ++it) {
        std::vector< int > const & cluster = it -> second;
        std::vector< double > cluster_values;
        for (int idx : cluster) {
            cluster_values.emplace_back(targets[idx]);
            clustered_targets_mapping[idx] = cluster_idx;
        }
        double sum = std::accumulate(cluster_values.begin(), cluster_values.end(), 0.0);
        double mean = sum / cluster_values.size();
        clustered_targets.emplace_back(mean);
        cluster_order.emplace_back(cluster_idx++);
    }
    
    // Step 3: Sort clustered target values and update data index to cluster
    // index mapping
    auto compi = [clustered_targets](size_t i, size_t j) {
        return clustered_targets[i] < clustered_targets[j];
    };
    std::sort(cluster_order.begin(), cluster_order.end(), compi);
    std::vector< double > sorted_clustered_targets(clustered_targets.size());
    for (int i = 0; i < clustered_targets.size(); i++) {
        sorted_clustered_targets[i] = clustered_targets[cluster_order[i]];
    }
    std::vector< int > inverted_cluster_order(cluster_order.size());
    for (int i = 0; i < cluster_order.size(); i++) {
        inverted_cluster_order[cluster_order[i]] = i;
    }
    for (int i = 0; i < size(); i++) {
        clustered_targets_mapping[i] = inverted_cluster_order[clustered_targets_mapping[i]];
    }

    this -> clustered_targets = sorted_clustered_targets;
    this -> clustered_targets_mapping = clustered_targets_mapping;

}

// TODO: investigate 
float Dataset::distance(Bitmask const & set, unsigned int i, unsigned int j, unsigned int id) const {
    return 0;
}


void Dataset::construct_ordering(void) {
    auto targets = this -> targets;
    auto compi = [targets](size_t i, size_t j) {
        return targets[i] < targets[j];
    };
    std::vector<int> order(size());

    for (size_t i=0; i<order.size(); ++i) {
        order[i] = i;
    }
    
    std::sort(order.begin(), order.end(), compi);

    // this -> targets_ordering = order;
}

double Dataset::mse_loss(Bitmask capture_set) const {
    int max = capture_set.count();
    double cumsum1 = 0;
    double cumsum2 = 0;
    for (int i = capture_set.scan(0, true); i < max; i = capture_set.scan(i + 1, true)) {
        cumsum1 += targets[i];
        cumsum2 += targets[i] * targets[i];
    }
    return cumsum2 / max - cumsum1 * cumsum1 * 2 / max / max;
}

double Dataset::compute_loss(Bitmask capture_set) const {
    // return compute_loss(capture_set) / loss_normalizer;
    return 0;
}

void Dataset::normalize_data() {

    double loss_normalizer = std::sqrt(mse_loss(Bitmask(size(), true)));

    for (int i = 0; i < size(); i++) {
        targets[i] = targets[i] / loss_normalizer;
    }
}

double Dataset::compute_kmeans_lower_bound(Bitmask capture_set) const {
    int max = capture_set.size();
    
    int normalizer = capture_set.count();
    double reg = Configuration::regularization;
    
    if (normalizer == 1) {
        return mse_loss(capture_set) + reg;
    }
    
    std::vector< int > count(clustered_targets_mapping.size());
    for (int i = capture_set.scan(0, true); i < max; i = capture_set.scan(i + 1, true)) {
        count[clustered_targets_mapping[i]]++;
    }
    
    // Why do you need this? 
    std::vector< double > weights;
    std::vector< double > values;
    for (int i = 0; i < count.size(); i++) {
        if (count[i] > 0) {
            weights.emplace_back(count[i]);
            values.emplace_back(clustered_targets[i]);
        }
    }
    
    int N = weights.size();
    int Kmax = std::min(10, N);
    std::vector< std::vector< ldouble > > S( Kmax, std::vector<ldouble>(N) );
    std::vector< std::vector< size_t > > J( Kmax, std::vector<size_t>(N) );
    fill_dp_matrix(values, weights, S, J, "linear", L2);
    
    
    long double min = std::numeric_limits<double>::max();;
    int argmin = -1;
    for (int i = 0; i < Kmax; i++) {
        ldouble obj = S[i][N-1] / normalizer + (i + 1) * reg;
        if (min > obj) {
            min = obj;
            argmin = i;
        }
    }

    if (argmin == 10 - 1) {
        std::cout << "WARNING";
    }
    return min;

}

// @param feature_index: selects the feature on which to split
// @param positive: determines whether to provide the subset that tests positive on the feature or tests negative on the feature
// @param set: pointer to bit blocks which indicate the original set before splitting
// @modifies set: set will be modified to indicate the positive or negative subset after splitting
// @notes the set in question is an array of the type bitblock. this allows us to specify the set using a stack-allocated array
void Dataset::subset(unsigned int feature_index, bool positive, Bitmask & set) const {
    // Performs bit-wise and between feature and set with possible bit-flip if performing negative test
    this -> features[feature_index].bit_and(set, !positive);
}

void Dataset::subset(unsigned int feature_index, Bitmask & negative, Bitmask & positive) const {
    // Performs bit-wise and between feature and set with possible bit-flip if performing negative test
    this -> features[feature_index].bit_and(negative, true);
    this -> features[feature_index].bit_and(positive, false);
}

void Dataset::summary(Bitmask const & capture_set, float & info, float & potential, float & min_loss, float & max_loss, unsigned int & target_index, unsigned int id) const {
    summary_calls++;
    Bitmask & buffer = State::locals[id].columns[0];
    unsigned int * distribution; // The frequencies of each class
    distribution = (unsigned int *) alloca(sizeof(unsigned int) * depth());

    float min_cost = std::numeric_limits<float>::max();
    unsigned int cost_minimizer = 0;

    min_cost = mse_loss(capture_set);
    float max_cost_reduction = 0.0;
    float equivalent_point_loss = 0.0;
    float support = (float)(capture_set.count()) / (float)(height());
    float information = 0.0;
    
    // equivalent_point_loss = 2 * Configuration::regularization;
    equivalent_point_loss = compute_kmeans_lower_bound(capture_set);

    min_loss = equivalent_point_loss;
    max_loss = min_cost;
    potential = max_loss - min_loss;
    info = information;
    target_index = cost_minimizer;
}

// Assume that data is already of the right size
void Dataset::tile(Bitmask const & capture_set, Bitmask const & feature_set, Tile & tile, std::vector< int > & order, unsigned int id) const {
    tile.content() = capture_set;
    tile.width(0);
    return;
}


unsigned int Dataset::height(void) const {
    return std::get<0>(this -> shape);
}

unsigned int Dataset::width(void) const {
    return std::get<1>(this -> shape);
}

unsigned int Dataset::depth(void) const {
    return std::get<2>(this -> shape);
}

unsigned int Dataset::size(void) const {
    return this -> _size;
}

bool Dataset::index_comparator(const std::pair< unsigned int, unsigned int > & left, const std::pair< unsigned int, unsigned int > & right) {
    return left.second < right.second;
}