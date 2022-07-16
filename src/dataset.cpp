#include "dataset.hpp"

Dataset::Dataset(void) {}
Dataset::~Dataset(void) {}

Dataset::Dataset(std::istream & data_source) { load(data_source); }

// Loads the binary-encoded data set into precomputed form:
void Dataset::load(std::istream & data_source) {
    // Construct all rows, features in binary form
    construct_bitmasks(data_source);
    // Normalize target column
    normalize_data();

    // Build cluster and cluster target indicating whether a point is the equivalent set
    construct_clusters();
    
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
}

void Dataset::construct_bitmasks(std::istream & data_source) {
    this -> encoder = Encoder(data_source);
    std::vector< Bitmask > rows = this -> encoder.read_binary_rows();
    unsigned int number_of_samples = this -> encoder.samples(); // Number of samples in the dataset
    unsigned int number_of_binary_features = this -> encoder.binary_features(); // Number of source features
    // unsigned int number_of_binary_targets = this -> encoder.binary_targets(); // Number of target features
    this -> _size = number_of_samples;
    this -> weights = encoder.get_weights();
    this -> rows = this -> encoder.read_binary_rows();

    this -> features.resize(number_of_binary_features, number_of_samples);
    this -> feature_rows.resize(number_of_samples, number_of_binary_features);
    this -> targets = encoder.read_numerical_targets();

    for (unsigned int i = 0; i < number_of_samples; ++i) {
        for (unsigned int j = 0; j < number_of_binary_features; ++j) {
            this -> features[j].set(i, bool(rows[i][j]));
            this -> feature_rows[i].set(j, bool(rows[i][j]));
        }
    }
    //TODO: check depth for regression
    this -> shape = std::tuple< int, int, int >(this -> rows.size(), this -> features.size(), this -> targets.size());
};

void Dataset::construct_clusters(void) {
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
    std::vector< double > cluster_loss;
    std::vector< int > clustered_targets_mapping(size());
    // std::vector<double> cluster_weights;
    int cluster_idx = 0;
    for (auto it = clusters.begin(); it != clusters.end(); ++it) {
        std::vector< int > const & cluster = it -> second;
        std::vector< double > cluster_values;
        double sum_weights;
        for (int idx : cluster) {
            cluster_values.emplace_back(weights[idx] * targets[idx]);
            clustered_targets_mapping[idx] = cluster_idx;
        }
        cluster_loss.emplace_back(ssq_loss(cluster, sum_weights));
        double sum = std::accumulate(cluster_values.begin(), cluster_values.end(), 0.0);
        double target = sum / sum_weights;
        clustered_targets.emplace_back(target);
        cluster_order.emplace_back(cluster_idx++);
    }

    // Step 3: Sort clustered target values and update data index to cluster
    // index mapping
    auto compi = [clustered_targets](size_t i, size_t j) {
        return clustered_targets[i] < clustered_targets[j];
    };
    std::sort(cluster_order.begin(), cluster_order.end(), compi);
    std::vector< double > sorted_clustered_targets(clustered_targets.size());
    std::vector< double > sorted_cluster_loss(clustered_targets.size());
    for (int i = 0; i < clustered_targets.size(); i++) {
        sorted_clustered_targets[i] = clustered_targets[cluster_order[i]];
        sorted_cluster_loss[i] = cluster_loss[cluster_order[i]];
    }
    std::vector< int > inverted_cluster_order(cluster_order.size());
    for (int i = 0; i < cluster_order.size(); i++) {
        inverted_cluster_order[cluster_order[i]] = i;
    }
    for (int i = 0; i < size(); i++) {
        clustered_targets_mapping[i] = inverted_cluster_order[clustered_targets_mapping[i]];
    }

    this -> clustered_targets = sorted_clustered_targets;
    this -> cluster_loss = sorted_cluster_loss;
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

void Dataset::target_value(Bitmask capture_set, std::string & prediction_value) const{
    int max = capture_set.size();
    double sum = 0.0;
    for (int i = capture_set.scan(0, true); i < max; i = capture_set.scan(i + 1, true)) {
        sum += targets[i];
    }
    int count = capture_set.count();
    prediction_value = std::to_string(sum/count);
}
double Dataset::ssq_loss(Bitmask capture_set) const {
    double cumsum1 = 0;
    double cumsum2 = 0;
    double wsum = 0;
    int max = capture_set.size();
    for (int i = capture_set.scan(0, true); i < max; i = capture_set.scan(i + 1, true)) {
        cumsum1 += weights[i] * targets[i];
        cumsum2 += weights[i] * targets[i] * targets[i];
        wsum += weights[i];
    }
    // int count = capture_set.count();
    // TODO: check overhead of weights implementation
    return cumsum2 - cumsum1 * cumsum1 / wsum;
}

double Dataset::ssq_loss(std::vector< int > capture_set_idx, double & sum_weights) const {
    double cumsum1 = 0;
    double cumsum2 = 0;
    // int count = 0;
    double wsum = 0;
    for (int i : capture_set_idx) {
        cumsum1 += weights[i] * targets[i];
        cumsum2 += weights[i] * targets[i] * targets[i];
        wsum += weights[i];
    }
    sum_weights = wsum;
    return cumsum2 - cumsum1 * cumsum1 / wsum;
}

double Dataset::sad_loss(Bitmask capture_set) const {
    double cumsum1 = 0;
    double cumsum2 = 0;
    int max = capture_set.size();
    for (int i = capture_set.scan(0, true); i < max; i = capture_set.scan(i + 1, true)) {
        cumsum1 += targets[i];
        cumsum2 += targets[i] * targets[i];
    }
    int count = capture_set.count();
    return cumsum2 - cumsum1 * cumsum1 / count;
}

void Dataset::normalize_data() {
    double loss_normalizer;
    switch (Configuration::metric) {
        case Configuration::l2_loss: {
            loss_normalizer = std::sqrt(ssq_loss(Bitmask(size(), true)));
            break;
        }
        case Configuration::l1_loss: {
            loss_normalizer = sad_loss(Bitmask(size(), true));
            break;
        }
        default:{
            std::stringstream reason;
            reason << "Unsupported Metric: " << Configuration::metric;
            throw IntegrityViolation("Dataset::normalize_data", reason.str());
        }
    }


    for (int i = 0; i < size(); i++) {
        targets[i] = targets[i] / loss_normalizer;
    }
    std::cout << "loss_normalizer: " << loss_normalizer << std::endl;
}

// N := Number of datapoints in the original dataset
// E := Number of equivalent points (clusters) in the original dataset 
double Dataset::compute_kmeans_lower_bound(Bitmask capture_set) const {
    int max = capture_set.size();
    
    int normalizer = capture_set.count();
    double reg = Configuration::regularization;
    
    if (normalizer == 1) {
        return reg;
    }
    double correction = 0;
    
    // count: E
    std::vector< double > count(clustered_targets.size());
    for (int i = capture_set.scan(0, true); i < max; i = capture_set.scan(i + 1, true)) {
        count[clustered_targets_mapping[i]] += weights[i];
        correction += weights[i] * targets[i] * targets[i];
    }

    std::vector< double > weights;
    std::vector< double > values;

    // TODO: we can precompute all the sum of squares so one less passthrough
    for (int i = 0; i < count.size(); i++) {
        if (count[i] > 0) {
            weights.emplace_back(count[i]);
            values.emplace_back(clustered_targets[i]);
            correction -= clustered_targets[i] * clustered_targets[i] * count[i];
        }
    }
    
    int N = weights.size();
    int Kmax = std::min(100, N);
    std::vector< std::vector< ldouble > > S( Kmax, std::vector<ldouble>(N) );
    std::vector< std::vector< size_t > > J( Kmax, std::vector<size_t>(N) );
    
    // TODO: add dynamically assigned Kmax via scope
    ldouble min = fill_dp_matrix_dynamic_stop(values, weights, S, J, reg) + correction;

    return min;

}
double Dataset::compute_equivalent_points_lower_bound(Bitmask capture_set) const {
    int max = capture_set.size();
    
    
    double sum_loss = 0;
    std::vector< int > count(clustered_targets_mapping.size());
    for (int i = capture_set.scan(0, true); i < max; i = capture_set.scan(i + 1, true)) {
        if (count[clustered_targets_mapping[i]] == 0) {
            sum_loss += cluster_loss[clustered_targets_mapping[i]];
        }
        count[clustered_targets_mapping[i]]++;
    }
    
    return sum_loss;

}

// @param feature_index: selects the feature on which to split
// @param positive: determines whether to provide the subset that tests positive on the feature or tests negative on the feature
// @param set: pointer to bit blocks which indicate the original set before splitting
// @modifies set: set will be modified to indicate the positive or negative subset after splitting
// @notes the set in question is an array of the type bitblock. this allows us to specify the set using a stack-allocated array
void Dataset::subset(unsigned int feature_index, bool positive, Bitmask & set) const {
    // Performs bit-wise and between feature and set with possible bit-flip if performing negative test
    this -> features[feature_index].bit_and(set, !positive);
    if (Configuration::depth_budget != 0){ set.set_depth_budget(set.get_depth_budget()-1);} //subproblems have one less depth_budget than their parent
}

void Dataset::subset(unsigned int feature_index, Bitmask & negative, Bitmask & positive) const {
    // Performs bit-wise and between feature and set with possible bit-flip if performing negative test
    this -> features[feature_index].bit_and(negative, true);
    this -> features[feature_index].bit_and(positive, false);
    if (Configuration::depth_budget != 0){
        negative.set_depth_budget(negative.get_depth_budget()-1);
        positive.set_depth_budget(positive.get_depth_budget()-1);
    } //subproblems have one less depth_budget than their parent
}

// Performance Boost ideas:
// 1. Store everything in summary 
// 2. Introduce scope and apply it to kmeans so that it could be even tighter 
// 3. Check equiv (points lower bound + 2 * reg) before using Kmeans to
//    determine if we need more split as it has a way lower overhead

void Dataset::summary(Bitmask const & capture_set, float & info, float & potential, float & min_obj, float & max_loss, unsigned int & target_index, unsigned int id) const {
    summary_calls++;
    Bitmask & buffer = State::locals[id].columns[0];
    //unsigned int * distribution; // The frequencies of each class
    //distribution = (unsigned int *) alloca(sizeof(unsigned int) * depth());

    unsigned int cost_minimizer = 0;

    max_loss = ssq_loss(capture_set);
    //float max_cost_reduction = 0.0;
    float equivalent_point_loss = 0.0;
    //float support = (float)(capture_set.count()) / (float)(height());
    float information = 0.0;
    
    // if (summary_calls > 30000) {
    //     equivalent_point_loss = 2 * Configuration::regularization + compute_equivalent_points_lower_bound(capture_set);
    // } else {
    //     equivalent_point_loss = compute_kmeans_lower_bound(capture_set);
    // }
    // assert(min_cost + Configuration::regularization < equivalent_point_loss_1 || equivalent_point_loss_1 < equivalent_point_loss);
    // equivalent_point_loss = 2 * Configuration::regularization + compute_equivalent_points_lower_bound(capture_set);
    kmeans_accessor stored_kmeans_accessor;
    if (State::graph.kmeans.find(stored_kmeans_accessor, capture_set)) {
        equivalent_point_loss = stored_kmeans_accessor->second;
        stored_kmeans_accessor.release();
    } else {
        equivalent_point_loss = compute_kmeans_lower_bound(capture_set);
        auto new_kmeans = std::make_pair(capture_set, equivalent_point_loss);
        State::graph.kmeans.insert(new_kmeans);
        compute_kmeans_calls++;
    }

    // float equivalent_point_loss_1 = 2 * Configuration::regularization + compute_equivalent_points_lower_bound(capture_set);
    // float max_loss_1 = min_cost + Configuration::regularization;
    // float diff = equivalent_point_loss - equivalent_point_loss_1;

    // float gap = max_loss_1 - equivalent_point_loss_1;
    // if (gap > 0.0001) {
    //     float percent = diff / gap;
    //     summary_calls_has_gap++;
    //     cum_percent += percent;
        
    // }

    min_obj = equivalent_point_loss;
    potential = max_loss + Configuration::regularization - min_obj;
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