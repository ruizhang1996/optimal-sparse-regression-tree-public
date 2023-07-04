#include "main.hpp"

int main(int argc, char *argv[]) {


	bool standard_input = false;
	// Check program input
	if ((standard_input && (argc < 1 || argc > 2)) || (!standard_input && (argc < 2 || argc > 3))) {
		std::cout << "Usage: gosdt [path to feature set] ?[path to config]" << std::endl;
		return 0;
	}
	if (argc >= 2 && !std::ifstream(argv[1]).good()) {
		std::cout << "File Not Found: " << argv[1] << std::endl;
		return 1;
	}
	if (argc >= 3 && !std::ifstream(argv[2]).good()) {
		std::cout << "File Not Found: " << argv[2] << std::endl;
		return 1;
	}
	
	if ((standard_input && argc == 2) || (!standard_input && argc == 3)) {
		// Use custom configuration if provided
		std::ifstream configuration(argv[argc - 1]);
		Configuration::configure(configuration);
	}

	// Print messages to help user ensure they've provided the correct inputs
	if (Configuration::verbose) {
		std::cout << "Generalized Optimal Sparse Regression Tree" << std::endl;
		std::cout << "Using data set: " << argv[1] << std::endl;
	}
	std::string result;
	GOSDT model;
	if (standard_input) {
		model.fit(std::cin, result);
	} else {
		std::ifstream data(argv[1]);
		model.fit(data, result);
	}
	if (Configuration::model == "" || Configuration::verbose) { std::cout << result << std::endl; }
	return 0;
}
