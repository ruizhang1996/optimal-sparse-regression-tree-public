# OSRT Documentation
Implementation of [Optimal Sparse Regression Tree (OSRT)](https://arxiv.org/pdf/). This is implemented based on [Generalized Optimal Sparse Decision Tree framework (GOSDT)](https://github.com/ubc-systopia/gosdt-guesses). If you need classification trees, please use GOSDT.

![image](https://user-images.githubusercontent.com/60573138/189567116-0b719588-d670-4038-a242-2cc4be26816b.png)


# Table of Content
- [Quick Start](#quick-start)
- [Usage](#usage)
- [Configuration](#configuration)
- [Development](#development)
- [Dependencies](#dependencies)
- [License](#license)
- [FAQs](#faqs)

---

# Quick Start

### Build and Installation
```
./autobuild --install-python
```
_If you have multiple Python installations, please make sure to build and install using the same Python installation as the one intended for interacting with this library._


### Fitting the Data

```python
import gosdt

with open ("data.csv", "r") as data_file:
    data = data_file.read()

with open ("config.json", "r") as config_file:
    config = config_file.read()


print("Config:", config)
print("Data:", data)

gosdt.configure(config)
result = gosdt.fit(data)

print("Result: ", result)
print("Time (seconds): ", gosdt.time())
print("Iterations: ", gosdt.iterations())
print("Graph Size: ", gosdt.size())
```

# Usage

Guide for end-users who want to use the library without modification.

Describes how to install and use the library as a stand-alone command-line program or as an embedded extension in a larger project.
Currently supported as a Python extension.

## Installing Dependencies
Refer to [**Dependency Installation**](/doc/dependencies.md##Installation)

## As a Stand-Alone Command Line Program
### Installation
```
./autobuild --install
```

### Executing the Program
```bash
gosdt dataset.csv config.json
# or 
cat dataset.csv | gosdt config.json >> output.json
```

For examples of dataset files, refer to `experiments/datasets/airfoil/airfoil.csv`.
For an example configuration file, refer to `experiments/configurations/config.json`.
For documentation on the configuration file, refer to [**Dependency Installation**](/doc/configuration.md)

## As a Python Library with C++ Extensions
### Build and Installation
```
./autobuild --install-python
```
_If you have multiple Python installations, please make sure to build and install using the same Python installation as the one intended for interacting with this library._


### Importing the C++ Extension
```python
import gosdt

with open ("data.csv", "r") as data_file:
    data = data_file.read()

with open ("config.json", "r") as config_file:
    config = config_file.read()


print("Config:", config)
print("Data:", data)

gosdt.configure(config)
result = gosdt.fit(data)

print("Result: ", result)
print("Time (seconds): ", gosdt.time())
print("Iterations: ", gosdt.iterations())
print("Graph Size: ", gosdt.size())
```

### Importing Extension with local Python Wrapper
```python
import pandas as pd
import numpy as np
from model.gosdt import GOSDT

dataframe = pd.DataFrame(pd.read_csv("experiments/datasets/airfoil/airfoil.csv"))

X = dataframe[dataframe.columns[:-1]]
y = dataframe[dataframe.columns[-1:]]

hyperparameters = {
    "regularization": 0.1,
    "time_limit": 3600,
    "verbose": True,
}

model = GOSDT(hyperparameters)
model.fit(X, y)
print("Execution Time: {}".format(model.time))

prediction = model.predict(X)
training_accuracy = model.score(X, y)
print("Training Accuracy: {}".format(training_accuracy))
print(model.tree)
```

---
# Configuration

Details on the configuration options.

```bash
gosdt dataset.csv config.json
# or
cat dataset.csv | gosdt config.json
```

Here the file `config.json` is optional.
There is a default configuration which will be used if no such file is specified.

## Configuration Description

The configuration file is a JSON object and has the following structure and default values:
```json
{
  "k_cluster": true,

  "diagnostics": false,
  "verbose": true,

  "regularization": 0.05,
  "depth_budget": 5,
  "uncertainty_tolerance": 0.0,
  "upperbound": 0.0,

  "model_limit": 10000,
  "precision_limit": 0,
  "stack_limit": 0,
  "tile_limit": 0,
  "time_limit": 0,
  "worker_limit": 1,

  "model": "",
  "profile": "",
  "timing": "",
  "trace": "",
  "tree": "",
  "datatset_encoding": "",

  "metric": "L2",
  "weights": []
}
```

### Flags

**k_cluster**
 - Values: true or false
 - Description: Enables usage of the k-Means Equivalent Points Bound

**diagnostics**
 - Values: true or false
 - Description: Enables printing of diagnostic trace when an error is encountered to standard output

**verbose**
 - Values: true or false
 - Description: Enables printing of configuration, progress, and results to standard output

 ### Tuners

 **regularization**
 - Values: Decimal within range [0,1]
 - Description: Used to penalize complexity. A complexity penalty is added to the risk in the following way.
   ```
   ComplexityPenalty = # Leaves x regularization
   ```
   
 **depth_budget**
 - Values: Integer 
 - Description: The maximum tree depth for solutions, counting a tree with just the root node as depth 1. 0 means unlimited.

 **uncertainty_tolerance**
 - Values: Decimal within range [0,1]
 - Description: Used to allow early termination of the algorithm. Any models produced as a result are guaranteed to score within the lowerbound and upperbound at the time of termination. However, the algorithm does not guarantee that the optimal model is within the produced model unless the uncertainty value has reached 0.

 **upperbound**
 - Values: Decimal within range [0,1]
 - Description: Used to limit the risk of model search space. This can be used to ensure that no models are produced if even the optimal model exceeds a desired maximum risk. This also accelerates learning if the upperbound is taken from the risk of a nearly optimal model.

### Limits

**model_limit**
 - Values: Decimal greater than or equal to 0
 - Description: The maximum number of models that will be extracted into the output.
 - Special Cases: When set to 0, no output is produced.

**precision_limit**
 - Values: Decimal greater than or equal to 0
 - Description: The maximum number of significant figures considered when converting ordinal features into binary features.
 - Special Cases: When set to 0, no limit is imposed.

**stack_limit**
 - Values: Decimal greater than or equal to 0
 - Description: The maximum number of bytes considered for use when allocating local buffers for worker threads.
 - Special Cases: When set to 0, all local buffers will be allocated from the heap.

**tile_limit**
 - Values: Decimal greater than or equal to 0
 - Description: The maximum number of bits used for the finding tile-equivalence
 - Special Cases: When set to 0, no tiling is performed.

**time_limit**
 - Values: Decimal greater than or equal to 0
 - Description: A time limit upon which the algorithm will terminate. If the time limit is reached, the algorithm will terminate with an error.
 - Special Cases: When set to 0, no time limit is imposed.

**worker_limit**
 - Values: Decimal greater than or equal to 1
 - Description: The maximum number of threads allocated to executing th algorithm.
 - Special Cases: When set to 0, a single thread is created for each core detected on the machine.

### Files

**model**
 - Values: string representing a path to a file.
 - Description: The output models will be written to this file.
 - Special Case: When set to empty string, no model will be stored.

**profile**
 - Values: string representing a path to a file.
 - Description: Various analytics will be logged to this file.
 - Special Case: When set to empty string, no analytics will be stored.

**timing**
 - Values: string representing a path to a file.
 - Description: The training time will be appended to this file.
 - Special Case: When set to empty string, no training time will be stored.

**trace**
 - Values: string representing a path to a directory.
 - Description: snapshots used for trace visualization will be stored in this directory
 - Special Case: When set to empty string, no snapshots are stored.

**tree**
 - Values: string representing a path to a directory.
 - Description: snapshots used for trace-tree visualization will be stored in this directory
 - Special Case: When set to empty string, no snapshots are stored.

## Optimizing Different Loss Functions

OSRT currently supports weighted L1 and L2 losses.

**metric**
 - Values: string of `L1` or `L2`
 - Description: specify the loss that OSRT is using

**weights**
 - Values: array of decimal within [0, 1] of length of the dataset size
 - Description: specify the weight for the given loss
 - Special Case: When set to empty array, all data points are weighted equally.


---

# Development


Guide for developers who want to use, modify and test the library.

Describes how to install and use the library with details on project structure.

## Repository Structure
 - **notebooks** - interactive notebooks for examples and visualizations
 - **experiments** - configurations, datasets, and models to run experiments
 - **doc** - documentation
 - **python** - code relating to the Python implementation and wrappers around C++ implementation
 - **auto** - automations for checking and installing project dependencies
 - **dist** - compiled binaries for distribution
 - **build** - compiled binary objects and other build artifacts
 - **lib** - headers for external libraries
 - **log** - log files
 - **src** - source files
 - **test** - test files

## Installing Dependencies
Refer to [**Dependency Installation**](/doc/dependencies.md##Installation)

## Build Process
 - **Check Updates to the Dependency Tests or Makefile** 
   ```
   ./autobuild --regenerate
   ```
 - **Check for Missing Dependencies** 
   ```
   ./autobuild --configure --enable-tests
   ```
 - **Build and Run Test Suite**
   ```
   ./autobuild --test
   ```
 - **Build and Install Program**
   ```
   ./autobuild --install --enable-tests
   ```
 - **Run the Program** 
   ```
   gosdt dataset.csv config.json
   ```
 - **Build and Install the Python Extension**
   ```
   ./autobuild --install-python
   ```
 For a full list of build options, run `./autobuild --help`

---

# Dependencies

List of external dependencies

The following dependencies need to be installed to build the program. 
 - [**Boost**](https://www.boost.org/) - Collection of portable C++ source libraries
 - [**GMP**](http://gmplib.org/) - Collection of functions for high-precision artihmetics
 - [**Intel TBB**](https://www.threadingbuildingblocks.org/) - Rich and complete approach to parallelism in C++
 - [**WiredTiger**](https://source.wiredtiger.com/2.5.2/index.html) - WiredTiger is an high performance, scalable, production quality, NoSQL, Open Source extensible platform for data management
 - [**OpenCL**](https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=14&cad=rja&uact=8&ved=2ahUKEwizj4n2k8LlAhVcCTQIHZlADscQFjANegQIAhAB&url=https%3A%2F%2Fwww.khronos.org%2Fregistry%2FOpenCL%2F&usg=AOvVaw3JjOwbrewRqPxpTXRZ6vN9)(Optional) - A framework for execution across heterogeneous hardware accelerators.

### Bundled Dependencies
The following dependencies are included as part of the repository, thus requiring no additional installation.
 - [**nlohmann/json**](https://github.com/nlohmann/json) - JSON Parser
 - [**ben-strasser/fast-cpp-csv-parser**](https://github.com/ben-strasser/fast-cpp-csv-parser) - CSV Parser
 - [**OpenCL C++ Bindings 1.2**](https://www.khronos.org/registry/OpenCL/specs/opencl-cplusplus-1.2.pdf) - OpenCL bindings for GPU computing

 ### Installation
 Install these using your system package manager.
 There are also installation scripts provided for your convenience: **trainer/auto**
 
 These currently support interface with **brew** and **apt**
  - **Boost** - `auto/boost.sh --install`
  - **GMP** - `auto/gmp.sh --install`
  - **Intel TBB** - `auto/tbb.sh --install`
  - **WiredTiger** - `auto/wiredtiger.sh --install`
  - **OpenCL** - `auto/opencl.sh --install`

---

# FAQs
_Note we will work on this and following sections later_

If you run into any issues, consult the [**FAQs**](/doc/faqs.md) first. 

---

# License

Licensing information

---

**Inquiries**

For general inquiries, send an email to `r.zhang@duke.edu`
