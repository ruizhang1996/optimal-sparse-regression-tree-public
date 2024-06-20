# OSRT Documentation
Implementation of [Optimal Sparse Regression Tree (OSRT)](https://arxiv.org/abs/2211.14980). This is implemented based on [Generalized Optimal Sparse Decision Tree framework (GOSDT)](https://github.com/ubc-systopia/gosdt-guesses). If you need classification trees, please use GOSDT.

![image](https://user-images.githubusercontent.com/60573138/189567116-0b719588-d670-4038-a242-2cc4be26816b.png)


# Table of Content
- [Installation](#installation)
- [Compilation](#compilation)
- [Configuration](#configuration)
- [Example](#example)
- [Repository Structure](#structure)
- [License](#license)
- [FAQs](#faqs)

---

# Installation

You may use the following commands to install OSRT along with its dependencies on macOS, Ubuntu and Windows.  
You need **Python 3.9 or later** to use the module `osrt` in your project.

```bash
pip3 install attrs packaging editables pandas sklearn sortedcontainers gmpy2 matplotlib
pip3 install osrt
```
You need to install `gmpy2==2.0.a1` if You are using Python 3.12

You can find a list of available wheels on [PyPI](https://pypi.org/project/osrt/).  
Please feel free to open an issue if you do not see your distribution offered.

# Compilation

Please refer to the [manual](doc/build.md) to build the C++ command line interface and the Python extension module and run the experiment with example datasets on your machine.

---

# Configuration

The configuration is a JSON object and has the following structure and default values:
```json
{ 
  "regularization": 0.05,
  "depth_budget": 0,
  "k_cluster": true,

  "metric": "L2",
  "weights": [],

  "time_limit": 0,
  "uncertainty_tolerance": 0.0,
  "upperbound": 0.0,
  "worker_limit": 1,
  "stack_limit": 0,
  "precision_limit": 0,
  "model_limit": 1,

  "verbose": false,
  "diagnostics": false,
  "balance": false,
  "look_ahead": true,

  "model": "",
  "timing": "",
  "trace": "",
  "tree": "",
  "profile": ""
}
```

## Key parameters

**regularization**
- Values: Decimal within range [0,1]
- Description: Used to penalize complexity. A complexity penalty is added to the risk in the following way.
  ```
  ComplexityPenalty = # Leaves x regularization
  ```
- Default: 0.05
- **Note: We highly recommend setting the regularization to a value larger than 1/num_samples. A small regularization could lead to a longer training time and possible overfitting.**

**depth_budget**
- Values: Integers >= 1
- Description: Used to set the maximum tree depth for solutions, counting a tree with just the root node as depth 1. 0 means unlimited.
- Default: 0

**k_cluster**
- Values: true or false
- Description: Enables the kmeans lower bound
- Default: true

**metric**
- Values: L1 or L2
- Description: The metric used in loss function. Mean squared error if L2, mean absolute error if L1.
- Default: L2

**weights**
- Values: Vector of real numbers
- Description: Weights assigned to each sample in training dataset. Empty vector means samples are unweighted. 
- Default: []
## More parameters
### Flag
**look_ahead**
- Values: true or false
- Description: Enables the one-step look-ahead bound implemented via scopes
- Default: true

**diagnostics**
- Values: true or false
- Description: Enables printing of diagnostic trace when an error is encountered to standard output
- Default: false

**verbose**
- Values: true or false
- Description: Enables printing of configuration, progress, and results to standard output
- Default: false




### Tuners

**uncertainty_tolerance**
- Values: Decimal within range [0,1]
- Description: Used to allow early termination of the algorithm. Any models produced as a result are guaranteed to score within the lowerbound and upperbound at the time of termination. However, the algorithm does not guarantee that the optimal model is within the produced model unless the uncertainty value has reached 0.
- Default: 0.0

**upperbound**
- Values: Decimal within range [0,1]
- Description: Used to limit the risk of model search space. This can be used to ensure that no models are produced if even the optimal model exceeds a desired maximum risk. This also accelerates learning if the upperbound is taken from the risk of a nearly optimal model.
- Special Cases: When set to 0, the bound is not activated.
- Default: 0.0

### Limits
**time_limit**
- Values: Decimal greater than or equal to 0
- Description: A time limit upon which the algorithm will terminate. If the time limit is reached, the algorithm will terminate with an error.
- Special Cases: When set to 0, no time limit is imposed.
- Default: 0

**model_limit**
- Values: Decimal greater than or equal to 0
- Description: The maximum number of models that will be extracted into the output.
- Special Cases: When set to 0, no output is produced.
- Default: 1

**precision_limit**
- Values: Decimal greater than or equal to 0
- Description: The maximum number of significant figures considered when converting ordinal features into binary features.
- Special Cases: When set to 0, no limit is imposed.
- Default: 0

**stack_limit**
- Values: Decimal greater than or equal to 0
- Description: The maximum number of bytes considered for use when allocating local buffers for worker threads.
- Special Cases: When set to 0, all local buffers will be allocated from the heap.
- Default: 0


**worker_limit**
- Values: Decimal greater than or equal to 1
- Description: The maximum number of threads allocated to executing th algorithm.
- Special Cases: When set to 0, a single thread is created for each core detected on the machine.
- Default: 1

### Files

**model**
- Values: string representing a path to a file.
- Description: The output models will be written to this file.
- Special Case: When set to empty string, no model will be stored.
- Default: Empty string

**profile**
- Values: string representing a path to a file.
- Description: Various analytics will be logged to this file.
- Special Case: When set to empty string, no analytics will be stored.
- Default: Empty string

**timing**
- Values: string representing a path to a file.
- Description: The training time will be appended to this file.
- Special Case: When set to empty string, no training time will be stored.
- Default: Empty string

**trace**
- Values: string representing a path to a directory.
- Description: snapshots used for trace visualization will be stored in this directory
- Special Case: When set to empty string, no snapshots are stored.
- Default: Empty string

**tree**
- Values: string representing a path to a directory.
- Description: snapshots used for trace-tree visualization will be stored in this directory
- Special Case: When set to empty string, no snapshots are stored.
- Default: Empty string

---

# Example

Example code to run GOSDT with threshold guessing, lower bound guessing, and depth limit. The example python file is available in [gosdt/example.py](/gosdt/example.py). A tutorial ipython notebook is available in [gosdt/tutorial.ipynb](/gosdt/tutorial.ipynb).

```
import pandas as pd
import numpy as np
import time
from model.osrt import OSRT

# read the dataset
# preprocess your data otherwise OSRT will binarize continuous feature using all threshold values.
df = pd.read_csv("experiments/datasets/airfoil/airfoil.csv")
X, y = df.iloc[:,:-1].values, df.iloc[:,-1].values
h = df.columns[:-1]
X = pd.DataFrame(X, columns=h)
X_train = X
y_train = pd.DataFrame(y)
print("X:", X.shape)
print("y:",y.shape)

# train OSRT model
config = {
    "regularization": 0.007,
    "depth_budget": 6,
    "model_limit": 100,

    "metric": "L2",
    "weights": [],

    "verbose": False,
    "diagnostics": True,
    }

model = OSRT(config)

model.fit(X_train, y_train)

print("evaluate the model, extracting tree and scores", flush=True)

# get the results
train_acc = model.score(X_train, y_train)
n_leaves = model.leaves()
n_nodes = model.nodes()
time = model.time

print("Model training time: {}".format(time))
print("Training score: {}".format(train_acc))
print("# of leaves: {}".format(n_leaves))
print(model.tree)
```

**Output**

```
X: (1503, 17)
y: (1503,)
osrt reported successful execution
training completed. 4.664 seconds.
bounds: [0.743839..0.743839] (0.000000) normalized loss=0.631839, iterations=46272
evaluate the model, extracting tree and scores
Model training time: 4.664000034332275
Training score: 30.060801844008466
# of leaves: 16
if feature_1_1 = 1 and feature_2_2 = 1 then:
    predicted class: 112.945831
    normalized loss penalty: 0.01
    complexity penalty: 0.007

else if feature_1_1 != 1 and feature_2_2 = 1 and feature_5_3 = 1 then:
    predicted class: 116.111771
    normalized loss penalty: 0.028
    complexity penalty: 0.007

else if feature_1_1 != 1 and feature_2_2 = 1 and feature_4_71.3 = 1 and feature_5_3 != 1 then:
    predicted class: 128.063248
    normalized loss penalty: 0.034
    complexity penalty: 0.007

else if feature_1_1 != 1 and feature_2_2 = 1 and feature_3_0.1016 = 1 and feature_4_71.3 != 1 and feature_5_3 != 1 then:
    predicted class: 120.686447
    normalized loss penalty: 0.037
    complexity penalty: 0.007

else if feature_1_1 != 1 and feature_2_2 = 1 and feature_3_0.1016 != 1 and feature_4_71.3 != 1 and feature_5_3 != 1 then:
    predicted class: 125.050087
    normalized loss penalty: 0.021
    complexity penalty: 0.007

else if feature_1_2 = 1 and feature_2_2 != 1 and feature_3_0.3048 = 1 then:
    predicted class: 109.278999
    normalized loss penalty: 0.0
    complexity penalty: 0.007

else if feature_1_2 != 1 and feature_1_3 = 1 and feature_2_2 != 1 and feature_3_0.3048 = 1 then:
    predicted class: 107.651497
    normalized loss penalty: 0.0
    complexity penalty: 0.007

else if feature_1_1 = 1 and feature_1_2 != 1 and feature_1_3 != 1 and feature_2_2 != 1 and feature_3_0.3048 = 1 then:
    predicted class: 113.869255
    normalized loss penalty: 0.003
    complexity penalty: 0.007

else if feature_1_1 != 1 and feature_1_2 != 1 and feature_1_3 != 1 and feature_2_2 != 1 and feature_3_0.3048 = 1 then:
    predicted class: 124.200935
    normalized loss penalty: 0.038
    complexity penalty: 0.007

else if feature_1_1 = 1 and feature_2_2 != 1 and feature_3_0.2286 = 1 and feature_3_0.3048 != 1 then:
    predicted class: 115.355225
    normalized loss penalty: 0.004
    complexity penalty: 0.007

else if feature_1_1 != 1 and feature_1_3 = 1 and feature_2_2 != 1 and feature_3_0.2286 = 1 and feature_3_0.3048 != 1 then:
    predicted class: 112.966003
    normalized loss penalty: 0.0
    complexity penalty: 0.007

else if feature_1_1 != 1 and feature_1_3 != 1 and feature_2_2 != 1 and feature_3_0.2286 = 1 and feature_3_0.3048 != 1 then:
    predicted class: 125.296906
    normalized loss penalty: 0.096
    complexity penalty: 0.007

else if feature_1_1 = 1 and feature_2_2 != 1 and feature_3_0.1524 = 1 and feature_3_0.2286 != 1 and feature_3_0.3048 != 1 then:
    predicted class: 116.648323
    normalized loss penalty: 0.009
    complexity penalty: 0.007

else if feature_1_1 != 1 and feature_2_2 != 1 and feature_3_0.1524 = 1 and feature_3_0.2286 != 1 and feature_3_0.3048 != 1 then:
    predicted class: 125.097855
    normalized loss penalty: 0.112
    complexity penalty: 0.007

else if feature_2_2 != 1 and feature_2_3 = 1 and feature_3_0.1524 != 1 and feature_3_0.2286 != 1 and feature_3_0.3048 != 1 then:
    predicted class: 122.649429
    normalized loss penalty: 0.067
    complexity penalty: 0.007

else if feature_2_2 != 1 and feature_2_3 != 1 and feature_3_0.1524 != 1 and feature_3_0.2286 != 1 and feature_3_0.3048 != 1 then:
    predicted class: 128.906433
    normalized loss penalty: 0.173
    complexity penalty: 0.007
```


# Structure

This repository contains the following directories and files:
- **.github**: Configurations for GitHub action runners.
- **doc**: Documentation
- **experiments**: Datasets and their configurations to run experiments
- **osrt**: Python implementation and wrappers around C++ implementation
- **include**: Required 3rd-party header-only libraries
- **log**: Log files
- **src**: Source files for C++ implementation and Python binding
- **test**: Source files for unit tests
- **build.py**: Python script that builds the project automatically
- **CMakeLists.txt**: Configuration file for the CMake build system
- **pyproject.toml**: Configuration file for the SciKit build system
- **setup.py**: Python script that builds the wheel file

---

# Structure

This repository contains the following directories and files:
- **.github**: Configurations for GitHub action runners.
- **doc**: Documentation
- **experiments**: Datasets and their configurations to run experiments
- **osrt**: Jupyter notebook, Python implementation and wrappers around C++ implementation
- **include**: Required 3rd-party header-only libraries
- **log**: Log files
- **src**: Source files for C++ implementation and Python binding
- **test**: Source files for unit tests
- **build.py**: Python script that builds the project automatically
- **CMakeLists.txt**: Configuration file for the CMake build system
- **pyproject.toml**: Configuration file for the SciKit build system
- **setup.py**: Python script that builds the wheel file

---

# FAQs

If you run into any issues when running OSRT, consult the [**FAQs**](/doc/faqs.md) first.

---

# License

This software is licensed under a 3-clause BSD license (see the LICENSE file for details).

---