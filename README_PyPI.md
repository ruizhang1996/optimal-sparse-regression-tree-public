# OSRT Documentation
Implementation of [Optimal Sparse Regression Tree (OSRT)](https://arxiv.org/abs/2211.14980). This is implemented based on [Generalized Optimal Sparse Decision Tree framework (GOSDT)](https://github.com/ubc-systopia/gosdt-guesses). If you need classification trees, please use GOSDT.

![image](https://user-images.githubusercontent.com/60573138/189567116-0b719588-d670-4038-a242-2cc4be26816b.png)


# Table of Content
- [Installation](#installation)
- [Configuration](#configuration)
- [Example](#example)
- [License](#license)
- [FAQs](#faqs)

---
# Installation

You may use the following commands to install GOSDT along with its dependencies on macOS, Ubuntu and Windows.  
You need **Python 3.7 or later** to use the module `gosdt` in your project.

```bash
pip3 install attrs packaging editables pandas scikit-learn sortedcontainers gmpy2 matplotlib
pip3 install osrt
```

# Configuration

The configuration is a JSON object and has the following structure and default values:
```json
{ 
  "regularization": 0.05,
  "depth_budget": 0,
  "k_cluster": false,
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
  "similar_support": false,
  "cancellation": true,
  "continuous_feature_exchange": false,
  "feature_exchange": false,
  "feature_transform": true,
  "rule_list": false,
  "non_binary": false,
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

**time_limit**
- Values: Decimal greater than or equal to 0
- Description: A time limit upon which the algorithm will terminate. If the time limit is reached, the algorithm will terminate with an error.
- Special Cases: When set to 0, no time limit is imposed.
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

**balance**
- Values: true or false
- Description: Enables overriding the sample importance by equalizing the importance of each present class
- Default: false

**cancellation**
- Values: true or false
- Description: Enables propagate up the dependency graph of task cancellations
- Default: true

**look_ahead**
- Values: true or false
- Description: Enables the one-step look-ahead bound implemented via scopes
- Default: true

**similar_support**
- Values: true or false
- Description: Enables the similar support bound implemented via the distance index
- Default: true

**feature_exchange**
- Values: true or false
- Description: Enables pruning of pairs of features using subset comparison
- Default: false

**continuous_feature_exchange**
- Values: true or false
- Description: Enables pruning of pairs continuous of feature thresholds using subset comparison
- Default: false

**feature_transform**
- Values: true or false
- Description: Enables the equivalence discovery through simple feature transformations
- Default: true

**rule_list**
- Values: true or false
- Description: Enables rule-list constraints on models
- Default: false

**non_binary**
- Values: true or false
- Description: Enables non-binary encoding
- Default: false

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
import pathlib
from sklearn.ensemble import GradientBoostingRegressor
from model.threshold_guess import compute_thresholds
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


# guess thresholds (OPTIONAL) uncomment following lines if you want to speed up optimization
# NOTE: You should also evaluate accuracy on guessed data if you choose to guess thresholds
# GBRT parameters for threshold guesses
# n_est = 40
# max_depth = 1
# X_train, thresholds, header, threshold_guess_time = compute_thresholds(X, y, n_est, max_depth)


# train OSRT model
config = {
    "similar_support": False,
    "feature_exchange": False,
    "continuous_feature_exchange": False,
    "regularization": 0.007,
    "depth_budget": 6,
    "model_limit": 1,
    "time_limit": 0,
    "similar_support": False,
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
training completed. 3.341 seconds.
bounds: [0.744063..0.744063] (0.000000) normalized loss=0.632063, iterations=45664
evaluate the model, extracting tree and scores
Model training time: 3.3410000801086426
Training score: 30.06080184358605
# of leaves: 16
if feature_1_1 = 1 and feature_2_2 = 1 then:
    predicted class: 112.945833
    normalized loss penalty: 0.01
    complexity penalty: 0.007

else if feature_1_1 != 1 and feature_2_2 = 1 and feature_5_3 = 1 then:
    predicted class: 116.111778
    normalized loss penalty: 0.028
    complexity penalty: 0.007

else if feature_1_1 != 1 and feature_2_2 = 1 and feature_4_71.3 = 1 and feature_5_3 != 1 then:
    predicted class: 128.063236
    normalized loss penalty: 0.034
    complexity penalty: 0.007

else if feature_1_1 != 1 and feature_2_2 = 1 and feature_3_0.1016 = 1 and feature_4_71.3 != 1 and feature_5_3 != 1 then:
    predicted class: 120.686444
    normalized loss penalty: 0.037
    complexity penalty: 0.007

else if feature_1_1 != 1 and feature_2_2 = 1 and feature_3_0.1016 != 1 and feature_4_71.3 != 1 and feature_5_3 != 1 then:
    predicted class: 125.05011
    normalized loss penalty: 0.021
    complexity penalty: 0.007

else if feature_1_2 = 1 and feature_2_2 != 1 and feature_3_0.3048 = 1 then:
    predicted class: 109.279
    normalized loss penalty: 0.0
    complexity penalty: 0.007

else if feature_1_1 = 1 and feature_1_2 != 1 and feature_2_2 != 1 and feature_3_0.3048 = 1 then:
    predicted class: 113.869267
    normalized loss penalty: 0.003
    complexity penalty: 0.007

else if feature_1_1 != 1 and feature_1_2 != 1 and feature_1_3 = 1 and feature_2_2 != 1 and feature_3_0.3048 = 1 then:
    predicted class: 107.6515
    normalized loss penalty: 0.0
    complexity penalty: 0.007

else if feature_1_1 != 1 and feature_1_2 != 1 and feature_1_3 != 1 and feature_2_2 != 1 and feature_3_0.3048 = 1 then:
    predicted class: 124.20096
    normalized loss penalty: 0.038
    complexity penalty: 0.007

else if feature_1_1 = 1 and feature_2_2 != 1 and feature_3_0.2286 = 1 and feature_3_0.3048 != 1 then:
    predicted class: 115.355214
    normalized loss penalty: 0.004
    complexity penalty: 0.007

else if feature_1_1 != 1 and feature_1_3 = 1 and feature_2_2 != 1 and feature_3_0.2286 = 1 and feature_3_0.3048 != 1 then:
    predicted class: 112.966
    normalized loss penalty: 0.0
    complexity penalty: 0.007

else if feature_1_1 != 1 and feature_1_3 != 1 and feature_2_2 != 1 and feature_3_0.2286 = 1 and feature_3_0.3048 != 1 then:
    predicted class: 125.296885
    normalized loss penalty: 0.097
    complexity penalty: 0.007

else if feature_1_1 = 1 and feature_2_2 != 1 and feature_3_0.1524 = 1 and feature_3_0.2286 != 1 and feature_3_0.3048 != 1 then:
    predicted class: 116.648313
    normalized loss penalty: 0.009
    complexity penalty: 0.007

else if feature_1_1 != 1 and feature_2_2 != 1 and feature_3_0.1524 = 1 and feature_3_0.2286 != 1 and feature_3_0.3048 != 1 then:
    predicted class: 125.097889
    normalized loss penalty: 0.112
    complexity penalty: 0.007

else if feature_2_2 != 1 and feature_2_3 = 1 and feature_3_0.1524 != 1 and feature_3_0.2286 != 1 and feature_3_0.3048 != 1 then:
    predicted class: 122.649413
    normalized loss penalty: 0.067
    complexity penalty: 0.007

else if feature_2_2 != 1 and feature_2_3 != 1 and feature_3_0.1524 != 1 and feature_3_0.2286 != 1 and feature_3_0.3048 != 1 then:
    predicted class: 128.906417
    normalized loss penalty: 0.173
    complexity penalty: 0.007
```

# FAQs

If you run into any issues when running GOSDT, consult the [**FAQs**](/doc/faqs.md) first.

---

# License

This software is licensed under a 3-clause BSD license (see the LICENSE file for details).

---
