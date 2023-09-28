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


