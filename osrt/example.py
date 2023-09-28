import pandas as pd
import numpy as np
import time
import pathlib
from sklearn.ensemble import GradientBoostingRegressor
from model.threshold_guess import compute_thresholds
from model.osrt import OSRT

# read the dataset
df = pd.read_csv("experiments/datasets/airfoil/airfoil.csv")
X, y = df.iloc[:,:-1].values, df.iloc[:,-1].values
h = df.columns[:-1]

# GBDT parameters for threshold and lower bound guesses
n_est = 40
max_depth = 1

# guess thresholds
X = pd.DataFrame(X, columns=h)
print("X:", X.shape)
print("y:",y.shape)
# X_train, thresholds, header, threshold_guess_time = compute_thresholds(X, y, n_est, max_depth)
y_train = pd.DataFrame(y)

# guess lower bound
# start_time = time.perf_counter()
# clf = GradientBoostingRegressor(n_estimators=n_est, max_depth=max_depth, random_state=42)
# clf.fit(X_train, y_train.values.flatten())
# warm_labels = clf.predict(X_train)

# elapsed_time = time.perf_counter() - start_time
#
# lb_time = elapsed_time

# save the labels as a tmp file and return the path to it.
# labelsdir = pathlib.Path('/tmp/warm_lb_labels')
# labelsdir.mkdir(exist_ok=True, parents=True)

# labelpath = labelsdir / 'warm_label.tmp'
# labelpath = str(labelpath)
# pd.DataFrame(warm_labels, columns=["class_labels"]).to_csv(labelpath, header="class_labels",index=None)


# train GOSDT model
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
    "verbose": True,
    "diagnostics": True,
        }

model = OSRT(config)

model.fit(X, y_train)

print("evaluate the model, extracting tree and scores", flush=True)

# get the results
train_acc = model.score(X, y_train)
n_leaves = model.leaves()
n_nodes = model.nodes()
# time = model.utime

# print("Model training time: {}".format(time))
print("Training accuracy: {}".format(train_acc))
print("# of leaves: {}".format(n_leaves))
print(model.tree)


