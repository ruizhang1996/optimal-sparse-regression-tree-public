import pandas as pd
import numpy as np
import time, pathlib
from model.osrt import OSRT
from sklearn.ensemble import GradientBoostingRegressor
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

clf = GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42)
clf.fit(X_train, y_train.values.flatten())
warm_labels = clf.predict(X_train)
labelsdir = pathlib.Path('/tmp/warm_lb_labels')
labelsdir.mkdir(exist_ok=True, parents=True)
labelpath = labelsdir / 'warm_label.tmp'
labelpath = str(labelpath)
pd.DataFrame(warm_labels, columns=["class_labels"]).to_csv(labelpath, header="class_labels",index=None)


# train OSRT model
config = {
    "regularization": 0.007,
    "depth_budget": 6,
    "model_limit": 100,
    "minimum_captured_points": 1,
    "reference_LB": True,
    "path_to_labels": labelpath,
    
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


