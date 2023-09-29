import json
import pandas as pd
import time
from numpy import array

import osrt.libosrt as osrt  # Import the GOSDT extension
# from osrt.model.encoder import Encoder
# from osrt.model.imbalance.osdt_imb_v9 import bbound, predict  # Import the special objective implementation
from osrt.model.tree_regressor import TreeRegressor  # Import the tree classification model


class OSRT:
    def __init__(self, configuration={}):
        self.configuration = configuration
        self.time = 0.0
        # self.stime = 0.0
        # self.utime = 0.0
        # self.maxmem = 0
        # self.numswap = 0
        # self.numctxtswitch = 0
        self.iterations = 0
        self.size = 0
        self.tree = None
        self.encoder = None
        self.lb = 0
        self.ub = 0
        self.timeout = False
        self.reported_loss = 0

    def load(self, path):
        """
        Parameters
        ---
        path : string
            path to a JSON file representing a model
        """
        with open(path, 'r') as model_source:
            result = model_source.read()
        result = json.loads(result)
        self.tree = TreeRegressor(result[0])

    def __train__(self, X, y):
        """
        Parameters
        ---
        X : matrix-like, shape = [n_samples by m_features]
            matrix containing the training samples and features
        y : array-like, shape = [n_samples by 1]
            column containing the correct label for each sample in X
        Modifies
        ---
        trains a model using the GOSDT native extension
        """
        (n, m) = X.shape
        dataset = X.copy()
        dataset.insert(m, "class", y)  # It is expected that the last column is the label column

        osrt.configure(json.dumps(self.configuration, separators=(',', ':')))
        result = osrt.fit(dataset.to_csv(index=False))  # Perform extension call to train the model

        self.time = osrt.time()  # Record the training time

        if osrt.status() == 0:
            print("osrt reported successful execution")
            self.timeout = False
        elif osrt.status() == 2:
            print("osrt reported possible timeout.")
            self.timeout = True
            self.time = -1
            # self.stime = -1
            # self.utime = -1
        else:
            print('----------------------------------------------')
            print(result)
            print('----------------------------------------------')
            raise Exception("Error: OSRT encountered an error while training")

        result = json.loads(result)  # Deserialize resu

        self.tree = TreeRegressor(result[0])  # Parse the first result into model
        self.iterations = osrt.iterations()  # Record the number of iterations
        self.size = osrt.size()  # Record the graph size required

        # self.maxmem = osrt.maxmem()
        # self.numswap = osrt.numswap()
        # self.numctxtswitch = osrt.numctxtswitch()

        self.lb = osrt.lower_bound()  # Record reported global lower bound of algorithm
        self.ub = osrt.upper_bound()  # Record reported global upper bound of algorithm
        self.reported_loss = osrt.model_loss()  # Record reported training loss of returned tree

        print("training completed. {:.3f} seconds.".format(self.time))
        print("bounds: [{:.6f}..{:.6f}] ({:.6f}) normalized loss={:.6f}, iterations={}".format(self.lb, self.ub, self.ub - self.lb,
                                                                                    self.reported_loss,
                                                                                    self.iterations))

    def fit(self, X, y):
        """
        Parameters
        ---
        X : matrix-like, shape = [n_samples by m_features]
            matrix containing the training samples and features
        y : array-like, shape = [n_samples by 1]
            column containing the correct label for each sample in X
        Modifies
        ---
        trains the model so that this model instance is ready for prediction
        """

        self.__train__(X, y)

        return self

    def predict(self, X):
        """
        Parameters
        ---
        X : matrix-like, shape = [n_samples by m_features]
            a matrix where each row is a sample to be predicted and each column is a feature to be used for prediction
        Returns
        ---
        array-like, shape = [n_sampels by 1] : a column where each element is the prediction associated with each row
        """
        if self.tree is None:
            raise Exception("Error: Model not yet trained")
        return self.tree.predict(X)

    def score(self, X, y, weight=None):
        """
        Parameters
        ---
        X : matrix-like, shape = [n_samples by m_features]
            an n-by-m matrix of sample and their features
        y : array-like, shape = [n_samples by 1]
            an n-by-1 column of labels associated with each sample
        weight : real number
            an n-by-1 column of weights to apply to each sample's misclassification
        Returns
        ---
        real number : the accuracy produced by applying this model overthe given dataset, with optionals for weighted accuracy
        """
        if self.tree is None:
            raise Exception("Error: Model not yet trained")
        return self.tree.score(X, y, weight=weight)

    def __len__(self):
        """
        Returns
        ---
        natural number : The number of terminal nodes present in this tree
        """
        if self.tree is None:
            raise Exception("Error: Model not yet trained")
        return len(self.tree)

    def leaves(self):
        """
        Returns
        ---
        natural number : The number of terminal nodes present in this tree
        """
        if self.tree is None:
            raise Exception("Error: Model not yet trained")
        return self.tree.leaves()

    def nodes(self):
        """
        Returns
        ---
        natural number : The number of nodes present in this tree
        """
        if self.tree is None:
            raise Exception("Error: Model not yet trained")
        return self.tree.nodes()

    def max_depth(self):
        """
        Returns
        ---
        natural number : the length of the longest decision path in this tree. A single-node tree will return 1.
        """
        if self.tree is None:
            raise Exception("Error: Model not yet trained")
        return self.tree.maximum_depth()

    def latex(self):
        """
        Note
        ---
        This method doesn't work well for label headers that contain underscores due to underscore being a reserved character in LaTeX
        Returns
        ---
        string : A LaTeX string representing the model
        """
        if self.tree is None:
            raise Exception("Error: Model not yet trained")
        return self.tree.latex()

    def json(self):
        """
        Returns
        ---
        string : A JSON string representing the model
        """
        if self.tree is None:
            raise Exception("Error: Model not yet trained")
        return self.tree.json()
