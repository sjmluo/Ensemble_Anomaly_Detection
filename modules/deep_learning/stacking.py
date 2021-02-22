from modules.evaluation import EvaluationFramework
from modules.metrics import metrics

from pyod.models.iforest import IForest
from pyod.models.lof import LOF
from pyod.models.knn import KNN
from pyod.models.pca import PCA
from pyod.models.ocsvm import OCSVM
from pyod.utils.utility import standardizer

import pandas as pd
import re
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

from os import listdir
from os.path import isfile, join
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from modules.deep_learning.VAE import ReconstructionVAE, VAErcp, VAEvampprior
from modules.deep_learning.VAE import CompileHelper

class MLP(tf.keras.Model):
    def __init__(self, sizes, n_inps = None):
        super(MLP, self).__init__()
        self.pre = None
        if n_inps is not None:
            self.pre = [tf.keras.layers.Dense(16, activation = 'relu') for x in range(n_inps)]
        self.mlp = [tf.keras.layers.Dense(s, activation = 'relu') for s in sizes[:-1]]
        self.mlp.append(tf.keras.layers.Dense(sizes[-1], activation = 'sigmoid'))

    def call(self, x):
        if self.pre is not None:
            x = [l(tf.expand_dims(x[:,i], -1)) for i, l in enumerate(self.pre)]
            x = tf.concat(x, -1)
            assert(x.shape[-1] == len(self.pre), f'{x.shape} vs {len(self.pre)}')
        for l in self.mlp:
            x = l(x)
        return x

class WCEHelper:
    def __init__(self, weights, conf = 0.5):
        self.weights = tf.constant(weights, dtype = tf.float32)
        self.conf = conf
    
    def __call__(self, y_true, y_pred):
        return weightedce(y_true, y_pred, self.weights, self.conf)
    
    def __eq__(self, other):
        return tf.math.reduce_all(self.weights == other.weights) and self.conf == other.conf

    @property
    def __name__(self):
        return f"WCEHelper{self.weights}conf{self.conf}".replace(" ", "").replace("\n", "").replace("[", "").replace("]", "")

def weightedce(y_true, y_pred, weights = [[1,1],[1,1]], conf = 0.5):
    """
    Weighted binary crossentropy, weights is a 2D array corresponding to each type of prediction
    |      |Class 0|Class 1|
    |------|-------|-------|
    |Pred 0| [0][0]| [0][1]|
    |Pred 1| [1][0]| [1][1]|
    Each entry corresponds to index of weights argument
    Prediction is categories as class 0 for a prediction of < conf
    """
    conf = tf.constant(conf)

    predclass = tf.where(tf.less(tf.squeeze(y_pred), conf), tf.constant(0,dtype=tf.int32), tf.constant(1,dtype=tf.int32))

    if len(predclass.shape) == 0:
        predclass = tf.expand_dims(predclass, 0)
        
    predclass = tf.gather(weights, predclass)
    mask = tf.where(tf.math.equal(tf.squeeze(y_true), tf.constant(0)), predclass[:,0], predclass[:,1])

    return tf.math.multiply(tf.keras.losses.binary_crossentropy(y_true, y_pred), mask)

class Stacking:
    def __init__(self, models = None):
        self.models = models
        if self.models == None: self.models = [ReconstructionVAE(),
                                                VAErcp(),
                                                VAEvampprior(),
                                                IForest(),
                                                KNN(),
                                                LOF(),
                                                PCA(),
                                                OCSVM()]
        self.stack = None
        self.verbose = 0
        self.epochs = 1500
    
    def fit(self, X, y):
        y = np.array(y, dtype = 'int32')
        sizes = [256, 64, 16, 1]
        self.stack = MLP(sizes)
        X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=1,train_size=0.6)
        for model in self.models:
            if isinstance(model, tf.keras.Model):
                model.addEpochs(self.epochs)
                model.addVerbose(self.verbose)
            model.fit(X_train)
            if isinstance(model, tf.keras.Model):
                model.trainable = False
                model.addEpochs(self.epochs)
        X_test = [model.predict_proba(X_test)[:,1] for model in self.models]
        X_test = np.stack(X_test, -1)

        ratio = np.ceil((y == 0).sum()/(y == 1).sum())
        losses = [WCEHelper(weights = [[1,ratio],[1,1]])]
        loss_weights = [1]
        ch = CompileHelper(losses, loss_weights)
        ch(self.stack)
        self.stack.fit(X_test, y_test, callbacks = self.callbacks(), 
        verbose = self.verbose, epochs = self.epochs)

    def predict(self, x):
        y_pred = self.predict_proba(x)[:,1]
        t_val = y_pred < 0.5
        return np.where(y_pred < t_val, 0, 1)

    def predict_proba(self, x):
        X_test = [model.predict_proba(x)[:,1] for model in self.models]
        X_test = np.stack(X_test, -1)
        y_pred = self.stack(X_test)
        class0 = 1-y_pred
        return np.concatenate([class0, y_pred], 1)

    def callbacks(self):
        plateau = tf.keras.callbacks.ReduceLROnPlateau(verbose=1, patience=10, monitor='loss')
        earlystop = tf.keras.callbacks.EarlyStopping(
        monitor='loss', min_delta=10e-7, patience=25, verbose=2,
        mode='auto', baseline=None, restore_best_weights=True
        )

        return [plateau, earlystop]

def run():
    methods = {
        'stacking': Stacking()
    }
    scoring = list(metrics.keys())
    columns = ["dataset", "method",] + scoring
    df = pd.DataFrame(
            columns=columns)
    df.to_csv("results/results.csv",index=False)

    dataset_folder = "datasets/"
    onlyfiles = [join(dataset_folder, f) for f in listdir(dataset_folder) if isfile(join(dataset_folder, f))]

    index = 0
    for file in onlyfiles[:11]:
        mat = loadmat(file)
        X = mat['X']
        y = mat['y'].ravel()

        X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=1,train_size=0.6)

        X_train_norm, X_test_norm = standardizer(X_train, X_test)
        for key,model in methods.items():
            eva = EvaluationFramework(model)

            if isinstance(model, Ensemble):
                eva.supervised_fit(X_train_norm, y_train)



if __name__ == '__main__':
    run()
