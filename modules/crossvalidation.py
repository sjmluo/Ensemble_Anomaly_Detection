from .metrics import MetricCollection, metrics
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm
import itertools
import numpy as np
import pandas as pd

class HyperparameterTune:
    def __init__(self, detector, n_splits=5, **parameters):
        self.detector = detector
        self.n_splits = n_splits 
        self.params = parameters
        self.param_keys = list(parameters.keys())
    
    def evaluate(self, X, y):
        self.scores = pd.DataFrame(columns=['AUC']+list(self.params.keys()))
        tmp_n_splits = min(self.n_splits, int(sum(y)))
        kf = StratifiedKFold(n_splits=tmp_n_splits)
        fold_idxs = list(kf.split(X,y))
        param_combs = itertools.product(*self.params.values())
        
        for param_comb in param_combs:
            input_params = dict(zip(self.param_keys,param_comb))
            score = []
            for train_idx, test_idx in fold_idxs:
                X_train = X[train_idx]
                X_test, y_test = X[test_idx], y[test_idx]
                model = self.detector(**input_params)
                model.fit(X_train)
                y_pred = model.predict_proba(X_test)
                score.append(roc_auc_score(y_test, y_pred[:,1]))
            self.scores = self.scores.append({'AUC':np.mean(score), **input_params}, ignore_index=True)
        # Return parameters dict of first param combination with largest AUC
        return self.scores.loc[self.scores['AUC']==self.scores['AUC'].max()].drop(['AUC'], axis=1).iloc[0].to_dict()

    def fit(self, X,y, metrics=metrics, **parameters):
        # Fit model for particular parameter and retrieve score
        m = MetricCollection(metrics)
        model = self.detector(**parameters)
        model.fit(X)
        y_pred = model.predict_proba(X)
        return m.compute(y, y_pred)

class CrossvalidationFramework:
    def __init__(self, detector, n_splits=5, **parameters):
        self.detector = detector
        self.n_splits = n_splits 
        self.params = parameters
        self.param_keys = list(parameters.keys())
        
    def evaluate(self, X, y, metrics=metrics):
        # Determines scores based off metrics using cross-validation
        m = MetricCollection(metrics)
        metric_names = list(metrics.keys())
        kf = StratifiedKFold(n_splits=min(self.n_splits, int(sum(y))))
        scores = pd.DataFrame(columns=metric_names)
        
        for train_idx, test_idx in tqdm(kf.split(X,y)):
            X_train, y_train = X[train_idx], y[train_idx]
            X_test, y_test = X[test_idx], y[test_idx]
            # Nest Cross-Validation
            h = HyperparameterTune(self.detector, self.n_splits, **self.params)
            tuned_params = h.evaluate(X_train, y_train)
            # Fit Tuned Model and Append Score
            model = self.detector(**tuned_params)
            model.fit(X_train)
            
            y_pred = model.predict_proba(X_test)[:,1]
            scores = scores.append(m.compute(y_test, y_pred), ignore_index=True)
        return scores.mean().to_dict(), scores.std().to_dict()