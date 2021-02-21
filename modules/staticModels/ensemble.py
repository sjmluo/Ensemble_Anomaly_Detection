from collections import defaultdict
import numpy as np

'''
    Module for static graph ensembling methods
'''

class EnsembleStaticGraphs:
    def __init__(self, model, parameter_list, method='avg'):
        self.model_func = model
        self.parameter_list = parameter_list
        
        param_dictlist = map(dict, zip(*[[(k, v) for v in value] for k, value in parameter_list.items()]))
        self.models = [model(**params_dict) for params_dict in param_dictlist]
        self.num_models = len(self.models)
        self.method = method

    def fit(self, G):
        for i in range(len(self.models)):
            self.models[i].fit(G) 

    def predict(self):
        nodes_dict = defaultdict(int)
        for i in range(self.num_models):
            node_preds = self.models[i].predict()
            for node in node_preds:
                nodes_dict[node] += 1
        if self.method == 'max':
            outliers = list(nodes_dict.keys())
        else:
            outliers = list(filter(lambda x: x >= self.num_models/2, nodes_dict))
        return outliers

    def predict__proba(self):
        nodes_dict = defaultdict(int)
        for i in range(self.num_models):
            node_preds = self.models[i].predict()
            for node in node_preds:
                nodes_dict[node] += 1
        if self.method == 'max':
            outliers = {x:1 for x in nodes_dict}
        else:
            outliers = {x:nodes_dict[x]/self.num_models for x in nodes_dict}
        return outliers