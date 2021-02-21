import numpy as np

'''
    Module for ensembling methods
'''

class EnsembleModelParameters:
    """
        Ensembling by averaging the predictions of a model 
        fitted with different parameters.
    ...

    Attributes
    ----------
    model_func : pointcloud model function
    model: list of models
    parameter_list : dictionary of the form (parameter_name, list of parameters)
    method (max, avg): max/avg takes the maximum/average anomaly score as the prediction
    """
    def __init__(self, model, parameter_list, method='avg'):
        self.model_func = model
        self.parameter_list = parameter_list
        
        param_dictlist = map(dict, zip(*[[(k, v) for v in value] for k, value in parameter_list.items()]))
        self.models = [model(**params_dict) for params_dict in param_dictlist]
        self.num_models = len(self.models)
        self.method = method
        
    def fit(self, x):
        for i in range(len(self.models)):
            self.models[i].fit(x)
    
    def predict(self, x):
        y_preds = np.zeros(x.shape[0])
        for i in range(self.num_models):
            if self.method == 'max':
                y_preds = np.maximum(y_preds, self.models[i].predict(x))
            else:
                y_preds += self.models[i].predict(x)
            
        y_preds = np.array([1 if i >= self.num_models/2 else 0 for i in y_preds])
        return y_preds
    
    def predict_proba(self, x):
        y_preds = np.zeros(x.shape[0])
        for i in range(self.num_models):
            if self.method == 'max':
                y_preds = np.maximum(y_preds, self.models[i].predict_proba(x))
            else:
                y_preds += self.models[i].predict_proba(x)/self.num_models
        return y_preds