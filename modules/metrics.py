from sklearn.metrics import roc_auc_score, recall_score, f1_score, log_loss
from collections import defaultdict

def false_alarm_rate(y_true, y_pred):
    threshold = 0.5
    FP, N = 0, 0
    for true, pred in zip(y_true, y_pred):
        if true == 0:
            N += 1
            if pred > threshold:
                FP += 1
    if N == 0:
        return 0
    return float(FP/N)

def recall(y_true, y_pred):
    y_pred_binary = [1 if x > 0.5 else 0 for x in y_pred]
    return recall_score(y_true, y_pred_binary)

def f1(y_true, y_pred):
    y_pred_binary = [1 if x > 0.5 else 0 for x in y_pred]
    return f1_score(y_true, y_pred_binary)
    
metrics = {
    'ROC-AUC': roc_auc_score,
    'Detection-Rate': recall,
    'False-Alarm': false_alarm_rate,
    'F1-Score': f1,
    'Log-Likelihood': log_loss
}

class MetricCollection:
    """
        Class to compute a collection of metrics.
    """
    def __init__(self, metrics=metrics):
        """
            Initialise a dictionary of metrics to compute.
            Metrics of the form (name,metrics_functions).
            Compute metrics from a list of `y_true` and 
            `y_pred` labels.
        """
        self._metrics = metrics
        self._store = defaultdict(list)
                
    def compute(self, y_true, y_pred):
        """
            Compute each metric from a list of 
            `y_true` and `y_pred` labels.
        """
        results = {}
        for metric, func in self._metrics.items():
            val = func(y_true,y_pred)
            results[metric] = val
            self._store[metric].append(val)
        return results
    
    def clear(self):
        """
            Clear stored metric computations.
        """
        self._store = defaultdict(list)
        
    def get_computes(self):
        return self._store 
    
