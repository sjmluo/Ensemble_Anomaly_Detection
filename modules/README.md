# Graph Framework 

The following graph frameworks are for `static` and `bipartite` graphs and take a `networkx graph object` as input.
These graphs can be labelled or unlabelled. 
To compute scores/metrics for a particular method, labels are required.

In order to label a node/edge, the appropriate node/edge in the `networkx graph object` must have a **label** attribute that is either **0** (normal) or **1** (anomalous). 

Additionally, node/edge predictions based off the anomaly detection methods assign either **0** (normal) or **1** (anomalous) to the node/edge.

For examples of usage, refer to the [notebooks](../notebooks).



# Metrics

`MetricCollection` can be used for benchmarking models.

## Usage

```python
    from Metrics import MetricCollection
    from sklearn.metrics import roc_auc_score, recall_score

    metrics = {'ROC-AUC': roc_auc_score, 'Detection-Rate': recall_score}
    m = MetricCollection(metrics)
    y_pred1, y_true1 = [0.9,0.3,...], [1,0,...]
    y_pred2, y_true2 = [0.8,0.3,...], [1,0,...]
    ## Compute metric for a pair of predictions and true labels
    m.compute(y_true, y_pred)  # {'ROC-AUC': 0.8, 'Detection-Rate': 0.3}
    m.compute(y_true, y_pred)  # {'ROC-AUC': 0.6, 'Detection-Rate': 0.2}
    ## Retrieve stored results
    m.get_computes() # {'ROC-AUC': [0.8,0.6], 'Detection-Rate': [0.3,0.2]}
    ## Clear stored results
    m.clear()
    m.get_computes() # {'ROC-AUC': [], 'Detection-Rate': []}

```

### Default Metrics 

```python
    m = MetricCollection() 
    m.get_computes() # { 'ROC-AUC': [], 'Detection-Rate': [], 'False-Alarm': [], 'F1-Score': [], 'Log-Likelihood': [] }
```
