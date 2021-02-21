import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import spartan as st
from .metrics import metrics, MetricCollection

class BipartiteFramework:
    '''
        Framework for evaluating bipartite graph models found in bipartiteModels/models.py
    '''
    def __init__(self, model, metrics=metrics):
        self.model = model
        self.metrics = MetricCollection(metrics)
        self.G = None

    def fit(self, G):
        self.G = G
        self.model.fit(G)

        node_labels = self.G.nodes()
        edge_labels = self.G.edges()

        pred_users, pred_products = self.identify()
        # pred df for users
        y_pred = [1 if int(node) in pred_users or int(node) in pred_products else 0 for node in node_labels]
        y_true = [node_labels[node]['label'] if 'label' in node_labels[node] else None for node in node_labels]
        node_df = pd.DataFrame({'node':node_labels, 'y_pred':y_pred, 'y_true':y_true})

        # pred df for edges
        y_pred = [1 if int(u) in pred_users and int(v) in pred_products else 0 for u,v in edge_labels]
        y_true = [edge_labels[edge]['label'] if 'label' in edge_labels[edge] else None for edge in edge_labels]
        edge_df = pd.DataFrame({'edge':edge_labels, 'y_pred':y_pred, 'y_true':y_true})

        return node_df, edge_df
    
    def identify(self):
        # Identify anomalous nodes
        if self.G:
            return self.model.predict()
        raise AssertionError('No graph found.')

    def score(self,y,y_pred):
        labelled_map = ~y.isnull()
        if labelled_map.sum() == 0:
            print('No valid labels')
            return None
        tmp_y, tmp_y_pred = y[labelled_map], y_pred[labelled_map]
        scores = self.metrics.compute(tmp_y,tmp_y_pred)
        return scores