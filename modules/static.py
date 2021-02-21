import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import spartan as st
from .metrics import metrics, MetricCollection

class StaticFramework:
    '''
        Framework for evaluating static graph models found in staticModels/models.py
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

        pred_nodes = self.identify()
        # pred df for users
        y_pred = [1 if int(node) in pred_nodes else 0 for node in node_labels]
        y_true = [node_labels[node]['label'] if 'label' in node_labels[node] else None for node in node_labels]
        node_df = pd.DataFrame({'node':node_labels, 'y_pred':y_pred, 'y_true':y_true})

        # pred df for edges
        y_pred = [1 if int(u) in pred_nodes and int(v) in pred_nodes else 0 for u,v in edge_labels]
        y_true = [edge_labels[edge]['label'] if 'label' in edge_labels[edge] else None for edge in edge_labels]
        edge_df = pd.DataFrame({'edge':edge_labels, 'y_pred':y_pred, 'y_true':y_true})

        return node_df, edge_df

    def identify(self):
         # Identify anomalous nodes
         # Requires the model to be fit before this function can run
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

    def degreedensity(self,G,nodes,labels, count=False):
        # Compare degree density of anomalous nodes (y_nodes) vs. normal nodes (exclude y_nodes) 
        degreesDict = G.degree()
        degrees = [degreesDict[node]  for node in nodes]
        df = pd.DataFrame({'Node':nodes, 'Label':['Anomaly' if x == 1 else 'Normal' for x in labels],'Degree': degrees})
        num_anom = df.loc[df['Label'] == 'Anomaly'].shape[0]
        if count or num_anom < 2:
            plot = sns.displot(df, x='Degree',hue='Label')
        else:
            plot = sns.displot(df, x='Degree',hue='Label', kind='kde')

        plt.title('Degree Distributions')
        return plot

    def drawAnomalousSubgraphs(self,G,anomalous_nodes): 
        connected_components = list(nx.connected_components(G))
        num_components = len(connected_components)
        ## Add subplots
        for i,component in enumerate(connected_components):
            G_ = G.subgraph(component)
            colour_map = ['red' if node in anomalous_nodes else 'blue' for node in component]
            nx.draw(G_, with_labels=True, node_color=colour_map)
            
        return