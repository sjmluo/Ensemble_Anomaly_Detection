import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from .metrics import metrics, MetricCollection
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from umap import UMAP

class EvaluationFramework:
    def __init__(self, model, metrics=metrics):
        self.model = model
        self.metrics = MetricCollection(metrics)

    def fit(self, x):
        self.model.fit(x)

    def predict(self, x):
        y_pred = self.model.predict(x)
        return y_pred

    def predict_proba(self, x):
        y_pred = self.model.predict_proba(x)
        return y_pred

    def score(self,y,y_pred):
        scores = self.metrics.compute(y,y_pred)
        return scores

    def dim_reduction(self, x, method='pca'):
        '''
            method :: pca, tsne, umap
            returns a (,2) numpy array
        '''
        if method == 'pca':
            pca = PCA(n_components=2)
            return pca.fit_transform(x)
        elif method == 'tsne':
            tsne = TSNE(n_components=2)
            return tsne.fit_transform(x)
        elif method == 'umap':
            umap = UMAP()
            return umap.fit_transform(x)
        else:
            return None

    def visualise(self, x, y=None, method='pca'):
        x_reduced = self.dim_reduction(x, method=method)

        if y is None:
            y = self.predict(x)

        if x_reduced is None:
            print('Method not recognised')
            return
        plt.plot(x_reduced[y==0,0], x_reduced[y==0,1], '.',label="Normal")
        plt.plot(x_reduced[y==1,0], x_reduced[y==1,1], 'x',label="Anomaly")
        plt.title(label=method.upper())
        plt.legend()

        if method == 'pca':
            plt.ylabel('PCA2')
            plt.xlabel('PCA1')

    def contour(self,x,method="pca"):
        # Dimension Reduction
        if method == 'pca':
            reducer = PCA(n_components=2)
        elif method == 'umap':
            reducer = UMAP()
        else:
            print('Failed to find method.')
            return

        # Grid Coordinates
        reducer.fit(x)
        x_reduced = reducer.fit_transform(x)
        x_min, y_min = x_reduced.min(0)
        x_max, y_max = x_reduced.max(0)
        xcoords = np.linspace(x_min,x_max,50)
        ycoords = np.linspace(y_min,y_max,50)

        x_grid = np.array([[i,j] for i in xcoords for j in ycoords])

        x_ = reducer.inverse_transform(x_grid)
        y_preds = self.model.predict_proba(x_)
        return xcoords,ycoords,x_grid,y_preds

    def heatmap(self, x, method='pca'):
        _,_,x_grid,y_preds = self.contour(x,method)

        # Heatmap
        plt.scatter(x_grid[:,0], x_grid[:,1], c=y_preds[:,1])
        cbar = plt.colorbar()
        cbar.ax.set_ylabel('Anomaly Likelihood', rotation=270)
        cbar.ax.get_yaxis().labelpad = 15
        plt.title(label=method.upper())

    def pairplot(self, x, palette='RdBu'):
        df = pd.DataFrame(x)
        # Colorbar Setup
        prob_anom = self.predict_proba(x)[:,1]
        norm = plt.Normalize(prob_anom.min(), prob_anom.max())
        sm = plt.cm.ScalarMappable(cmap=palette, norm=norm)
        sm.set_array([])
        # Graphs
        g = sns.PairGrid(df, palette=palette)
        g.map_lower(sns.kdeplot)
        g.map_diag(sns.kdeplot)
        g.map_upper(sns.scatterplot, hue=prob_anom)
        # Styling
        cbar_ax = g.fig.add_axes([0.85, 0.15, 0.05, 0.7])
        g.fig.subplots_adjust(right=0.8, top=0.9)
        g.fig.colorbar(sm, cax=cbar_ax)
        g.fig.suptitle('Pairwise Plot')
        cbar_ax.set_ylabel('Anomaly Likelihood', rotation=270)
        cbar_ax.get_yaxis().labelpad = 15
        return g
