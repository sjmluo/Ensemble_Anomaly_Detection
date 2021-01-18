import matplotlib.pyplot as plt
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
            #y = predict_results['predictions']

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
