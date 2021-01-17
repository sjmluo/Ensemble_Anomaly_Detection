import matplotlib.pyplot as plt
from .Metrics import metrics, MetricCollection
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from umap import UMAP

class EvaluationFramework:
    def __init__(self, model, metrics=metrics):
        self.model = model
        self.metrics = MetricCollection(metrics)
        
    def fit(self, x):
        self.model.fit(x)
        
    def predict(self, x, y=None):
        y_pred = self.model.predict(x)
        if y is not None:
            scores = self.metrics.compute(y,y_pred)
            return { 'predictions': y_pred, 'scores': scores }
        return { 'predictions': y_pred }
    
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
            predict_results = self.predict(x)
            y = predict_results['predictions']
        
        if x_reduced is None:
            print('Method not recognised')
            return
        plt.plot(x_reduced[y==0,0], x_reduced[y==0,1], '.')
        plt.plot(x_reduced[y==1,0], x_reduced[y==1,1], 'x')
        plt.title(label=method)
        
        if method == 'pca':
            plt.ylabel('PCA2')
            plt.xlabel('PCA1')