from .helpers import *
from .fraudar import logWeightedAveDegree
import networkx as nx
import spartan as st



class Fraudar:
    '''
        Outlier subgraph fraud detection using graph structure.

        Attributes
        ----------
        G : bipartite graph 
        M : Adjacency matrix
    '''
    def __init__(self, priors=False):
        self.priors = False
        self.G = None 
        self.M = None

    def fit(self, G):
        self.G = G
        self.M = generateSparse(G)

    def predict(self):
        if self.G:
            pred_users, pred_products = logWeightedAveDegree(self.M)[0]
            return (pred_users, pred_products)
        else:
            raise AssertionError('No graph found.')

class Eigenspoke: 
    '''
        Outlier node detection using SVD.

        Attributes
        ----------
        G : bipartite graph 
        k : rank
    '''
    def __init__(self, k=10):
        self.k = k

    def fit(self, G):
        self.G = G 

        tmp_df = nx.to_pandas_edgelist(G)
        tensor = st.TensorData(tmp_df[['source','target']].rename(columns={'source':0,'target':1}))
        stensor = tensor.toSTensor(hasvalue=False)
        self.model = st.Eigenspokes(stensor)

    def predict(self):
        preds = self.model.run(k=self.k)
        return (preds, preds)

