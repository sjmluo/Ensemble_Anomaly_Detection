import networkx as nx
import spartan as st
from .SCAN import SCAN as SCAN_MODEL
from .oddball import get_feature, star_or_clique, star_or_clique_withLOF

class Eigenspoke: 
    '''
        Outlier node detection using SVD.

        Attributes
        ----------
        G : networkx graph object
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
        return preds

class SCAN:
    '''
        Outlier node detection using community detection (via clustering).
        Sensitive to input parameters.
        
        Outliers are nodes with all its neighbours either belonging to one cluster or no cluster.
        
        Attributes
        ----------
        G: networkx graph object (undirected, unweighted graph)
        epsilon: threshold for node structural similarity. Recommended 0-5-0.8
        mu: threshold number of neighbours sharing structural similarity
        SCAN: SCAN model object
    '''
    
    def __init__(self, epsilon=0.5, mu=2):
        self.epsilon = epsilon
        self.mu = mu
        self.G = None
        self.SCAN = None
        
    def fit(self, G):
        self.G = G
        self.SCAN = SCAN_MODEL(self.G, self.epsilon, self.mu)
        
    def predict(self):
        communities = self.SCAN.execute() 
        hubs, outliers = self.SCAN.get_hubs_outliers(communities)
        return outliers
    
class Oddball:
    '''
        Outlier node detection with a (graph) feature-based approach. 
        
        Attributes
        ----------
        G: networkx graph object (undirected, weighted/unweighted graph)
        featureDict: dictionary of node attributes computed from G
        withLOF: boolean whether local outlier factor (LOF) is used 
        numOutliers: int how many outliers
    '''
    
    def __init__(self, withLOF=True, numOutliers=None):
        self.withLOF = withLOF
        self.numOutliers = numOutliers
        self.G = None
        self.featureDict = None            
            
    def fit(self, G):
        self.G = G
        self.featureDict = get_feature(G)
        
        if self.numOutliers is None:
            ## Highest 5% of nodes are outliers
            n = self.G.number_of_nodes()
            self.numOutliers = max(1, int(n*5/100))
        
    def predict(self):
        if self.withLOF:
            pred_nodes = star_or_clique_withLOF(self.featureDict)
        else:
            pred_nodes = star_or_clique(self.featureDict)
            
        sorted_pred_nodes = sorted(pred_nodes.keys(), key=lambda x: pred_nodes[x], reverse=True)
        return sorted_pred_nodes[:self.numOutliers]