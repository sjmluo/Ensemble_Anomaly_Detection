from sklearn.decomposition import NMF


class OutlierLink(object):
    """
    A model for outlier link prediction
    Uses nonnegative matrix factorisation
    ...

    Attributes
    ----------
    nmf : NMF object
        Model used for outlier link detection

    """


    def __init__(self,n_components=10):
        self.nmf = NMF(n_components,*args,**kwargs)

    def fit(self,A):
        """
        Learn a NMF model for the adjacency matrix A.

        Args:
            A (array-like, sparse matrix of shape (n_nodes, n_nodes)):
                A is an adjacency matrix where each element represents the (weighted) link between each node

        Returns:
            None
        """
        self.nmf.fit(A)

    def predict(self,A,tolerance=1e-01):
        """
        Predicts the anomalous links in given adjacency matrix

        Args:
            A (array-like, sparse matrix} of shape (n_nodes, n_nodes)):
                A is an adjacency matrix where each element represents the (weighted) link between each node
            tolerance (float, optional):
                The threshold used to determine what are anomalous links

        Returns:
            l (array-like booleans of shape (n_nodes,n_nodes)): 
                True if link is anomalous, False otherwise
        """
        transformed_A = nmf.transform(A)
        recon_A = nmf.inverse_transform(transformed_A)

        return np.logical_not(
            np.isclose(A,recon_A,atol=tolerance)))
