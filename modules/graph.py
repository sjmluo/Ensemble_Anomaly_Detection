from sklearn.decomposition import NMF
import pickle
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
        self.A = None

    def fit(self,A):
        """
        Learn a NMF model for the adjacency matrix A.

        Args:
            A (array-like, sparse matrix of shape (n_nodes, n_nodes)):
                A is an adjacency matrix where each element represents the (weighted) link between each node

        Returns:
            None
        """
        self.A = A
        self.nmf.fit(A)

    def predict(self,tolerance=1e-01):
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
        if not self.A:
            raise AssertionError('No graph found.')
        transformed_A = nmf.transform(self.A)
        recon_A = nmf.inverse_transform(transformed_A)

        return np.logical_not(
            np.isclose(self.A,recon_A,atol=tolerance))

def networkx_to_dash(G,filename=None):
    element_list = []
    for n in G:
        node_dict = {}
        node_dict['id'] = n
        node_dict['label'] = n
        node_dict = {**node_dict,**G.nodes[n]}
        print(node_dict)
        outer_dict = {
            'data': node_dict
        }
        element_list.append(outer_dict)
        for e in G[n]:
            edge_dict = {
                'target':n,
                'source':e
            }
            edge_outer_dict= {
                'data':edge_dict
            }
            element_list.append(edge_outer_dict)
    if filename:
        with open(f"{filename}.pickle", "wb") as output_file:
            pickle.dump(element_list, output_file)
    return element_list
