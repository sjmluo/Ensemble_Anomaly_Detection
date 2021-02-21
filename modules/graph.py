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
            np.isclose(self.A,recon_A,atol=tolerance)))

class EnsembleGraphOutlier(object):
    """
    Used for ensembling/combining predictions of the same class of model
    ...

    Attributes
    ----------
    models : a list of models used for ensembling

    hyperparams : a list of dicts containing the hyperparameters for each model

    """


    def __init__(self,model,hyperparams):
        """
        Args:
            model:
                the base model for the ensemble
            hyperparameter:
                a list of dicts containing the hyperparameters for each model
        """
        self.models = []
        for i in range(len(hyperparameters)):
            model_object = model(**hyparapmeters[i])
            self.models.append(model_object)
        self.hyperparams = hyperparams

    def fit(self,G):
        """
        Learn each of the models in the ensemble

        Args:
            G:
             The graph, either networkx or adjacency matrix to train the model

        Returns:
            None
        """
        for ele in self.models:
            ele.fit(G)

    def predict(self,*args,**kwargs):
        labels = []
        for ele in self.models:
            labels.append(ele.predict())


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
        with open(f"../datasets/{filename}.pickle", "wb") as output_file:
            pickle.dump(element_list, output_file)
    return element_list
