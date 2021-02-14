from scipy import sparse
import networkx as nx

def readEdges(filename):
    g = nx.Graph()
    with open(filename,'r') as f:
        for i,line in enumerate(f):
            toks = line.split()
            g.add_edge(f'50{int(toks[0])}', f'100{int(toks[1])}')
    return g

def generateSparse(g):
    src, dest = [], []
    for u,v in g.edges():
        src.append(int(u))
        dest.append(int(v))
        
    num_src =  max(src)+1 # max(len(list(filter(lambda x: x[0]=='R', g.nodes()))), max(src)+1)
    num_dest= max(dest)+1 # max(g.number_of_nodes() - num_src, max(dest)+1)
    M = sparse.coo_matrix(([1]*len(src), (src, dest)), shape=(num_src, num_dest))
    return (M > 0).astype(int)