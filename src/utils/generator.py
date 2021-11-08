import tensorflow as tf
import numpy as np
import dgl
import datetime
import pdb

def stochastic_create_edges(g, n_edges = 0):
    assert n_edges>g.num_nodes(), "number of edges is smaller than that of nodes"

    with tf.device('/CPU:0'): # It seems that CPU works faster than GPU in edges creations
        max_edges = (g.num_nodes()-1) * g.num_nodes() // 2  # (0 + g.num_nodes()-1) * g.num_nodes() / 2 

        # Ensure that every nodes has least one edge 
        for i in range(1, g.num_nodes()-1):
            j = np.random.randint(i+1, g.num_nodes())
            g.add_edges(tf.constant([i], dtype = "int64"), tf.constant([j], dtype = "int64"))
        
        # Add the reset of edges
        if n_edges:
            while g.num_edges() < n_edges and g.num_edges() < max_edges:
                i = np.random.randint(0, g.num_nodes())
                j = np.random.randint(0, g.num_nodes())
                g.add_edges(tf.constant([i], dtype = "int64"), tf.constant([j], dtype = "int64")) if not (g.has_edges_between(i,j) or g.has_edges_between(j,i) or i == j) else 0
                if g.num_edges() == max_edges:
                    break
    return dgl.add_reverse_edges(g)


def heterograph():
    graph_data = {
        ('S1', 'contextual', 'S1'): (tf.constant([0]), tf.constant([1])),
        ('S2', 'contextual', 'S2'): (tf.constant([0]), tf.constant([1])),
        ('S3', 'contextual', 'S3'): (tf.constant([0]), tf.constant([1])),
        ('S4', 'contextual', 'S4'): (tf.constant([0]), tf.constant([1])),
        ('S5', 'contextual', 'S5'): (tf.constant([0]), tf.constant([1])),
        }
    
    return dgl.to_bidirected(dgl.heterograph(graph_data))


if __name__ == "__main__":
    g = heterograph()
    print(g.edges(etype = ('S1', 'contextual', 'S1')))
    

    # starttime = datetime.datetime.now()
    # g1 = dgl.graph(([0], [1]), num_nodes = 4096)
    # g1 = stochastic_create_edges(g1,100000)
    # endtime = datetime.datetime.now()
    # print((endtime - starttime).seconds)