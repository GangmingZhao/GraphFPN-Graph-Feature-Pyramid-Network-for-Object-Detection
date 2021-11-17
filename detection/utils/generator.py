import tensorflow as tf
import numpy as np
import dgl
import datetime
import pdb
import os, sys

src_dir = os.path.dirname(os.path.realpath(__file__))
while not src_dir.endswith("Graph-FPN"):
    src_dir = os.path.dirname(src_dir)
if src_dir not in sys.path:
    sys.path.append(src_dir)
from model.network import *

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
    return dgl.add_reverse_edges(g, copy_ndata = True)


def hetero_add_n_feature(g, name_n_feature, indice_node, val):
    g.nodes['n'].data[name_n_feature][indice_node]= val
    return g


def cnn_gnn(c3, c4, c5, g):
    size = np.ones([3, 2])
    size[0]= c3.shape[1:3]
    size[1] = c4.shape[1:3]
    size[2] = c5.shape[1:3]
    for i in range(size[0][0] * size[0][1]):
        g = hetero_add_n_feature(g, "pixel", i, c3[i]) 
    for i in range(size[1][0] * size[1][1]):
        g = hetero_add_n_feature(g, "pixel", size[1][0] * size[1][1] + i, c4[i]) 
    for i in range(size[2][0] * size[2][1]):
        g = hetero_add_n_feature(g, "pixel", size[1][0] * size[1][1] + size[2][0] * size[2][1] + i, c5[i]) 
    return g


def heterograph(name_n_feature, dim_n_feature, nb_nodes, is_birect = True):
    graph_data = {
        ('n', 'contextual', 'n'): (tf.constant([0]), tf.constant([1])),
        ('n', 'hierarchical', 'n'): (tf.constant([0]), tf.constant([1]))
        }
    g = dgl.heterograph(graph_data, num_nodes_dict = {'n': nb_nodes})
    g.nodes['n'].data[name_n_feature] = tf.zeros([g.num_nodes(), dim_n_feature])
    if is_birect:
        with tf.device("/cpu:0"):
            g = dgl.to_bidirected(g, copy_ndata = True)
        g = g.to("/gpu:0")
    return g


def hetero_add_edges(g, u, v, edges):
    if isinstance(u,int):
        g.add_edges(tf.constant([u]), tf.constant([v]), etype = edges)
    elif isinstance(u,list):
        g.add_edges(tf.constant(u), tf.constant(v), etype = edges)
    else:
        g.add_edges(u, v, etype = edges)
    return g




def build_c_edges(g, c3_shape = 28, c4_shape = 14, c5_shape = 7):
    c3_size, c4_size , c5_size= c3_shape * c3_shape, c4_shape * c4_shape, c5_shape * c5_shape
    c3 = tf.range(0, c3_size)
    c4 = tf.range(c3_size, c3_size + c4_size)   
    c5 = tf.range(c3_size + c4_size, c3_size + c4_size + c5_size)
    
    # build contextual and hierarchical edges
    for i in range(c3_shape - 1):
        g = hetero_add_edges(g, c3[i*c3_shape : (i+1)*c3_shape], c3[(i+1)*c3_shape : (i+2)*c3_shape], 'contextual')          # build edges between different rows (27 * 28 = 756)
        g = hetero_add_edges(g, c3[i : (c3_size+i) : c3_shape], c3[i+1 : (c3_size+i+1) : c3_shape], 'contextual')            # build edges between different column (27 * 28 = 756)
    for i in range(c4_shape - 1):
        g = hetero_add_edges(g, c4[i*c4_shape : (i+1)*c4_shape], c4[(i+1)*c4_shape : (i+2)*c4_shape], 'contextual')          # 14 * 13 = 182 
        g = hetero_add_edges(g, c4[i : (c4_size+i) : c4_shape], c4[i+1 : (c4_size+i+1) : c4_shape], 'contextual') 
        g = hetero_add_edges(g, c4[i*c4_shape : (i+1)*c4_shape], c3)
    for i in range(c5_shape - 1):
        g = hetero_add_edges(g, c5[i*c5_shape : (i+1)*c5_shape], c5[(i+1)*c5_shape : (i+2)*c5_shape], 'contextual')          # 6 * 7 = 42
        g = hetero_add_edges(g, c5[i : (c5_size+i) : c5_shape], c5[i+1 : (c5_size+i+1) : c5_shape], 'contextual') 
    
    return g

def simple_graph(g):
    return dgl.to_simple(g)


def hetero_subgraph(g, edges):
    return dgl.edge_type_subgraph(g, [edges])


if __name__ == "__main__":

    g = heterograph("pixel", 256, 2, is_birect = False)
    

    g = build_c_edges(g)
    g = simple_graph(g)
    subh = hetero_subgraph(g, "hierarchical")
    # print(subh.edges())
    # print(subh.num_edges())
    subc = hetero_subgraph(g, "contextual")
    print(subc.edges())
    print(subc.num_edges())
    # a = tf.range(0,16)
    # print(a[0:16:4])
    # dim_h = 3
    # g = heterograph("x", 256, 10)
    # subg = hetero_subgraph(g, "hierarchical")
    # # print(subg.edges())
    # lay1 = contextual_layers(subg.ndata['x'].shape[1], 256)

    # features = subg.ndata['x']
    # # print(features)
    # h_out = tf.squeeze(lay1(subg, features))

    # subg.apply_nodes(lambda nodes: {'x' : h_out})
    # print(g.ndata["x"])


    # print(subg.edges())
    # print(g.nodes['n'].data['x'])
    # hetero_add_n_feature(g, "x", 0, tf.constant([1,2,3,4]))
    # print(g.edges(etype = "contextual"))
    # g = hetero_add_edges(g, 0, 2, "contextual")
    # print(g.edges(etype = "contextual"))
    # print(g.ndata["x"][0])
    

    # starttime = datetime.datetime.now()
    # g1 = dgl.graph(([0], [1]), num_nodes = 4096)
    # g1 = stochastic_create_edges(g1,100000)
    # endtime = datetime.datetime.now()
    # print((endtime - starttime).seconds)