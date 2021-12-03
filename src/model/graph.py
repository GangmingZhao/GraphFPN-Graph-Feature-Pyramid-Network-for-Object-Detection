import tensorflow as tf
import numpy as np
import dgl
import keras.backend as K
from dgl.nn.tensorflow import glob

from .network import *

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


def heterograph(name_n_feature, dim_n_feature, nb_nodes = 2, is_birect = True):
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


def neighbor_9(i, c_shape):
    return tf.constant([i-c_shape-1, i-c_shape, i-c_shape+1, i-1, i, i+1, i+c_shape-1, i+c_shape, i+c_shape+1])


def neighbor_25(i, c_shape):
    return tf.constant([i-2*c_shape-2, i-2*c_shape-1, i-2*c_shape, i-2*c_shape+1, i-2*c_shape+2,
                        i-c_shape-2, i-c_shape-1, i-c_shape, i-c_shape+1, i-c_shape+2, 
                        i-2, i-1, i, i+1, i+2, 
                        i+c_shape-2, i+c_shape-1, i+c_shape, i+c_shape+1, i+c_shape+2,
                        i+2*c_shape-2, i+2*c_shape-1, i+2*c_shape, i+2*c_shape+1, i+2*c_shape+2])


def simple_graph(g):
    g = g.to("cpu:0")
    g = dgl.to_simple(g, copy_ndata = True)
    g = g.to("gpu:0")
    return g


def to_birected(g):
    with tf.device("/cpu:0"):
        g = dgl.to_bidirected(g, copy_ndata = True)
    g = g.to("gpu:0")
    return g


def simple_birected(g):
    g = g.to("cpu:0")
    g = dgl.to_simple(g, copy_ndata = True)
    with tf.device("/cpu:0"):
        g = dgl.to_bidirected(g, copy_ndata = True)
    g = g.to("gpu:0")
    return g


# local pooling based on neighbor nodes, worked but it slows the training loop
def avg_pool_local(g, etype):
    for node in range(g.num_nodes()):
        _, neighbor = g.out_edges(node, form='uv', etype = etype)  # return srcnodes and dstnodes
        # local_g = g.out_subgraph({"n" : [node]})
        local_g = dgl.node_subgraph(g, neighbor)
        # print(local_g.ndata["pixel"])
        pool = glob.AvgPooling()
        h = pool(local_g, local_g.ndata["pixel"])
        g.apply_nodes(lambda nodes: {'pixel' : h}, v = node)
        # h = dgl.nn.tensorflow.glob.AvgPooling(local_g, )
        # print(h)
        # pdb.set_trace()
        # _, neighbor = g.out_edges(node, form='uv', etype = etype)  # return srcnodes and dstnodes
        # neighbor_data = tf.gather(g.ndata["pixel"], neighbor)
        # mean = tf.expand_dims(tf.reduce_mean(neighbor_data, axis = 0), axis = 0)
        # g.apply_nodes(lambda nodes: {'pixel' : mean}, v = node)
    return g
        
    

def build_edges(g, c3_shape = 28, c4_shape = 14, c5_shape = 7):
    c3_size, c4_size , c5_size= c3_shape * c3_shape, c4_shape * c4_shape, c5_shape * c5_shape
    c3 = tf.range(0, c3_size)
    c4 = tf.range(c3_size, c3_size + c4_size)   
    c5 = tf.range(c3_size + c4_size, c3_size + c4_size + c5_size)
    
    # build contextual edges
    for i in range(c3_shape - 1):
        g = hetero_add_edges(g, c3[i*c3_shape : (i+1)*c3_shape], c3[(i+1)*c3_shape : (i+2)*c3_shape], 'contextual')          # build edges between different rows (27 * 28 = 756)
        g = hetero_add_edges(g, c3[i : (c3_size+i) : c3_shape], c3[i+1 : (c3_size+i+1) : c3_shape], 'contextual')            # build edges between different column (27 * 28 = 756)
    for i in range(c4_shape - 1):
        g = hetero_add_edges(g, c4[i*c4_shape : (i+1)*c4_shape], c4[(i+1)*c4_shape : (i+2)*c4_shape], 'contextual')          # 14 * 13 = 182 
        g = hetero_add_edges(g, c4[i : (c4_size+i) : c4_shape], c4[i+1 : (c4_size+i+1) : c4_shape], 'contextual') 
        # g = hetero_add_edges(g, c4[i*c4_shape : (i+1)*c4_shape], c3)
    for i in range(c5_shape - 1):
        g = hetero_add_edges(g, c5[i*c5_shape : (i+1)*c5_shape], c5[(i+1)*c5_shape : (i+2)*c5_shape], 'contextual')          # 6 * 7 = 42
        g = hetero_add_edges(g, c5[i : (c5_size+i) : c5_shape], c5[i+1 : (c5_size+i+1) : c5_shape], 'contextual') 
    
    # build hierarchical edges
    c3_stride = tf.reshape(c3, (c3_shape, c3_shape))[2:c3_shape:2, 2:c3_shape:2]  # Get pixel indices in C3 for build hierarchical edges
    c4_stride = tf.reshape(c4, (c4_shape, c4_shape))[2:c4_shape:2, 2:c4_shape:2]
    c5_stride = tf.reshape(c3, (c3_shape, c3_shape))[2:c3_shape-4:4, 2:c3_shape-4:4]
    

    
    # Edges between c3 and c4
    counter = 1
    for i in tf.reshape(c3_stride, [-1]).numpy():
        c3_9 = neighbor_9(i, c3_shape)
        g = hetero_add_edges(g, c3_9, c4[counter], 'hierarchical') 
        if counter % (c4_shape-1) == 0 :
            counter += 2 
        else:
            counter += 1

    # Edges between c4 and c5
    counter = 1
    for i in tf.reshape(c4_stride, [-1]).numpy():

        c4_9 = neighbor_9(i, c4_shape)
        g = hetero_add_edges(g, c4_9, c5[counter], 'hierarchical') 
        if counter % (c5_shape-1) == 0 :
            counter += 2 
        else:
            counter += 1
    
    # Edges between c3 and c5
    counter = 1
    for i in tf.reshape(c5_stride, [-1]).numpy():
        c5_9 = neighbor_25(i, c3_shape)
        g = hetero_add_edges(g, c5_9, c5[counter], 'hierarchical') 
        if counter % (c5_shape-1) == 0 :
            counter += 2 
        else:
            counter += 1
    return g


def nodes_update(g, val):
    g.apply_nodes(lambda nodes: {'pixel' : val})


def hetero_subgraph(g, edges):
    return dgl.edge_type_subgraph(g, [edges])


def cnn_gnn(g, c):
    g.ndata["pixel"] = c
    return g


def gnn_cnn(g): 
    p3 = tf.reshape(g.ndata["pixel"][:784], (1, 28, 28, 256))              # number of pixel in layers p3, 28*28 = 784
    p4 = tf.reshape(g.ndata["pixel"][784:980], (1, 14, 14, 256))            # number of pixel in layers p4, 14*14 = 196
    p5 = tf.reshape(g.ndata["pixel"][980:1029], (1, 7, 7, 256))           # number of pixel in layers p5, 7*7 = 49
    return p3, p4, p5


if __name__ == "__main__":

    g = heterograph("pixel", 256, 1029, is_birect = False)
    g = simple_birected(build_edges(g))
    g.ndata["pixel"] = tf.random.uniform([g.num_nodes(), 256], minval=-10, maxval=10)
    c_layer = contextual_layers(256, 256)
    subc = hetero_subgraph(g, "contextual")
    subh = hetero_subgraph(g, "hierarchical")
    nodes_update(subc, c_layer(subc, subc.ndata["pixel"]))
    print(subc.ndata["pixel"])
    print(g.ndata["pixel"])
    print(subh.ndata["pixel"])
    # g = avg_pool_local(sub_c, "contextual")
    # starttime = datetime.datetime.now()
    # g1 = dgl.graph(([0], [1]), num_nodes = 4096)

    # g1 = stochastic_create_edges(g1,100000)
    # endtime = datetime.datetime.now()
    # print((endtime - starttime).seconds)
