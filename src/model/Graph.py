from logging import log
import dgl
import tensorflow as tf
import numpy as np
import pdb

from dgl.nn.tensorflow import conv, glob, HeteroGraphConv
from tensorflow.keras import Model, layers, Input

class contextual_layers(layers.Layer):
    def __init__(self, in_feats, h_feats, **kwarg):
        super().__init__(**kwarg)
        self.gat1 = conv.GATConv(in_feats, h_feats, 1)
        self.gat2 = conv.GATConv(in_feats, h_feats, 1)
        self.gat3= conv.GATConv(in_feats, h_feats, 1)

    def call(self, g, in_feat):
        h = self.gat1(g, in_feat)
        h = tf.nn.relu(h)
        h = self.gat2(g, h)
        h = tf.nn.relu(h)
        h = self.gat3(g, h)
        return h

class hierarchical_layers(layers.Layer):
    def __init__(self, in_feats, h_feats, **kwarg):
        super().__init__(**kwarg)
        self.gat1 = conv.GATConv(in_feats, h_feats, 1)
        self.gat2 = conv.GATConv(in_feats, h_feats, 1)
        self.gat3= conv.GATConv(in_feats, h_feats, 1)

    def call(self, g, in_feat):
        h = self.gat1(g, in_feat)
        h = tf.nn.relu(h)
        h = self.gat2(g, h)
        h = tf.nn.relu(h)
        h = self.gat3(g, h)
        return h

if __name__ == '__main__':
    # Graph construction
    dim_h = 256
    g1 = dgl.graph(([0, 0, 0, 0, 0], [1, 2, 3, 4, 5]), num_nodes = 1024)
    print(g1.edges())


    g2 = dgl.graph(([0, 0, 0], [1, 2, 3]), num_nodes = 4)
    g1 = dgl.add_reverse_edges(g1)
    g2 = dgl.add_reverse_edges(g2)
    # load data to graph
    features = g1.ndata['x'] = tf.ones((g1.num_nodes(), 256))
    features = g2.ndata['x'] = tf.ones((g2.num_nodes(), 256))


    lay1 = contextual_layers(g1.ndata['x'].shape[1], dim_h)
    
    h_out = lay1(g1, features)
    h_out = tf.squeeze(h_out)
    # print(g1.ndata['x'].shape)
    # print(h_out.shape)
    features = g1.ndata['x'] = tf.ones((g1.num_nodes(), 256))
    model = contextual_layers(g1.ndata['x'].shape[1], dim_h)