import dgl
import numpy as np
import tensorflow as tf
import scipy.sparse as sp
from dgl.nn import GATConv
import networkx as nx

# # 1.2 图、节点和边
# # 边 0->1, 0->2, 0->3, 1->3
# u, v = tf.constant([0, 0, 0, 1]), tf.constant([1, 2, 3, 3])
# g = dgl.graph((u, v))
# print(g) # 图中节点的数量是DGL通过给定的图的边列表中最大的点ID推断所得出的
# # 获取节点的ID
# print(g.nodes())
# # 获取边的对应端点
# print(g.edges())
# # 获取边的对应端点和边ID
# print(g.edges(form='all'))
# # 如果具有最大ID的节点没有边，在创建图的时候，用户需要明确地指明节点的数量。



# # 1.3 节点和边的特征
# g = dgl.graph(([0, 0, 1, 5], [1, 2, 2, 0])) # 6个节点，4条边
# g
# g.ndata['x'] = tf.ones(g.num_nodes(), 3)               # 长度为3的节点特征
# g.edata['x'] = tf.ones(g.num_edges(), dtype=tf.int32)  # 标量整型特征

# # 不同名称的特征可以具有不同形状
# g.ndata['y'] = tf.random.normal([g.num_nodes(),5])
# g.ndata['x'][1]                  # 获取节点1的特征
# a =  tf.constant([0,3])
# print(tf.gather(g.edata['x'],a))

# # 边 0->1, 0->2, 0->3, 1->3
# edges = tf.constant([0, 0, 0, 1]), tf.constant([1, 2, 3, 3])
# weights = tf.constant([0.1, 0.6, 0.9, 0.7])  # 每条边的权重
# g = dgl.graph(edges)
# g.edata['w'] = weights  # 将其命名为 'w'



# # 1.4 从外部源创建图
# spmat = sp.rand(100, 100, density=0.05) # 5%非零项
# print(spmat)
# dgl.from_scipy(spmat) # 来自SciPy

# # nx中的有向图
# nx_g = nx.path_graph(5) # 一条链路0-1-2-3-4
# dgl.from_networkx(nx_g) # 来自NetworkX

# # nx中无向图
# nxg = nx.DiGraph([(2, 1), (1, 2), (2, 3), (0, 0)])
# dgl.from_networkx(nxg)


# # 1.5 Heterogeneous Graph (异构图)
# # 创建一个具有3种节点类型和3种边类型的异构图
# graph_data = {
#    ('drug', 'interacts', 'drug'): (tf.constant([0, 1]), tf.constant([1, 2])),
#    ('drug', 'interacts', 'gene'): (tf.constant([0, 1]), tf.constant([2, 3])),
#    ('drug', 'treats', 'disease'): (tf.constant([1]), tf.constant([2]))
# }
# g = dgl.heterograph(graph_data)



# Train

# import numpy as np
# import torch

# n_users = 1000
# n_items = 500
# n_follows = 3000
# n_clicks = 5000
# n_dislikes = 500
# n_hetero_features = 10
# n_user_classes = 5
# n_max_clicks = 10

# follow_src = np.random.randint(0, n_users, n_follows)
# follow_dst = np.random.randint(0, n_users, n_follows)
# click_src = np.random.randint(0, n_users, n_clicks)
# click_dst = np.random.randint(0, n_items, n_clicks)
# dislike_src = np.random.randint(0, n_users, n_dislikes)
# dislike_dst = np.random.randint(0, n_items, n_dislikes)

# hetero_graph = dgl.heterograph({
#     ('user', 'follow', 'user'): (follow_src, follow_dst),
#     ('user', 'followed-by', 'user'): (follow_dst, follow_src),
#     ('user', 'click', 'item'): (click_src, click_dst),
#     ('item', 'clicked-by', 'user'): (click_dst, click_src),
#     ('user', 'dislike', 'item'): (dislike_src, dislike_dst),
#     ('item', 'disliked-by', 'user'): (dislike_dst, dislike_src)})

# hetero_graph.nodes['user'].data['feature'] = tf.random.normal([n_users, n_hetero_features])
# hetero_graph.nodes['item'].data['feature'] = tf.random.normal([n_items, n_hetero_features])
# hetero_graph.nodes['user'].data['label'] = tf.random.uniform([n_users], 0, n_user_classes)
# hetero_graph.edges['click'].data['label'] = tf.random.uniform((n_clicks,), 1, n_max_clicks, )
# # 在user类型的节点和click类型的边上随机生成训练集的掩码
# hetero_graph.nodes['user'].data['train_mask'] = tf.zeros(n_users, dtype=tf.bool)
# hetero_graph.edges['click'].data['train_mask'] = tf.zeros(n_clicks, dtype=tf.bool)

# Case 2: Unidirectional bipartite graph
u = [0, 1, 0, 0, 1]
v = [0, 1, 2, 3, 2]
g = dgl.heterograph({('A', 'r', 'B'): (u, v), ('A', 'h', 'B'): (u, v)})
# with tf.device("CPU:0"):
u_feat = tf.convert_to_tensor(np.random.rand(2, 5))
v_feat = tf.convert_to_tensor(np.random.rand(4, 10))
gatconv = GATConv((5,10), 2, 3)
res = gatconv(g, (u_feat, v_feat))
print(res)