import spektral
import tensorflow as tf
import numpy as np

x = np.random.rand(6,5,4)
y = np.random.rand(6,)
a = np.ones((6,6))
g = spektral.data.graph.Graph(x, a, y=y, n_node_features = (5,4))

print(g)
