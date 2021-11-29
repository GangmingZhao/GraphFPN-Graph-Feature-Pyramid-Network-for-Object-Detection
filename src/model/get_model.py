from .network import Graph_RetinaNet, RetinaNet, get_backbone
from .losses import *

backbone = {"Resnet50": get_backbone(50),
            "Resnet101": get_backbone(101)}

models = {"Retinanet": RetinaNet,
       "Graph_Retinanet": Graph_RetinaNet}

loss = {"RetinaNetLoss":RetinaNetLoss}