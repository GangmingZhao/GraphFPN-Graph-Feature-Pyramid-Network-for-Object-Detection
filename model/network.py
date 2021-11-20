import tensorflow as tf
import keras.backend as K
import numpy as np
import os, sys
import dgl
import pdb

from tensorflow.python.ops.gen_math_ops import Sigmoid

src_dir = os.path.dirname(os.path.realpath(__file__))
while not src_dir.endswith("Graph-FPN"):
    src_dir = os.path.dirname(src_dir)
if src_dir not in sys.path:
    sys.path.append(src_dir)

from tensorflow import keras
from detection.utils.graph import *
from detection.utils.anchor import *
from detection.utils.bbox import *
from dgl.nn.tensorflow import conv, glob

"""Building the ResNet50 backbone
RetinaNet uses a ResNet based backbone, using which a feature pyramid network is constructed. In the example we use ResNet50 as the backbone, and return the feature maps at strides 8, 16 and 32."""


def get_backbone(number_layers):
    """Builds ResNet50 with pre-trained imagenet weights"""
    if number_layers == 50:
        backbone = keras.applications.ResNet50(include_top=False, input_shape=[224, 224, 3], weights = 'imagenet')
    elif number_layers == 101:
        backbone = keras.applications.ResNet101(include_top=False, input_shape=[224, 224, 3], weights = 'imagenet')
    c3_output, c4_output, c5_output = [backbone.get_layer(layer_name).output for layer_name in ["conv3_block4_out", "conv4_block6_out", "conv5_block3_out"]]
    return keras.Model(inputs=[backbone.inputs], outputs=[c3_output, c4_output, c5_output])


# Building Feature Pyramid Network as a custom layer

class FeaturePyramid(keras.layers.Layer):
    """Builds the Feature Pyramid with the feature maps from the backbone.

    Attributes:
      num_classes: Number of classes in the dataset.
      backbone: The backbone to build the feature pyramid from.
        Currently supports ResNet50 only.
    """

    def __init__(self, backbone=None, **kwargs):
        super(FeaturePyramid, self).__init__(name="FeaturePyramid", **kwargs)
        self.backbone = backbone if backbone else get_backbone()
        self.conv_c3_1x1 = keras.layers.Conv2D(256, 1, 1, "same")
        self.conv_c4_1x1 = keras.layers.Conv2D(256, 1, 1, "same")
        self.conv_c5_1x1 = keras.layers.Conv2D(256, 1, 1, "same")
        self.conv_c3_3x3 = keras.layers.Conv2D(256, 3, 1, "same")
        self.conv_c4_3x3 = keras.layers.Conv2D(256, 3, 1, "same")
        self.conv_c5_3x3 = keras.layers.Conv2D(256, 3, 1, "same")
        self.conv_c6_3x3 = keras.layers.Conv2D(256, 3, 2, "same")
        self.conv_c7_3x3 = keras.layers.Conv2D(256, 3, 2, "same")
        self.upsample_2x = keras.layers.UpSampling2D(2)

    def call(self, images, training=False):
        c3_output, c4_output, c5_output = self.backbone(images, training=training)
        p3_output = self.conv_c3_1x1(c3_output)
        p4_output = self.conv_c4_1x1(c4_output)
        p5_output = self.conv_c5_1x1(c5_output)
        p4_output = p4_output + self.upsample_2x(p5_output)
        p3_output = p3_output + self.upsample_2x(p4_output)
        p3_output = self.conv_c3_3x3(p3_output)
        p4_output = self.conv_c4_3x3(p4_output)
        p5_output = self.conv_c5_3x3(p5_output)
        p6_output = self.conv_c6_3x3(c5_output)
        p7_output = self.conv_c7_3x3(tf.nn.relu(p6_output))
        return p3_output, p4_output, p5_output, p6_output, p7_output


class channel_attention(keras.layers.Layer):
    def __init__(self, etype,**kwarg):
        super().__init__(**kwarg)
        self.etype = etype
        self.pool1 = avg_pool_local
        self.fc1 = keras.layers.Dense(256, activation = "sigmoid")
         
    def call(self, g, h):
        a = self.pool1(g, self.etype)
        h = tf.squeeze(h)
        h = tf.math.multiply(h, a)
        return h


# class contextual_layers(keras.layers.Layer):
#     def __init__(self, in_feats, h_feats, **kwarg):
#         super().__init__(**kwarg)
#         self.in_feats = in_feats
#         self.gat1 = conv.GATConv(in_feats, h_feats, 1)
#         self.cha2 = channel_attention("contextual")
#         self.gat3= conv.GATConv(in_feats, h_feats, 1)

#     def call(self, g, in_feat):
#         h = self.gat1(g, in_feat)
#         h = tf.nn.relu(h)
#         h = self.cha2(g, h)
#         h = tf.nn.relu(h)
#         h = self.gat3(g, h)
#         h= tf.squeeze(h)
#         return h

class contextual_layers(keras.layers.Layer):
    def __init__(self, in_feats, h_feats, **kwarg):
        super().__init__(**kwarg)
        self.in_feats = in_feats
        self.gat1 = conv.GATConv(in_feats, h_feats, 1)
        self.gat2 = conv.GATConv(in_feats, h_feats, 1)
        self.gat3= conv.GATConv(in_feats, h_feats, 1)

    def call(self, g, in_feat):
        h = self.gat1(g, in_feat)
        h = tf.nn.relu(h)
        h = self.gat2(g, h)
        h = tf.nn.relu(h)
        h = self.gat3(g, h)
        h= tf.squeeze(h)
        return h


class hierarchical_layers(keras.layers.Layer):
    def __init__(self, in_feats, h_feats, **kwarg):
        super().__init__(**kwarg)
        self.in_feats = in_feats
        self.gat1 = conv.GATConv(in_feats, h_feats, 1)
        self.gat2 = conv.GATConv(in_feats, h_feats, 1)
        self.gat3= conv.GATConv(in_feats, h_feats, 1)

    def call(self, g, in_feat):
        h = self.gat1(g, in_feat)
        h = tf.nn.relu(h)
        h = self.gat2(g, h)
        h = tf.nn.relu(h)
        h = self.gat3(g, h)
        h = tf.squeeze(h)
        return h



class graph_FeaturePyramid(keras.layers.Layer):

    def __init__(self, backbone=None, **kwargs):
        super(graph_FeaturePyramid, self).__init__(name="graph_FeaturePyramid", **kwargs)
        self.backbone = backbone if backbone else get_backbone()
        self.conv_c3_1x1 = keras.layers.Conv2D(256, 1, 1, "same")
        self.conv_c4_1x1 = keras.layers.Conv2D(256, 1, 1, "same")
        self.conv_c5_1x1 = keras.layers.Conv2D(256, 1, 1, "same")
        self.conv_c3_3x3 = keras.layers.Conv2D(256, 3, 1, "same")
        self.conv_c4_3x3 = keras.layers.Conv2D(256, 3, 1, "same")
        self.conv_c5_3x3 = keras.layers.Conv2D(256, 3, 1, "same")
        self.conv_c6_3x3 = keras.layers.Conv2D(256, 3, 2, "same")
        self.conv_c7_3x3 = keras.layers.Conv2D(256, 3, 2, "same")
        self.upsample_2x = keras.layers.UpSampling2D(2)
        self.context1 = contextual_layers(256, 256)
        self.context2 = contextual_layers(256, 256)
        self.context3 = contextual_layers(256, 256)
        self.hierarch1 = hierarchical_layers(256, 256)
        self.hierarch2 = hierarchical_layers(256, 256)
        self.hierarch3 = hierarchical_layers(256, 256)
        self.context4 = contextual_layers(256, 256)
        self.context5 = contextual_layers(256, 256)
        self.context6 = contextual_layers(256, 256)
        self.g = simple_birected(build_edges(heterograph("pixel", 256, 1029)))
        self.g = dgl.add_self_loop(self.g, etype = "hierarchical")
        self.subg_h = hetero_subgraph(self.g, "hierarchical")
        self.subg_c = hetero_subgraph(self.g, "contextual")


    def call(self, images, training=False):
        c3_output, c4_output, c5_output = self.backbone(images, training=training)
        # Convolutional feature pyramid network
        p3_output = self.conv_c3_1x1(c3_output)
        p4_output = self.conv_c4_1x1(c4_output)
        p5_output = self.conv_c5_1x1(c5_output)
        # Graph feature pyramid network
        p3_gnn = tf.reshape(p3_output, [-1, 256])        
        p4_gnn = tf.reshape(p4_output, [-1, 256])
        p5_gnn = tf.reshape(p5_output, [-1, 256])
        p_final = tf.concat([p3_gnn, p4_gnn, p5_gnn], axis = 0)
        self.g = cnn_gnn(self.g, p_final)
        nodes_update(self.subg_c, self.context1(self.subg_c, self.subg_c.ndata["pixel"]))
        nodes_update(self.subg_c, self.context2(self.subg_c, self.subg_c.ndata["pixel"]))
        nodes_update(self.subg_c, self.context3(self.subg_c, self.subg_c.ndata["pixel"]))
        nodes_update(self.subg_h, self.hierarch1(self.subg_h, self.subg_h.ndata["pixel"]))
        nodes_update(self.subg_h, self.hierarch2(self.subg_h, self.subg_h.ndata["pixel"]))
        nodes_update(self.subg_h, self.hierarch3(self.subg_h, self.subg_h.ndata["pixel"]))
        nodes_update(self.subg_c, self.context4(self.subg_c, self.subg_c.ndata["pixel"]))
        nodes_update(self.subg_c, self.context5(self.subg_c, self.subg_c.ndata["pixel"]))
        nodes_update(self.subg_c, self.context6(self.subg_c, self.subg_c.ndata["pixel"]))
        # data fusion 
        p3_gnn, p4_gnn, p5_gnn = gnn_cnn(self.g)
        p5_output = p5_output + p5_gnn
        p4_output = p4_output + self.upsample_2x(p5_output) + p4_gnn
        p3_output = p3_output + self.upsample_2x(p4_output) + p3_gnn
        p5_output = p5_output 
        p3_output = self.conv_c3_3x3(p3_output)
        p4_output = self.conv_c4_3x3(p4_output)
        p5_output = self.conv_c5_3x3(p5_output)
        p6_output = self.conv_c6_3x3(c5_output)
        p7_output = self.conv_c7_3x3(tf.nn.relu(p6_output))
        return p3_output, p4_output, p5_output, p6_output, p7_output




# Building the classification and box regression heads.
# The RetinaNet model has separate heads for bounding box regression and for predicting class probabilities for the objects. These heads are shared between all the feature maps of the feature pyramid.

def build_head(output_filters, bias_init):
    """Builds the class/box predictions head.

    Arguments:
      output_filters: Number of convolution filters in the final layer.
      bias_init: Bias Initializer for the final convolution layer.

    Returns:
      A keras sequential model representing either the classification
        or the box regression head depending on `output_filters`.
    """
    head = keras.Sequential([keras.Input(shape=[None, None, 256])])
    kernel_init = tf.initializers.RandomNormal(0.0, 0.01)
    for _ in range(4):
        head.add(
            keras.layers.Conv2D(256, 3, padding="same", kernel_initializer=kernel_init)
        )
        head.add(keras.layers.ReLU())
    head.add(
        keras.layers.Conv2D(
            output_filters,
            3,
            1,
            padding="same",
            kernel_initializer=kernel_init,
            bias_initializer=bias_init,
        )
    )
    return head



class Graph_RetinaNet(keras.Model):
    """A subclassed Keras model implementing the RetinaNet architecture.

    Attributes:
      num_classes: Number of classes in the dataset.
      backbone: The backbone to build the feature pyramid from.
        Currently supports ResNet50 only.
    """

    def __init__(self, num_classes, backbone=None, **kwargs):
        super().__init__(name="Graph_RetinaNet", **kwargs)
        self.fpn = graph_FeaturePyramid(backbone)
        self.num_classes = num_classes

        prior_probability = tf.constant_initializer(-np.log((1 - 0.01) / 0.01))
        self.cls_head = build_head(9 * num_classes, prior_probability)
        self.box_head = build_head(9 * 4, "zeros")

    def call(self, image, training=False):
        features = self.fpn(image, training=training)
        N = tf.shape(image)[0]
        cls_outputs = []
        box_outputs = []
        for feature in features:
            box_outputs.append(tf.reshape(self.box_head(feature), [N, -1, 4]))
            cls_outputs.append(
                tf.reshape(self.cls_head(feature), [N, -1, self.num_classes])
            )
        cls_outputs = tf.concat(cls_outputs, axis=1)
        box_outputs = tf.concat(box_outputs, axis=1)
        return tf.concat([box_outputs, cls_outputs], axis=-1)

    def train_step(self, data):
        x, y = data
        with tf.GradientTape(persistent = True) as t:
            y_pred = self(x, training = True)
            loss = self.compiled_loss(y, y_pred)

        # graph_grad = t.gradient(loss, self.fpn.contextual.trainable_variables)
        # print(graph_grad)
        # pdb.set_trace()
        vars = self.trainable_variables
        grad = t.gradient(loss, vars)
        # self.optimizer.apply_gradients((grad, vars) for (grad, vars) in zip(grad, vars) if grad is not None)
        self.optimizer.apply_gradients(zip(grad, vars))
        self.compiled_metrics.update_state(y, y_pred)
        return {m.name: m.result() for m in self.metrics}




class RetinaNet(keras.Model):
    """A subclassed Keras model implementing the RetinaNet architecture.

    Attributes:
      num_classes: Number of classes in the dataset.
      backbone: The backbone to build the feature pyramid from.
        Currently supports ResNet50 only.
    """

    def __init__(self, num_classes, backbone=None, **kwargs):
        super(RetinaNet, self).__init__(name="RetinaNet", **kwargs)
        self.fpn = FeaturePyramid(backbone)
        self.num_classes = num_classes

        prior_probability = tf.constant_initializer(-np.log((1 - 0.01) / 0.01))
        self.cls_head = build_head(9 * num_classes, prior_probability)
        self.box_head = build_head(9 * 4, "zeros")

    def call(self, image, training=False):
        features = self.fpn(image, training=training)
        N = tf.shape(image)[0]
        cls_outputs = []
        box_outputs = []
        for feature in features:
            box_outputs.append(tf.reshape(self.box_head(feature), [N, -1, 4]))
            cls_outputs.append(
                tf.reshape(self.cls_head(feature), [N, -1, self.num_classes])
            )
        cls_outputs = tf.concat(cls_outputs, axis=1)
        box_outputs = tf.concat(box_outputs, axis=1)
        return tf.concat([box_outputs, cls_outputs], axis=-1)





# Implementing a custom layer to decode predictions

class DecodePredictions(tf.keras.layers.Layer):
    """A Keras layer that decodes predictions of the RetinaNet model.

    Attributes:
      num_classes: Number of classes in the dataset
      confidence_threshold: Minimum class probability, below which detections
        are pruned.
      nms_iou_threshold: IOU threshold for the NMS operation
      max_detections_per_class: Maximum number of detections to retain per
       class.
      max_detections: Maximum number of detections to retain across all
        classes.
      box_variance: The scaling factors used to scale the bounding box
        predictions.
    """

    def __init__(
        self,
        num_classes=80,
        confidence_threshold=0.05,
        nms_iou_threshold=0.5,
        max_detections_per_class=100,
        max_detections=100,
        box_variance=[0.1, 0.1, 0.2, 0.2],
        **kwargs
    ):
        super(DecodePredictions, self).__init__(**kwargs)
        self.num_classes = num_classes
        self.confidence_threshold = confidence_threshold
        self.nms_iou_threshold = nms_iou_threshold
        self.max_detections_per_class = max_detections_per_class
        self.max_detections = max_detections

        self._anchor_box = AnchorBox()
        self._box_variance = tf.convert_to_tensor(
            [0.1, 0.1, 0.2, 0.2], dtype=tf.float32
        )

    def _decode_box_predictions(self, anchor_boxes, box_predictions):
        boxes = box_predictions * self._box_variance
        boxes = tf.concat(
            [
                boxes[:, :, :2] * anchor_boxes[:, :, 2:] + anchor_boxes[:, :, :2],
                tf.math.exp(boxes[:, :, 2:]) * anchor_boxes[:, :, 2:],
            ],
            axis=-1,
        )
        boxes_transformed = convert_to_corners(boxes)
        return boxes_transformed

    def call(self, images, predictions):
        image_shape = tf.cast(tf.shape(images), dtype=tf.float32)
        anchor_boxes = self._anchor_box.get_anchors(image_shape[1], image_shape[2])
        box_predictions = predictions[:, :, :4]
        cls_predictions = tf.nn.sigmoid(predictions[:, :, 4:])
        boxes = self._decode_box_predictions(anchor_boxes[None, ...], box_predictions)

        return tf.image.combined_non_max_suppression(
            tf.expand_dims(boxes, axis=2),
            cls_predictions,
            self.max_detections_per_class,
            self.max_detections,
            self.nms_iou_threshold,
            self.confidence_threshold,
            clip_boxes=False,
        )


if __name__ == '__main__':
    # Graph construction
    dim_h = 256
    g1 = dgl.graph(([0, 0, 0, 0, 0], [1, 2, 3, 4, 5]), num_nodes = 100)
    g1 = dgl.add_reverse_edges(g1)
    # load data to graph
    g1.ndata['x'] = tf.ones((g1.num_nodes(), 2))
    print(g1.ndata)
    h = tf.constant([[100.0 ,50.0]])
    g1.apply_nodes(lambda nodes: {'x' : h}, v=0)
    print(g1.ndata)
    # lay1 = glob.AvgPooling()
    # h_out = lay1(g1, g1.ndata['x'])
    # print(h_out)