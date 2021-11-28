import os,sys
import numpy as np
import zipfile
import pdb
import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow_datasets as tfds
import pdb

src_dir = os.path.dirname(os.path.realpath(__file__))
while not src_dir.endswith("Graph-FPN"):
    src_dir = os.path.dirname(src_dir)
if src_dir not in sys.path:
    sys.path.append(src_dir)

from detection.utils.Label import *
from model.network import get_backbone, Graph_RetinaNet
from detection.utils.preprocess import *
from model.network import DecodePredictions
from configs.configs import parse_configs
config = parse_configs()


def g_demo():
    weights_dir = "checkpoint"
    resnet101_backbone = get_backbone(101)
    model = Graph_RetinaNet(config.num_classes, resnet101_backbone)

    # fine_tune_checkpoint_type
    ckpt = tf.train.Checkpoint(model)
    ckpt.restore(tf.train.latest_checkpoint(weights_dir)).expect_partial()

    # Prepare one image for demo
    val_dataset, dataset_info = tfds.load("coco/2017", split="validation", with_info=True, data_dir="data_demo/data", download=False)
    Input = [sample for sample in val_dataset.take(1)][0]
    image = tf.cast(Input["image"], dtype=tf.float32)
    input_image, ratio_short, ratio_long = prepare_image(image)

    # Inference
    predictions = model(input_image)
    detections = DecodePredictions(confidence_threshold=0.5)(input_image, predictions)
    num_detections = detections.valid_detections[0]
    int2str = dataset_info.features["objects"]["label"].int2str
    class_names = [int2str(int(x)) for x in detections.nmsed_classes[0][:num_detections]]
    visualize_detections(image,
            detections.nmsed_boxes[0][:num_detections].numpy(),
            class_names,
            detections.nmsed_scores[0][:num_detections].numpy(),
            ratio_short, ratio_long
        )


if __name__ == "__main__":
    g_demo()