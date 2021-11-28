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
from model.network import get_backbone, RetinaNet, Graph_RetinaNet
from detection.utils.preprocess import *
from model.network import DecodePredictions
from configs.configs import parse_configs
config = parse_configs()
    
def get_demo_data():
    url = "https://github.com/srihari-humbarwadi/datasets/releases/download/v0.1.0/data.zip"
    filename = os.path.join(os.getcwd(), "data_demo", "data.zip")
    tf.keras.utils.get_file(filename, url)

    with zipfile.ZipFile(filename, "r") as z_fp:
        z_fp.extractall("data_demo/")


def demo():
    get_demo_data()
    weights_dir = "data_demo/data"
    resnet50_backbone = get_backbone(50)
    model = RetinaNet(config.num_classes, resnet50_backbone)
    # fine_tune_checkpoint_type
    ckpt = tf.train.Checkpoint(model)
    ckpt.restore(tf.train.latest_checkpoint(weights_dir)).expect_partial()

    # Building inference model 
    image = tf.keras.Input(shape=[224, 224, 3], batch_size = 1, name="image")
    predictions = model(image, training = False)
    detections = DecodePredictions(confidence_threshold=0.5)(image, predictions)
    inference_model = tf.keras.Model(inputs=image, outputs=detections)
    # inference_model.summary()

    val_dataset, dataset_info = tfds.load("coco/2017", split="validation", with_info=True, data_dir="data_demo/data", download=False)
    int2str = dataset_info.features["objects"]["label"].int2str

    for sample in val_dataset.take(2):
        image = tf.cast(sample["image"], dtype=tf.float32)
        input_image, ratio_short, ratio_long = prepare_image(image)
       
        detections = inference_model.predict(input_image)
        num_detections = detections.valid_detections[0]
        class_names = [int2str(int(x)) for x in detections.nmsed_classes[0][:num_detections]]
        visualize_detections(
            image,
            detections.nmsed_boxes[0][:num_detections],
            class_names,
            detections.nmsed_scores[0][:num_detections],
            ratio_short, ratio_long
        )

if __name__ == "__main__":
    demo()