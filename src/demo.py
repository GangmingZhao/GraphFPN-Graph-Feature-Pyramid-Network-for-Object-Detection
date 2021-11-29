import os
import zipfile
import tensorflow as tf
import tensorflow_datasets as tfds
import init_path

from configs.configs import parse_configs
from detection.utils.Label import *
from detection.utils.preprocess import *
from model.network import DecodePredictions
from model.get_model import backbone, models
config = parse_configs()

def get_demo_data():
    url = "https://github.com/srihari-humbarwadi/datasets/releases/download/v0.1.0/data.zip"
    filename = os.path.join(config.root_dir, "data_demo", "data.zip")
    tf.keras.utils.get_file(filename, url)
    with zipfile.ZipFile(filename, "r") as z_fp:
        z_fp.extractall(os.path.join(config.root_dir,"data_demo/"))

def demo():
    get_demo_data()
    model = models[config.Arch](config.num_classes, backbone[config.backbone])

    # fine_tune_checkpoint_type
    ckpt = tf.train.Checkpoint(model)
    ckpt.restore(tf.train.latest_checkpoint(config.weight)).expect_partial()

    # Prepare image for demo
    val_dataset, dataset_info = tfds.load("coco/2017", 
                                          split="validation", 
                                          with_info=True, 
                                          data_dir=os.path.join(config.root_dir,"data_demo/data"), 
                                          download=False)
    int2str = dataset_info.features["objects"]["label"].int2str

    for sample in val_dataset.take(2):
        image = tf.cast(sample["image"], dtype=tf.float32)
        input_image, ratio_short, ratio_long = prepare_image(image)

        # Inference
        predictions = model(input_image)
        detections = DecodePredictions(confidence_threshold=0.5)(input_image, predictions)
        num_detections = detections.valid_detections[0]
        class_names = [int2str(int(x)) for x in detections.nmsed_classes[0][:num_detections]]
        visualize_detections(image,
                detections.nmsed_boxes[0][:num_detections].numpy(),
                class_names,
                detections.nmsed_scores[0][:num_detections].numpy(),
                ratio_short, ratio_long
        )


if __name__ == "__main__":
    demo()