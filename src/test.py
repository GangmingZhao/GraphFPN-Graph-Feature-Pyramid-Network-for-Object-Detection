import os
import tensorflow as tf
import tensorflow_datasets as tfds
import init_path

from detection.utils.Label import *
from model.network import DecodePredictions
from detection.datasets.coco import COCO, Info
from model.get_model import models, backbone
from detection.utils.preprocess import *
from configs.configs import parse_configs

config = parse_configs()
# Setting up training parameters

def test():
    (val_dataset, test_dataset), dataset_info = tfds.load("coco/2017", split=["validation", "test"], with_info=True, data_dir=os.path.join(config.root_dir, "COCO"), download = False) 
    int2str = dataset_info.features["objects"]["label"].int2str
    
    model = models[config.Arch](config.num_classes, backbone[config.backbone])
    ckpt = tf.train.Checkpoint(model)
    ckpt.restore(tf.train.latest_checkpoint(config.weight)).expect_partial()
    eval_coco = COCO()
    res = []

    for sample in val_dataset.take(100):
        # Prepare data
        image = tf.cast(sample["image"], dtype=tf.float32)
        image_id = sample["image/id"]
        input_image, ratio_short, ratio_long = prepare_image(image)

        # Inferrence
        predictions = model(input_image)
        detections = DecodePredictions(confidence_threshold=0.5)(input_image, predictions)  
        nmsed_bbox, nmsed_scores, nmsed_classes, valid_detections = detections
        num_detections = valid_detections[0]
        class_names = [int2str(int(x)) for x in nmsed_classes[0][:num_detections]]
        bbox = np.array([x for x in nmsed_bbox[0][:num_detections]])                     # Attention, Here we have bboxs's coordinate in (224, 224) size, need to resize it to original image shape
        reshape_bbox(image, bbox, ratio_long, ratio_short)
        scores = [x for x in nmsed_scores[0][:num_detections]]

        # Get class id in format COCO
        class_ids = []
        for name in class_names:
            indice = np.where(name == Info().class_name)
            id = Info()._valid_ids[indice].item()
            class_ids.append(id)
        eval_coco.convert_eval_format(bbox, image_id, class_ids, scores, num_detections, res)
    eval_coco.save_results(config.result_dir, res)
    eval_coco.run_eval(config.result_dir)
    
if __name__ == "__main__":
    test()