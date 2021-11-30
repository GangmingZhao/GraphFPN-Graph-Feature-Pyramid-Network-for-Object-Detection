import pycocotools.coco as coco
from pycocotools.cocoeval import COCOeval
import numpy as np
import json
import os, sys

src_dir = os.path.dirname(os.path.realpath(__file__))
while not src_dir.endswith("Graph-FPN"):
    src_dir = os.path.dirname(src_dir) 
if src_dir not in sys.path:
    sys.path.append(src_dir)

class Info:
  def __init__(self):
      self.class_name = np.array([
      'person', 'bicycle', 'car', 'motorcycle', 'airplane',
      'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
      'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
      'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
      'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
      'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
      'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass',
      'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
      'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
      'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv',
      'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
      'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
      'scissors', 'teddy bear', 'hair drier', 'toothbrush'])
      self._valid_ids = np.array([
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 
        14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 
        24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 
        37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 
        48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 
        58, 59, 60, 61, 62, 63, 64, 65, 67, 70, 
        72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 
        82, 84, 85, 86, 87, 88, 89, 90])

class COCO:
  def __init__(self):
    self.annot_path = os.path.join(src_dir, "COCO/downloads/extracted/ZIP.images.cocodat.org_annotat_annotat_trainvaETqDbZAZXuH4hOcE2mME36rs_x8CP0m2ypPEqq5HAmg.zip/annotations/instances_val2017.json")
    self.coco = coco.COCO(self.annot_path)
    self.images = self.coco.getImgIds()
    self.num_samples = len(self.images)

  def _to_float(self, x):
    return float("{:.2f}".format(x))

  def convert_eval_format(self, bbox, image_id, cat_id, scores, num_bboxs, detections):
    for j in range(num_bboxs):
      bbox[j][2] -= bbox[j][0]
      bbox[j][3] -= bbox[j][1]
      bbox_out = list(map(self._to_float, bbox[j][0:4]))
      category_id = cat_id[j]
      score = scores[j]
      detection = {
          "image_id": int(image_id),
          "category_id": int(category_id),
          "bbox": bbox_out,
          "score": float("{:.2f}".format(score))
      }
      detections.append(detection)
    return detections


  def save_results(self, save_dir, detections):
    json.dump(detections, open('{}/results.json'.format(save_dir.replace('\\', '/')), 'w'))
  
  def run_eval(self, save_dir):
    coco_dets = self.coco.loadRes('{}/results.json'.format(save_dir.replace('\\', '/')))
    coco_eval = COCOeval(self.coco, coco_dets, "bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
