import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from .bbox import convert_to_corners


def compute_iou(boxes1, boxes2):
    """Computing pairwise Intersection Over Union (IOU)
    As we will see later in the example, we would be assigning ground truth boxes to anchor boxes based on the extent of overlapping. 
    This will require us to calculate the Intersection Over Union (IOU) between all the anchor boxes and ground truth boxes pairs.
    
    Computes pairwise IOU matrix for given two sets of boxes

    Arguments:
      boxes1: A tensor with shape `(N, 4)` representing bounding boxes
        where each box is of the format `[x, y, width, height]`.
        boxes2: A tensor with shape `(M, 4)` representing bounding boxes
        where each box is of the format `[x, y, width, height]`.

    Returns:
      pairwise IOU matrix with shape `(N, M)`, where the value at ith row
        jth column holds the IOU between ith box and jth box from
        boxes1 and boxes2 respectively.
    """
    boxes1_corners = convert_to_corners(boxes1)
    boxes2_corners = convert_to_corners(boxes2)
    lu = tf.maximum(boxes1_corners[:, None, :2], boxes2_corners[:, :2])   # left up, None is for add a dimension, boxes2 could broadcast directly
    rd = tf.minimum(boxes1_corners[:, None, 2:], boxes2_corners[:, 2:])   # right down
    intersection = tf.maximum(0.0, rd - lu)
    intersection_area = intersection[:, :, 0] * intersection[:, :, 1]
    boxes1_area = boxes1[:, 2] * boxes1[:, 3]
    boxes2_area = boxes2[:, 2] * boxes2[:, 3]
    union_area = tf.maximum(
        boxes1_area[:, None] + boxes2_area - intersection_area, 1e-8
    )
    return tf.clip_by_value(intersection_area / union_area, 0.0, 1.0)


def reshape_bbox(image, boxes, ratio_long, ratio_short):
    """Reshape bboxes to original shape"""
    for i in range(boxes.shape[0]):
        if image.shape[0] < image.shape[1]:
            boxes[i][2] /= ratio_long
            boxes[i][0] /= ratio_long
            boxes[i][3] /= ratio_short
            boxes[i][1] /= ratio_short
        else:
            boxes[i][2] /= ratio_short
            boxes[i][0] /= ratio_short
            boxes[i][3] /= ratio_long
            boxes[i][1] /= ratio_long


def visualize_detections(
    image, boxes, classes, scores, ratio_short, ratio_long, figsize=(7, 7), linewidth=1, color=[0, 0, 1]
):
    """Visualize Detections"""
    # Reshape bbox sizes with corresponding ratio depends on original image shape
    reshape_bbox(image, boxes, ratio_long, ratio_short)
    image = np.array(image, dtype=np.uint8)
    plt.figure(figsize=figsize)
    plt.axis("off")
    plt.imshow(image)
    ax = plt.gca()
    for box, _cls, score in zip(boxes, classes, scores):
        text = "{}: {:.2f}".format(_cls, score)
        x1, y1, x2, y2 = box
        w, h = x2 - x1, y2 - y1
        patch = plt.Rectangle(
            [x1, y1], w, h, fill=False, edgecolor=color, linewidth=linewidth
        )
        ax.add_patch(patch)
        ax.text(
            x1,
            y1,
            text,
            bbox={"facecolor": color, "alpha": 0.4},
            clip_box=ax.clipbox,
            clip_on=True,
        )
    plt.show()
    return ax
