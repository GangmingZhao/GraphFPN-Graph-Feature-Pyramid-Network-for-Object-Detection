import tensorflow as tf
from .bbox import *
from .Label import LabelEncoder
from .compute_IoU import visualize_detections
import pdb

"""Preprocessing data
Preprocessing the images involves two steps:
Resizing the image: Images are resized such that the shortest size is equal to 224 px for resnet, 
Applying augmentation: Random scale jittering and random horizontal flipping are the only augmentations applied to the images.
Along with the images, bounding boxes are rescaled and flipped if required."""


def random_flip_horizontal(image, boxes):
    """Flips image and boxes horizontally with 50% chance

    Arguments:
      image: A 3-D tensor of shape `(height, width, channels)` representing an
        image.
      boxes: A tensor with shape `(num_boxes, 4)` representing bounding boxes,
        having normalized coordinates.

    Returns:
      Randomly flipped image and boxes
    """
    if tf.random.uniform(()) > 0.5:
        image = tf.image.flip_left_right(image)
        boxes = tf.stack(
            [1 - boxes[:, 2], boxes[:, 1], 1 - boxes[:, 0], boxes[:, 3]], axis=-1
        )
    return image, boxes



def resize_and_pad_image(image, reshape_size = 224.0):
    """Resizes and pads image while preserving aspect ratio.

    Arguments:
      image: A 3-D tensor of shape `(height, width, channels)` representing an
        image.

    Returns:
      image: Resized image.
      image.shape: Resized shape
      ratio_short: The scaling factor used to resize the short sides of image
      ratio_long: The scaling factor used to resize the long sides of image
    """
    image_shape = tf.cast(tf.shape(image)[:2], dtype=tf.float32)
    ratio_short = reshape_size / tf.reduce_min(image_shape)
    ratio_long = reshape_size / tf.reduce_max(image_shape)

    image = tf.image.resize(image, tf.cast(tf.constant([224, 224]), dtype=tf.int32))
    return image, image.shape, ratio_short, ratio_long




def preprocess_data(sample):
    """Applies preprocessing step to a single sample

    Arguments:
      sample: A dict representing a single training sample.

    Returns:
      image: Resized and padded image with random horizontal flipping applied.
      bbox: Bounding boxes with the shape `(num_objects, 4)` where each box is
        of the format `[x, y, width, height]`.
      class_id: An tensor representing the class id of the objects, having
        shape `(num_objects,)`.
    """

    image = sample["image"]
    bbox = swap_xy(sample["objects"]["bbox"])
    class_id = tf.cast(sample["objects"]["label"], dtype=tf.int32)
    image, bbox = random_flip_horizontal(image, bbox)
    image, image_shape, _, __ = resize_and_pad_image(image)

    bbox = tf.stack(
        [
            bbox[:, 0] * image_shape[1],
            bbox[:, 1] * image_shape[0],
            bbox[:, 2] * image_shape[1],
            bbox[:, 3] * image_shape[0],
        ],
        axis=-1,
    )
    bbox = convert_to_xywh(bbox)
    return image, bbox, class_id


def prepare_image(image):
    image, _, ratio_short, ratio_long = resize_and_pad_image(image)
    image = tf.keras.applications.resnet.preprocess_input(image)
    return tf.expand_dims(image, axis=0), ratio_short, ratio_long


def pipeline(dataset, batch_size):
    """To ensure that the model is fed with data efficiently we will be using tf.data API to create our input pipeline. The input pipeline consists for the following major processing steps:
    Apply the preprocessing function to the samples
    Create batches with fixed batch size. Since images in the batch can have different dimensions, and can also have different number of objects, we use padded_batch to the add the necessary padding to create rectangular tensors
    Create targets for each sample in the batch using LabelEncoder"""
    autotune = tf.data.AUTOTUNE                                                                                  # make sure that number of files readed is bigger or equal than batch size
    dataset = dataset.map(preprocess_data, num_parallel_calls=autotune)
    dataset = dataset.shuffle(4 * batch_size)                                                                    # randomly samples elements from this buffer
    dataset = dataset.padded_batch(batch_size=batch_size, padding_values=(0.0, 1e-8, -1), drop_remainder=True)   # padded_batch(bsz, pad_shape, pad_val)
    dataset = dataset.map(LabelEncoder().encode_batch, num_parallel_calls=autotune)
    dataset = dataset.apply(tf.data.experimental.ignore_errors())
    dataset = dataset.prefetch(autotune)
    return dataset