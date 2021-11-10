import os
import tensorflow as tf
import numpy as np
import visualize

# tensorflow config - using one gpu and extending the GPU 
# memory region needed by the TensorFlow process
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True
# session = tf.Session(config=config)

from detection.datasets import coco, data_generator
from detection.datasets.utils import get_original_image
from detection.models.detectors import faster_rcnn

# load dataset
img_mean = (123.675, 116.28, 103.53)
# img_std = (58.395, 57.12, 57.375)
img_std = (1., 1., 1.)

train_dataset = coco.CocoDataSet('./COCO2017/', 'val',
                                 flip_ratio=0.5,
                                 pad_mode='fixed',
                                 mean=img_mean,
                                 std=img_std,
                                 scale=(800, 1024))

train_generator = data_generator.DataGenerator(train_dataset)

# display a sample
img, img_meta, bboxes, labels = train_dataset[0]
rgb_img = np.round(img + img_mean)
ori_img = get_original_image(img, img_meta, img_mean)

visualize.display_instances(rgb_img, bboxes, labels, train_dataset.get_categories())

# load model
model = faster_rcnn.FasterRCNN(num_classes=len(train_dataset.get_categories()))

batch_imgs = tf.Variable(np.expand_dims(img, 0), dtype=tf.float32)
batch_metas = tf.Variable(np.expand_dims(img_meta, 0), dtype=tf.float32)
batch_bboxes = tf.Variable(np.expand_dims(bboxes, 0), dtype=tf.float32)
batch_labels = tf.Variable(np.expand_dims(labels, 0), dtype=tf.int32)

_ = model((batch_imgs, batch_metas), training=False)

model.load_weights('weights/faster_rcnn.h5', by_name=True)

proposals = model.simple_test_rpn(img, img_meta)
res = model.simple_test_bboxes(img, img_meta, proposals)

visualize.display_instances(ori_img, res['rois'], res['class_ids'], train_dataset.get_categories(), scores=res['scores'])

# overfit a sample
optimizer = tf.keras.optimizers.SGD(1e-3, momentum=0.9, nesterov=True)

for batch in range(100):
    with tf.GradientTape() as tape:
        rpn_class_loss, rpn_bbox_loss, rcnn_class_loss, rcnn_bbox_loss = \
            model((batch_imgs, batch_metas, batch_bboxes, batch_labels), training=True)

        loss_value = rpn_class_loss + rpn_bbox_loss + rcnn_class_loss + rcnn_bbox_loss

    grads = tape.gradient(loss_value, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    print('batch', batch, '-', loss_value.numpy())


proposals = model.simple_test_rpn(img, img_meta)
res = model.simple_test_bboxes(img, img_meta, proposals)
visualize.display_instances(ori_img, res['rois'], res['class_ids'], train_dataset.get_categories(), scores=res['scores'])

# use tf.data
batch_size = 1

train_tf_dataset = tf.data.Dataset.from_generator(
    train_generator, (tf.float32, tf.float32, tf.float32, tf.int32))
train_tf_dataset = train_tf_dataset.padded_batch(
    batch_size, padded_shapes=([None, None, None], [None], [None, None], [None]))
train_tf_dataset = train_tf_dataset.prefetch(100).shuffle(100)

optimizer = tf.keras.optimizers.SGD(1e-3, momentum=0.9, nesterov=True)



# train model
epochs = 20

for epoch in range(epochs):
    loss_history = []
    for (batch, inputs) in enumerate(train_tf_dataset):
    
        batch_imgs, batch_metas, batch_bboxes, batch_labels = inputs
        with tf.GradientTape() as tape:
            rpn_class_loss, rpn_bbox_loss, rcnn_class_loss, rcnn_bbox_loss = \
                model((batch_imgs, batch_metas, batch_bboxes, batch_labels), training=True)

            loss_value = rpn_class_loss + rpn_bbox_loss + rcnn_class_loss + rcnn_bbox_loss

        grads = tape.gradient(loss_value, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        loss_history.append(loss_value.numpy())
        
        if batch % 100 == 0:
            print('epoch:', epoch, ', batch:', batch, ', loss:', np.mean(loss_history))