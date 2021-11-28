import os,sys
import tensorflow as tf
import tensorflow_datasets as tfds
import datetime
src_dir = os.path.dirname(os.path.realpath(__file__))
while not src_dir.endswith("Graph-FPN"):
    src_dir = os.path.dirname(src_dir)
if src_dir not in sys.path:
    sys.path.append(src_dir)
from detection.utils.Label import *
from model.network import get_backbone, RetinaNet, Graph_RetinaNet
from model.losses import RetinaNetLoss
from detection.utils.logger import *
from detection.utils.preprocess import *
from configs.configs import parse_configs

config = parse_configs()
# Setting up training parameters



def main(config):
    model_dir = "checkpoint/"
    # learning_rate_fn = tf.optimizers.schedules.PiecewiseConstantDecay(boundaries = config.lr_boundaries, values = config.lr)

    # Initializing and compiling model
    resnet101_backbone = get_backbone(101)
    loss_fn = RetinaNetLoss(config.num_classes)
    model = Graph_RetinaNet(config.num_classes, resnet101_backbone)

    optimizer = tf.keras.optimizers.Adam()
    # optimizer = tf.optimizers.Adam(learning_rate=learning_rate_fn)
    model.compile(loss=loss_fn, optimizer=optimizer, run_eagerly = True)           # Only can be run eagerly cuz the Mapping between CNN GNN
    # Load the COCO2017 dataset using TensorFlow Datasets
    (train_dataset, val_dataset), dataset_info = tfds.load("coco/2017", split=["train", "validation"], with_info=True, data_dir="COCO")
    # fig = tfds.visualization.show_examples(train_dataset, dataset_info)
    
    # Setting up a tf.data pipeline
    train_dataset, val_dataset = pipeline(train_dataset, config.batch_size), pipeline(val_dataset, 1)

    # Training the model
    train_steps_per_epoch = dataset_info.splits["train"].num_examples // config.batch_size    # iteration nb in each epoch
    val_steps_per_epoch = dataset_info.splits["validation"].num_examples // config.batch_size

    # freq = 10
    epochs = 10

    callbacks_list = [
    tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(model_dir, "weights" + "_epoch_{epoch}"),
        monitor='val_loss',
        save_best_only=True,
        save_weights_only=True,
        verbose=1,
        save_freq="epoch"
        )
    ]
    
    hist = model.fit(
        train_dataset.take(100),
        validation_data=val_dataset.take(50),
        # train_dataset,
        # validation_data=val_dataset,
        epochs=epochs,
        callbacks=callbacks_list,
        verbose=1,
        batch_size = config.batch_size
    )
if __name__ == "__main__":
    main(config)
