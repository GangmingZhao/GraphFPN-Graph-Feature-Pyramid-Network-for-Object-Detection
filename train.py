import os,sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import tensorflow as tf
import tensorflow_datasets as tfds

src_dir = os.path.dirname(os.path.realpath(__file__))
while not src_dir.endswith("Graph-FPN"):
    src_dir = os.path.dirname(src_dir)
if src_dir not in sys.path:
    sys.path.append(src_dir)

from detection.utils.Label import *
from model.network import get_backbone, RetinaNet, Graph_RetinaNet
from model.losses import RetinaNetLoss
from detection.utils.preprocess import *
from configs.configs import parse_configs
from tqdm.keras import TqdmCallback

config = parse_configs()
# Setting up training parameters



def main(config):
    model_dir = "checkpoints/"
    label_encoder = LabelEncoder()

    learning_rate_fn = tf.optimizers.schedules.PiecewiseConstantDecay(boundaries = config.lr_boundaries, values = config.lr)

    # Initializing and compiling model
    resnet101_backbone = get_backbone(50)
    loss_fn = RetinaNetLoss(config.num_classes)
    model = Graph_RetinaNet(config.num_classes, resnet101_backbone)

    optimizer = tf.optimizers.SGD(learning_rate=learning_rate_fn, momentum=0.9)
    model.compile(loss=loss_fn, optimizer=optimizer, run_eagerly = True)           # Only can be run eagerly cuz the Mapping between CNN GNN

    # Load the COCO2017 dataset using TensorFlow Datasets
    (train_dataset, val_dataset), dataset_info = tfds.load("coco/2017", split=["train", "validation"], with_info=True, data_dir="COCO")
    # fig = tfds.visualization.show_examples(train_dataset, dataset_info)
    
    # Setting up a tf.data pipeline
    train_dataset, val_dataset = pipeline(train_dataset, config.batch_size), pipeline(val_dataset, 1)
    # Training the model

    # Uncomment the following lines, when training on full dataset
    train_steps_per_epoch = dataset_info.splits["train"].num_examples // config.batch_size    # iteration nb in each epoch
    val_steps_per_epoch = dataset_info.splits["validation"].num_examples // config.batch_size

    freq = 3 * train_steps_per_epoch
    epochs = 100

    callbacks_list = [
    tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(model_dir, "weights" + "_epoch_{epoch}"),
        monitor="val_loss",
        save_best_only=True,
        save_weights_only=True,
        verbose=1,
        save_freq=freq
        )
    ]

    model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=epochs,
        callbacks=callbacks_list,
        verbose=1,
        batch_size = config.batch_size
    )

if __name__ == "__main__":
    main(config)
