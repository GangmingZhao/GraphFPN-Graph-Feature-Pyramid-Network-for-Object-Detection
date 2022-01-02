import os
import tensorflow_datasets as tfds
import init_path

from detection.utils.Label import *
from model.get_model import models, backbone, loss
from detection.utils.preprocess import *
from configs.configs import parse_configs
config = parse_configs()   
                                 
def main():
    # Initializing and compiling model
    model = models[config.Arch](config.num_classes, backbone[config.backbone])
    optimizer = tf.keras.optimizers.Adam()
    model.compile(loss=loss[config.loss](config.num_classes), optimizer=optimizer, run_eagerly = True)           # Only can be run eagerly cuz the Mapping between CNN GNN
    # Load the COCO2017 dataset using TensorFlow Datasets
    (train_dataset, val_dataset)= tfds.load("coco/2017", split=["train", "validation"],  data_dir=os.path.join(config.root_dir, "COCO"))
    
    # Setting up a tf.data pipeline
    train_dataset, val_dataset = pipeline(train_dataset, config.batch_size), pipeline(val_dataset, 1)

    callbacks_list = [
    tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(config.root_dir, config.weight, "weights" + "_epoch_{epoch}"),
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
        epochs=config.num_epochs,
        callbacks=callbacks_list,
        verbose=1,
        batch_size = config.batch_size
    )
if __name__ == "__main__":
    main()
