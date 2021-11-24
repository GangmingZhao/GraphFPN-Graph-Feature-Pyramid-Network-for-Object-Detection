import os,sys
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

def test():

if __name__ == "__main__":
    test()