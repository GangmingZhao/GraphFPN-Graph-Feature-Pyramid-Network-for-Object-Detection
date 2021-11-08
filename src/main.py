import tensorflow as tf
import numpy as np
import os
import sys
import dgl 

src_dir = os.path.dirname(os.path.realpath(__file__))
while not src_dir.endswith("src"):
    src_dir = os.path.dirname(src_dir)
if src_dir not in sys.path:
    sys.path.append(src_dir)

from model import Graph

def main():
    a = 0


if __name__ == '__main__':
    main()