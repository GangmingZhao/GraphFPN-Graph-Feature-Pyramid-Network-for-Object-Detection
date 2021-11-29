import os.path as osp
import sys

def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)

src_dir = osp.dirname(__file__)

# Add src to PYTHONPATH
add_path(src_dir)