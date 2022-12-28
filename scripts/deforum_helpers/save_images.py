from typing import List, Tuple
from einops import rearrange
import numpy as np, os, torch
from PIL import Image
from torchvision.utils import make_grid
import time


def get_output_folder(output_path, batch_folder):
    out_path = os.path.join(output_path,time.strftime('%Y-%m'))
    if batch_folder != "":
        out_path = os.path.join(out_path, batch_folder)
    os.makedirs(out_path, exist_ok=True)
    return out_path
