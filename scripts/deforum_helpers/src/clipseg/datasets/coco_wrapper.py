import pickle
from types import new_class
import torch
import numpy as np
import os
import json

from os.path import join, dirname, isdir, isfile, expanduser, realpath, basename
from random import shuffle, seed as set_seed
from PIL import Image

from itertools import combinations
from torchvision import transforms
from torchvision.transforms.transforms import Resize

from datasets.utils import blend_image_segmentation
from general_utils import get_from_repository

COCO_CLASSES = {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus', 6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light', 10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 13: 'bench', 14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 20: 'elephant', 21: 'bear', 22: 'zebra', 23: 'giraffe', 24: 'backpack', 25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase', 29: 'frisbee', 30: 'skis', 31: 'snowboard', 32: 'sports ball', 33: 'kite', 34: 'baseball bat', 35: 'baseball glove', 36: 'skateboard', 37: 'surfboard', 38: 'tennis racket', 39: 'bottle', 40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl', 46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange', 50: 'broccoli', 51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut', 55: 'cake', 56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed', 60: 'dining table', 61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse', 65: 'remote', 66: 'keyboard', 67: 'cell phone', 68: 'microwave', 69: 'oven', 70: 'toaster', 71: 'sink', 72: 'refrigerator', 73: 'book', 74: 'clock', 75: 'vase', 76: 'scissors', 77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush'}

class COCOWrapper(object):

    def __init__(self, split, fold=0, image_size=400, aug=None, mask='separate', negative_prob=0,
                 with_class_label=False):
        super().__init__()

        self.mask = mask
        self.with_class_label = with_class_label
        self.negative_prob = negative_prob

        from third_party.hsnet.data.coco import DatasetCOCO

        get_from_repository('COCO-20i', ['COCO-20i.tar'])

        foldpath = join(dirname(__file__), '../third_party/hsnet/data/splits/coco/%s/fold%d.pkl')

        def build_img_metadata_classwise(self):
            with open(foldpath % (self.split, self.fold), 'rb') as f:
                img_metadata_classwise = pickle.load(f)
            return img_metadata_classwise


        DatasetCOCO.build_img_metadata_classwise = build_img_metadata_classwise
        # DatasetCOCO.read_mask = read_mask
        
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

        self.coco = DatasetCOCO(expanduser('~/datasets/COCO-20i/'), fold, transform, split, 1, False)
    
        self.all_classes = [self.coco.class_ids]
        self.coco.base_path = join(expanduser('~/datasets/COCO-20i'))

    def __len__(self):
        return len(self.coco)

    def __getitem__(self, i):
        sample = self.coco[i]

        label_name = COCO_CLASSES[int(sample['class_id'])]

        img_s, seg_s = sample['support_imgs'][0], sample['support_masks'][0]

        if self.negative_prob > 0 and torch.rand(1).item() < self.negative_prob:
            new_class_id = sample['class_id']
            while new_class_id == sample['class_id']:
                sample2 = self.coco[torch.randint(0, len(self), (1,)).item()]
                new_class_id = sample2['class_id']
            img_s = sample2['support_imgs'][0]
            seg_s = torch.zeros_like(seg_s)

        mask = self.mask
        if mask == 'separate':
            supp = (img_s, seg_s)
        elif mask == 'text_label':
            # DEPRECATED
            supp = [int(sample['class_id'])]
        elif mask == 'text':
            supp = [label_name]      
        else:
            if mask.startswith('text_and_'):
                mask = mask[9:]
                label_add = [label_name]
            else:
                label_add = []

            supp = label_add + blend_image_segmentation(img_s, seg_s, mode=mask)

        if self.with_class_label:
            label = (torch.zeros(0), sample['class_id'],)
        else:
            label = (torch.zeros(0), )

        return (sample['query_img'],) + tuple(supp), (sample['query_mask'].unsqueeze(0),) + label