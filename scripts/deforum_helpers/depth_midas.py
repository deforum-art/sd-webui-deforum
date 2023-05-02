import os
import cv2
import torch
import gc
import numpy as np
from basicsr.utils.download_util import load_file_from_url
from .general_utils import checksum
from midas.dpt_depth import DPTDepthModel
from midas.transforms import Resize, NormalizeImage, PrepareForNet
import torchvision.transforms as T


class MidasDepth:
    def __init__(self, models_path, device, half_precision=True):
        self.device = device
        self.normalization = NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        self.midas_transform = T.Compose([
            Resize(384, 384, resize_target=None, keep_aspect_ratio=True, ensure_multiple_of=32,
                   resize_method="minimal", image_interpolation_method=cv2.INTER_CUBIC),
            self.normalization,
            PrepareForNet()
        ])
        midas_model_filename = 'dpt_large-midas-2f21e586.pt'
        self.check_and_download_midas_model(models_path, midas_model_filename)
        self.load_midas_model(models_path, midas_model_filename)
        if half_precision:
            self.midas_model = self.midas_model.half()
            
    def load_midas_model(self, models_path, midas_model_filename):
        model_file = os.path.join(models_path, midas_model_filename)
        print("Loading MiDaS model...")
        self.midas_model = DPTDepthModel(
            path=model_file,
            backbone="vitl16_384",
            non_negative=True,
        )
        self.midas_model.eval().to(self.device, memory_format=torch.channels_last if self.device == torch.device("cuda") else None)
         
    def check_and_download_midas_model(self, models_path, midas_model_filename):
        model_file = os.path.join(models_path, midas_model_filename)
        if not os.path.exists(model_file):
            load_file_from_url(r"https://github.com/intel-isl/DPT/releases/download/1_0/dpt_large-midas-2f21e586.pt", models_path)
            if checksum(model_file) != "fcc4829e65d00eeed0a38e9001770676535d2e95c8a16965223aba094936e1316d569563552a852d471f310f83f597e8a238987a26a950d667815e08adaebc06":
                raise Exception(f"Error while downloading dpt_large-midas-2f21e586.pt. Please download from here: https://github.com/intel-isl/DPT/releases/download/1_0/dpt_large-midas-2f21e586.pt and place in: {models_path}")
                
    def predict_depth(self, prev_img_cv2, half_precision):
        img_midas = prev_img_cv2.astype(np.float32) / 255.0
        img_midas_input = self.midas_transform({"image": img_midas})["image"]
        sample = torch.from_numpy(img_midas_input).float().to(self.device).unsqueeze(0)

        if self.device.type == "cuda" or self.device.type == "mps":
            sample = sample.to(memory_format=torch.channels_last)
            if half_precision:
                sample = sample.half()

        with torch.no_grad():
            midas_depth = self.midas_model.forward(sample)
        midas_depth = torch.nn.functional.interpolate(
            midas_depth.unsqueeze(1),
            size=img_midas.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze().cpu().numpy()

        torch.cuda.empty_cache()
        depth_tensor = torch.from_numpy(np.expand_dims(midas_depth, axis=0)).squeeze().to(self.device)

        return depth_tensor
        
    def to(self, device):
        self.device = device
        self.midas_model = self.midas_model.to(device, memory_format=torch.channels_last if device == torch.device("cuda") else None)