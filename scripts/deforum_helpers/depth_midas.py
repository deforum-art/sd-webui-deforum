import os
import cv2
import torch
import numpy as np
from .general_utils import download_file_with_checksum
from midas.dpt_depth import DPTDepthModel
from midas.transforms import Resize, NormalizeImage, PrepareForNet
import torchvision.transforms as T

class MidasDepth:
    def __init__(self, models_path, device, half_precision=True, midas_model_type='Midas-3-Hybrid'):
        if midas_model_type.lower() == 'midas-3.1-beitlarge':
            self.midas_model_filename = 'dpt_beit_large_512.pt'
            self.midas_model_checksum='66cbb00ea7bccd6e43d3fd277bd21002d8d8c2c5c487e5fcd1e1d70c691688a19122418b3ddfa94e62ab9f086957aa67bbec39afe2b41c742aaaf0699ee50b33'
            self.midas_model_url = 'https://github.com/isl-org/MiDaS/releases/download/v3_1/dpt_beit_large_512.pt'
            self.resize_px = 512
            self.backbone = 'beitl16_512'
        else:
            self.midas_model_filename = 'dpt_large-midas-2f21e586.pt'
            self.midas_model_checksum = 'fcc4829e65d00eeed0a38e9001770676535d2e95c8a16965223aba094936e1316d569563552a852d471f310f83f597e8a238987a26a950d667815e08adaebc06'
            self.midas_model_url = 'https://github.com/intel-isl/DPT/releases/download/1_0/dpt_large-midas-2f21e586.pt'
            self.resize_px = 384
            self.backbone = 'vitl16_384'
        self.device = device
        self.normalization = NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        self.midas_transform = T.Compose([
            Resize(self.resize_px, self.resize_px, resize_target=None, keep_aspect_ratio=True, ensure_multiple_of=32,
                   resize_method="minimal", image_interpolation_method=cv2.INTER_CUBIC),
            self.normalization,
            PrepareForNet()
        ])
        
        download_file_with_checksum(url=self.midas_model_url, expected_checksum=self.midas_model_checksum, dest_folder=models_path, dest_filename=self.midas_model_filename)
        
        self.load_midas_model(models_path, self.midas_model_filename)
        if half_precision:
            self.midas_model = self.midas_model.half()
            
    def load_midas_model(self, models_path, midas_model_filename):
        model_file = os.path.join(models_path, midas_model_filename)
        print(f"Loading MiDaS model from {midas_model_filename}...")
        self.midas_model = DPTDepthModel(
            path=model_file,
            backbone=self.backbone,
            non_negative=True,
        )
        self.midas_model.eval().to(self.device, memory_format=torch.channels_last if self.device == torch.device("cuda") else None)
                
    def predict(self, prev_img_cv2, half_precision):
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