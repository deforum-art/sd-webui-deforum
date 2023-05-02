import math, os, subprocess
import cv2
import hashlib
import numpy as np
import torch
import gc
import torchvision.transforms as T
from einops import rearrange, repeat
from PIL import Image
from modules import lowvram, devices
from modules.shared import opts, cmd_opts
from .depth_midas import MidasDepth
from .depth_zoe import ZoeDepth
from .depth_leres import LeReSDepth
from .depth_adabins import AdaBinsModel

class DepthModel:
    _instance = None

    def __new__(cls, *args, **kwargs):
            keep_in_vram = kwargs.get('keep_in_vram', False)
            depth_algorithm = kwargs.get('depth_algorithm', 'Midas')
            Width, Height = kwargs.get('Width', 512), kwargs.get('Height', 512)
            midas_weight = kwargs.get('midas_weight', 0.2)
            model_switched = cls._instance and cls._instance.depth_algorithm != depth_algorithm
            resolution_changed = cls._instance and (cls._instance.Width != Width or cls._instance.Height != Height)

            if cls._instance is None or (not keep_in_vram and not hasattr(cls._instance, 'midas_model')) or model_switched or resolution_changed:
                cls._instance = super().__new__(cls)
                cls._instance._initialize(models_path=args[0], device=args[1], half_precision=not cmd_opts.no_half, keep_in_vram=keep_in_vram, depth_algorithm=depth_algorithm, Width=Width, Height=Height, midas_weight=midas_weight)
            elif cls._instance.should_delete and keep_in_vram:
                cls._instance._initialize(models_path=args[0], device=args[1], half_precision=not cmd_opts.no_half, keep_in_vram=keep_in_vram, depth_algorithm=depth_algorithm, Width=Width, Height=Height, midas_weight=midas_weight)
            cls._instance.should_delete = not keep_in_vram
            return cls._instance

    def _initialize(self, models_path, device, half_precision=not cmd_opts.no_half, keep_in_vram=False, depth_algorithm='midas', Width=512, Height=512, midas_weight=1.0):
        self.keep_in_vram = keep_in_vram
        self.Width = Width
        self.Height = Height
        self.depth_min = 1000
        self.depth_max = -1000
        self.device = device
        self.depth_algorithm = depth_algorithm
        self.adabins_helper = None

        if self.depth_algorithm.lower() == 'zoe':
            self.zoe_depth = ZoeDepth(self.Width, self.Height)
        elif self.depth_algorithm.lower() == 'leres':
            self.leres_depth = LeReSDepth(width=448, height=448, models_path=models_path, checkpoint_name='res101.pth', backbone='resnext101')
        elif self.depth_algorithm.lower() == 'adabins':
            self.adabins_model = AdaBinsModel(models_path, keep_in_vram=keep_in_vram)
            self.adabins_helper = self.adabins_model.adabins_helper
        elif self.depth_algorithm.lower().startswith('midas'): # Midas or Midas+AdaBins
            self.midas_depth = MidasDepth(models_path, device, half_precision=half_precision)
            if self.depth_algorithm.lower() == 'midas+adabins' and midas_weight < 1.0:
                self.adabins_model = AdaBinsModel(models_path, keep_in_vram=keep_in_vram)
                self.adabins_helper = self.adabins_model.adabins_helper
            
    def predict(self, prev_img_cv2, midas_weight, half_precision) -> torch.Tensor:

        img_pil = Image.fromarray(cv2.cvtColor(prev_img_cv2.astype(np.uint8), cv2.COLOR_RGB2BGR))

        if self.depth_algorithm.lower() == 'zoe':
            depth_tensor = self.zoe_depth.predict(img_pil).to(self.device)
        elif self.depth_algorithm.lower() == 'leres':
            depth_tensor = self.leres_depth.predict(prev_img_cv2.astype(np.float32) / 255.0)
        elif self.depth_algorithm.lower() == 'adabins':
            use_adabins, adabins_depth = AdaBinsModel._instance.predict(img_pil, prev_img_cv2)
            depth_tensor = torch.tensor(adabins_depth)
            if use_adabins is False:
                raise Exception("Error getting depth from AdaBins") # TODO: fallback to something else maybe?
        elif self.depth_algorithm.lower().startswith('midas'):
            depth_tensor = self.midas_depth.predict(prev_img_cv2, half_precision)
            if self.depth_algorithm.lower() == 'midas+adabins' and midas_weight < 1.0:
                use_adabins, adabins_depth = AdaBinsModel._instance.predict(img_pil, prev_img_cv2)
                if use_adabins: # if there was no error in getting the adabins depth, align midas with adabins
                    depth_tensor = self.blend_and_align_with_adabins(depth_tensor, adabins_depth, midas_weight)
        else: # Unknown!
            raise Exception(f"Unknown depth_algorithm passed to depth.predict function: {self.depth_algorithm}")

        return depth_tensor

    def blend_and_align_with_adabins(self, depth_tensor, adabins_depth, midas_weight):
        depth_tensor = torch.subtract(50.0, depth_tensor) / 19.0 # align midas depth with adabins depth. Original alignment code from Disco Diffusion
        blended_depth_map = (depth_tensor.cpu().numpy() * midas_weight + adabins_depth * (1.0 - midas_weight))
        depth_tensor = torch.from_numpy(np.expand_dims(blended_depth_map, axis=0)).squeeze().to(self.device)
        self.debug_print(f"Blended Midas Depth with AdaBins Depth")
        return depth_tensor
        
    def to(self, device):
        self.device = device
        if self.depth_algorithm.lower() == 'zoe':
            self.zoe_depth.zoe.to(device)
        elif self.depth_algorithm.lower() == 'leres':
            self.leres_depth.to(device)
        elif self.depth_algorithm.lower().startswith('midas'):
            self.midas_depth.to(device)
        if hasattr(self, 'adabins_model'):
            self.adabins_model.to(device)
        gc.collect()
        torch.cuda.empty_cache()
        
    def to_image(self, depth: torch.Tensor):
        depth = depth.cpu().numpy()
        depth = np.expand_dims(depth, axis=0) if len(depth.shape) == 2 else depth
        self.depth_min, self.depth_max = min(self.depth_min, depth.min()), max(self.depth_max, depth.max())
        denom = max(1e-8, self.depth_max - self.depth_min)
        temp = rearrange((depth - self.depth_min) / denom * 255, 'c h w -> h w c')
        return Image.fromarray(repeat(temp, 'h w 1 -> h w c', c=3).astype(np.uint8))

    def save(self, filename: str, depth: torch.Tensor):
        self.to_image(depth).save(filename)

    def delete_model(self):
        for attr in ['zoe_depth', 'leres_depth']:
            if hasattr(self, attr):
                getattr(self, attr).delete()
                delattr(self, attr)

        if hasattr(self, 'midas_depth'):
            del self.midas_depth

        if hasattr(self, 'adabins_model'):
            self.adabins_model.delete_model()

        gc.collect()
        torch.cuda.empty_cache()
        devices.torch_gc()
 
    def debug_print(self, message):
        DEBUG_MODE = opts.data.get("deforum_debug_mode_enabled", False)
        if DEBUG_MODE:
            print(message)