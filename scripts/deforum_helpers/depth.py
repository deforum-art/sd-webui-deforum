import math, os, subprocess
import cv2
import hashlib
import numpy as np
import torch
import gc
import torchvision.transforms as T
from einops import rearrange, repeat
from PIL import Image
from infer import InferenceHelper
from midas.dpt_depth import DPTDepthModel
from midas.transforms import Resize, NormalizeImage, PrepareForNet
import torchvision.transforms.functional as TF
from .general_utils import checksum

class DepthModel():
    def __init__(self, device):
        self.adabins_helper = None
        self.depth_min = 1000
        self.depth_max = -1000
        self.device = device
        self.midas_model = None
        self.midas_transform = None
    
    def load_adabins(self, models_path):
        if not os.path.exists(os.path.join(models_path,'AdaBins_nyu.pt')):
            from basicsr.utils.download_util import load_file_from_url
            load_file_from_url(r"https://cloudflare-ipfs.com/ipfs/Qmd2mMnDLWePKmgfS8m6ntAg4nhV5VkUyAydYBp8cWWeB7/AdaBins_nyu.pt", models_path)
            if checksum(os.path.join(models_path,'AdaBins_nyu.pt')) != "643db9785c663aca72f66739427642726b03acc6c4c1d3755a4587aa2239962746410d63722d87b49fc73581dbc98ed8e3f7e996ff7b9c0d56d0fbc98e23e41a":
                raise Exception(r"Error while downloading AdaBins_nyu.pt. Please download from here: https://drive.google.com/file/d/1lvyZZbC9NLcS8a__YPcUP7rDiIpbRpoF and place in: " + models_path)
        self.adabins_helper = InferenceHelper(models_path=models_path, dataset='nyu', device=self.device)

    def load_midas(self, models_path, half_precision=True):
        if not os.path.exists(os.path.join(models_path, 'dpt_large-midas-2f21e586.pt')):
            from basicsr.utils.download_util import load_file_from_url
            load_file_from_url(r"https://github.com/intel-isl/DPT/releases/download/1_0/dpt_large-midas-2f21e586.pt", models_path)
            if checksum(os.path.join(models_path,'dpt_large-midas-2f21e586.pt')) != "fcc4829e65d00eeed0a38e9001770676535d2e95c8a16965223aba094936e1316d569563552a852d471f310f83f597e8a238987a26a950d667815e08adaebc06":
                raise Exception(r"Error while downloading dpt_large-midas-2f21e586.pt. Please download from here: https://github.com/intel-isl/DPT/releases/download/1_0/dpt_large-midas-2f21e586.pt and place in: " + models_path)

        self.midas_model = DPTDepthModel(
            path=f"{models_path}/dpt_large-midas-2f21e586.pt",
            backbone="vitl16_384",
            non_negative=True,
        )
        normalization = NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

        self.midas_transform = T.Compose([
            Resize(
                384, 384,
                resize_target=None,
                keep_aspect_ratio=True,
                ensure_multiple_of=32,
                resize_method="minimal",
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            normalization,
            PrepareForNet()
        ])

        self.midas_model.eval()    
        if self.device == torch.device("cuda"):
            self.midas_model = self.midas_model.to(memory_format=torch.channels_last)
        if half_precision:
            self.midas_model = self.midas_model.half()
        self.midas_model.to(self.device)

    def predict(self, prev_img_cv2, anim_args, half_precision) -> torch.Tensor:
        w, h = prev_img_cv2.shape[1], prev_img_cv2.shape[0]

        # predict depth with AdaBins    
        use_adabins = anim_args.midas_weight < 1.0 and self.adabins_helper is not None
        if use_adabins:
            MAX_ADABINS_AREA = 500000
            MIN_ADABINS_AREA = 448*448

            # resize image if too large or too small
            img_pil = Image.fromarray(cv2.cvtColor(prev_img_cv2.astype(np.uint8), cv2.COLOR_RGB2BGR))
            image_pil_area = w*h
            resized = True
            if image_pil_area > MAX_ADABINS_AREA:
                scale = math.sqrt(MAX_ADABINS_AREA) / math.sqrt(image_pil_area)
                depth_input = img_pil.resize((int(w*scale), int(h*scale)), Image.LANCZOS) # LANCZOS is good for downsampling
                print(f"  resized to {depth_input.width}x{depth_input.height}")
            elif image_pil_area < MIN_ADABINS_AREA:
                scale = math.sqrt(MIN_ADABINS_AREA) / math.sqrt(image_pil_area)
                depth_input = img_pil.resize((int(w*scale), int(h*scale)), Image.BICUBIC)
                print(f"  resized to {depth_input.width}x{depth_input.height}")
            else:
                depth_input = img_pil
                resized = False

            # predict depth and resize back to original dimensions
            try:
                with torch.no_grad():
                    _, adabins_depth = self.adabins_helper.predict_pil(depth_input)
                if resized:
                    adabins_depth = TF.resize(
                        torch.from_numpy(adabins_depth), 
                        torch.Size([h, w]),
                        interpolation=TF.InterpolationMode.BICUBIC
                    )
                    adabins_depth = adabins_depth.cpu().numpy()
                adabins_depth = adabins_depth.squeeze()
            except:
                print(f"  exception encountered, falling back to pure MiDaS")
                use_adabins = False
            torch.cuda.empty_cache()

        if self.midas_model is not None:
            # convert image from 0->255 uint8 to 0->1 float for feeding to MiDaS
            img_midas = prev_img_cv2.astype(np.float32) / 255.0
            img_midas_input = self.midas_transform({"image": img_midas})["image"]

            # MiDaS depth estimation implementation
            sample = torch.from_numpy(img_midas_input).float().to(self.device).unsqueeze(0)
            if self.device == torch.device("cuda"):
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
            ).squeeze()
            midas_depth = midas_depth.cpu().numpy()
            torch.cuda.empty_cache()

            # MiDaS makes the near values greater, and the far values lesser. Let's reverse that and try to align with AdaBins a bit better.
            midas_depth = np.subtract(50.0, midas_depth)
            midas_depth = midas_depth / 19.0

            # blend between MiDaS and AdaBins predictions
            if use_adabins:
                depth_map = midas_depth*anim_args.midas_weight + adabins_depth*(1.0-anim_args.midas_weight)
            else:
                depth_map = midas_depth

            depth_map = np.expand_dims(depth_map, axis=0)
            depth_tensor = torch.from_numpy(depth_map).squeeze().to(self.device)
        else:
            depth_tensor = torch.ones((h, w), device=self.device)
        
        return depth_tensor

    def save(self, filename: str, depth: torch.Tensor):
        depth = depth.cpu().numpy()
        if len(depth.shape) == 2:
            depth = np.expand_dims(depth, axis=0)
        self.depth_min = min(self.depth_min, depth.min())
        self.depth_max = max(self.depth_max, depth.max())
        print(f"  depth min:{depth.min()} max:{depth.max()}")
        denom = max(1e-8, self.depth_max - self.depth_min)
        temp = rearrange((depth - self.depth_min) / denom * 255, 'c h w -> h w c')
        temp = repeat(temp, 'h w 1 -> h w c', c=3)
        Image.fromarray(temp.astype(np.uint8)).save(filename)
    
    def to(self, device):
        self.device = device
        self.midas_model.to(device)
        if self.adabins_helper is not None:
            self.adabins_helper.to(device)
        gc.collect()
        torch.cuda.empty_cache()
