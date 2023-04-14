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
from modules import lowvram, devices
from modules.shared import opts
from .ZoeDepth import ZoeDepth

class MidasModel:
    _instance = None

    def __new__(cls, *args, **kwargs):
        keep_in_vram = kwargs.get('keep_in_vram', False)
        use_zoe_depth = kwargs.get('use_zoe_depth', False)
        Width = kwargs.get('Width', 512)
        Height = kwargs.get('Height', 512)
        model_switched = cls._instance and cls._instance.use_zoe_depth != use_zoe_depth
        resolution_changed = cls._instance and (cls._instance.Width != Width or cls._instance.Height != Height)

        if cls._instance is None or (not keep_in_vram and not hasattr(cls._instance, 'midas_model')) or model_switched or resolution_changed:
            cls._instance = super().__new__(cls)
            cls._instance._initialize(models_path=args[0], device=args[1], half_precision=True, keep_in_vram=keep_in_vram, use_zoe_depth=use_zoe_depth, Width=Width, Height=Height)
        elif cls._instance.should_delete and keep_in_vram:
            cls._instance._initialize(models_path=args[0], device=args[1], half_precision=True, keep_in_vram=keep_in_vram, use_zoe_depth=use_zoe_depth, Width=Width, Height=Height)
        cls._instance.should_delete = not keep_in_vram
        return cls._instance

    def _initialize(self, models_path, device, half_precision=True, keep_in_vram=False, use_zoe_depth=False, Width=512, Height=512):
        self.keep_in_vram = keep_in_vram
        self.Width = Width
        self.Height = Height
        self.adabins_helper = None
        self.depth_min = 1000
        self.depth_max = -1000
        self.device = device
        self.use_zoe_depth = use_zoe_depth
        
        if self.use_zoe_depth:
            self.zoe_depth = ZoeDepth(self.Width, self.Height)
        if not self.use_zoe_depth:
            model_file = os.path.join(models_path, 'dpt_large-midas-2f21e586.pt')
            if not os.path.exists(model_file):
                from basicsr.utils.download_util import load_file_from_url
                load_file_from_url(r"https://github.com/intel-isl/DPT/releases/download/1_0/dpt_large-midas-2f21e586.pt", models_path)
                if checksum(model_file) != "fcc4829e65d00eeed0a38e9001770676535d2e95c8a16965223aba094936e1316d569563552a852d471f310f83f597e8a238987a26a950d667815e08adaebc06":
                    raise Exception(r"Error while downloading dpt_large-midas-2f21e586.pt. Please download from here: https://github.com/intel-isl/DPT/releases/download/1_0/dpt_large-midas-2f21e586.pt and place in: " + models_path)

            if not self.keep_in_vram or not hasattr(self, 'midas_model'):
                print("Loading MiDaS model...")
                self.midas_model = DPTDepthModel(
                    path=model_file,
                    backbone="vitl16_384",
                    non_negative=True,
                )

                normalization = NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

                self.midas_transform = T.Compose([
                    Resize(384, 384, resize_target=None, keep_aspect_ratio=True, ensure_multiple_of=32,
                           resize_method="minimal", image_interpolation_method=cv2.INTER_CUBIC),
                    normalization,
                    PrepareForNet()
                ])

            self.midas_model.eval().to(self.device, memory_format=torch.channels_last if self.device == torch.device("cuda") else None)
            if half_precision:
                self.midas_model = self.midas_model.half()

    def predict(self, prev_img_cv2, midas_weight, half_precision) -> torch.Tensor:
        DEBUG_MODE = opts.data.get("deforum_debug_mode_enabled", False)
        
        img_pil = Image.fromarray(cv2.cvtColor(prev_img_cv2.astype(np.uint8), cv2.COLOR_RGB2BGR))

        
        if self.use_zoe_depth:
            depth_tensor = self.zoe_depth.predict(img_pil).to(self.device)
            # depth_tensor = torch.subtract(50.0, depth_tensor) / 19
        else:
            w, h = prev_img_cv2.shape[1], prev_img_cv2.shape[0]

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
            
            if DEBUG_MODE:
                print("Midas depth tensor before 50/19 calculation:")
                midas_depth_tensor = print(torch.from_numpy(np.expand_dims(midas_depth, axis=0)).squeeze())


            torch.cuda.empty_cache()
            midas_depth = np.subtract(50.0, midas_depth) / 19.0
            depth_tensor = torch.from_numpy(np.expand_dims(midas_depth, axis=0)).squeeze().to(self.device)
        
        if DEBUG_MODE:
            print("Shape of depth_tensor:", depth_tensor.shape)
            print("Tensor data:")
            print(depth_tensor)

        w, h = prev_img_cv2.shape[1], prev_img_cv2.shape[0]
        use_adabins = midas_weight < 1.0 and self.adabins_helper is not None

        if use_adabins:
            MAX_ADABINS_AREA = 500000
            MIN_ADABINS_AREA = 448 * 448
            image_pil_area = w * h
            scale = math.sqrt(MIN_ADABINS_AREA) / math.sqrt(image_pil_area)
            depth_input = img_pil.resize((int(w * scale), int(h * scale)), Image.LANCZOS if image_pil_area > MAX_ADABINS_AREA else Image.BICUBIC)
            try:
                with torch.no_grad():
                    _, adabins_depth = self.adabins_helper.predict_pil(depth_input)
                adabins_depth = adabins_depth.squeeze().cpu().numpy()
                if image_pil_area != MAX_ADABINS_AREA:
                    adabins_depth = TF.resize(torch.from_numpy(adabins_depth),
                                    torch.Size([h, w]),
                                    interpolation=TF.InterpolationMode.BICUBIC).numpy()
            except:
                print("exception encountered, falling back to pure MiDaS")
                use_adabins = False
            torch.cuda.empty_cache()

            if not self.use_zoe_depth:
                midas_depth = (midas_depth * midas_weight + adabins_depth * (1.0 - midas_weight))
                depth_tensor = torch.from_numpy(np.expand_dims(midas_depth, axis=0)).squeeze().to(self.device)
            else:
                depth_map = (depth_tensor.cpu().numpy() * midas_weight + adabins_depth * (1.0 - midas_weight))
                depth_tensor = torch.from_numpy(np.expand_dims(depth_map, axis=0)).squeeze().to(self.device)

        return depth_tensor

    def to_image(self, depth: torch.Tensor):
        depth = depth.cpu().numpy()
        depth = np.expand_dims(depth, axis=0) if len(depth.shape) == 2 else depth
        self.depth_min = min(self.depth_min, depth.min())
        self.depth_max = max(self.depth_max, depth.max())
        denom = max(1e-8, self.depth_max - self.depth_min)
        temp = rearrange((depth - self.depth_min) / denom * 255, 'c h w -> h w c')
        temp = repeat(temp, 'h w 1 -> h w c', c=3)
        return Image.fromarray(temp.astype(np.uint8))

    def save(self, filename: str, depth: torch.Tensor):
        self.to_image(depth).save(filename)

    def to(self, device):
        self.device = device
        if self.use_zoe_depth:
            self.zoe_depth.zoe.to(device)
        else:
            self.midas_model.to(device)
        if self.adabins_helper is not None:
            self.adabins_helper.to(device)
        gc.collect()
        torch.cuda.empty_cache()

    def delete_model(self):
        if self.use_zoe_depth:
            self.zoe_depth.delete()
            del self.zoe_depth
        else:
            del self.midas_model
        gc.collect()
        torch.cuda.empty_cache()
        devices.torch_gc()
        
class AdaBinsModel:
    _instance = None
    
    def __new__(cls, *args, **kwargs):
        keep_in_vram = kwargs.get('keep_in_vram', False)
        if cls._instance is None or not keep_in_vram:
            cls._instance = super().__new__(cls)
            cls._instance._initialize(*args, **kwargs)
        return cls._instance

    def _initialize(self, models_path, keep_in_vram=False):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.keep_in_vram = keep_in_vram

        if self.keep_in_vram or not hasattr(self, 'adabins_helper'):
            if not os.path.exists(os.path.join(models_path, 'AdaBins_nyu.pt')):
                from basicsr.utils.download_util import load_file_from_url
                load_file_from_url(
                    r"https://cloudflare-ipfs.com/ipfs/Qmd2mMnDLWePKmgfS8m6ntAg4nhV5VkUyAydYBp8cWWeB7/AdaBins_nyu.pt",
                    models_path)
                if checksum(os.path.join(models_path, 'AdaBins_nyu.pt')) != "643db9785c663aca72f66739427642726b03acc6c4c1d3755a4587aa2239962746410d63722d87b49fc73581dbc98ed8e3f7e996ff7b9c0d56d0fbc98e23e41a":
                    raise Exception(
                        r"Error while downloading AdaBins_nyu.pt. Please download from here: https://drive.google.com/file/d/1lvyZZbC9NLcS8a__YPcUP7rDiIpbRpoF and place in: " + models_path)
            self.adabins_helper = InferenceHelper(models_path=models_path, dataset='nyu', device=self.device)
            
    def delete_model(self):
        del self.adabins_helper
        torch.cuda.empty_cache()
        devices.torch_gc()