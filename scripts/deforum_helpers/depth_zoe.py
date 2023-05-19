import torch
from zoedepth.models.builder import build_model
from zoedepth.utils.config import get_config

class ZoeDepth:
    def __init__(self, width=512, height=512):
        conf = get_config("zoedepth_nk", "infer")
        conf.img_size = [width, height]
        self.model_zoe = build_model(conf)
        self.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        self.zoe = self.model_zoe.to(self.DEVICE)
        self.width = width
        self.height = height
        
    def predict(self, image):
        self.zoe.core.prep.resizer._Resize__width = self.width
        self.zoe.core.prep.resizer._Resize__height = self.height
        depth_tensor = self.zoe.infer_pil(image, output_type="tensor")
        return depth_tensor
        
    def to(self, device):
        self.DEVICE = device
        self.zoe = self.model_zoe.to(device)
        
    def save_raw_depth(self, depth, filepath):
        depth.save(filepath, format='PNG', mode='I;16')
    
    def delete(self):
        del self.model_zoe
        del self.zoe