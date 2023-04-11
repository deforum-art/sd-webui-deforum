import torch
import gc
from PIL import Image
from modules import devices
from zoedepth.utils.misc import colorize 

class ZoeDepth:
    def __init__(self):
        repo = "isl-org/ZoeDepth"
        self.model_zoe = torch.hub.load(repo, "ZoeD_NK", pretrained=True)
        self.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        self.zoe = self.model_zoe.to(self.DEVICE)
        
    def predict(self, image):
        depth_tensor = self.zoe.infer_pil(image, output_type="tensor")
        return depth_tensor

    def save_raw_depth(self, depth, filepath):
        depth.save(filepath, format='PNG', mode='I;16')

    def colorize_depth(self, depth):
        colored = colorize(depth)
        return colored

    def save_colored_depth(self, depth, filepath):
        colored = colorize(depth)
        Image.fromarray(colored).save(filepath)
    
    def delete(self):
        del self.model_zoe
        del self.zoe
        gc.collect()
        torch.cuda.empty_cache()
        devices.torch_gc()
