import torch
import cv2
import os
import numpy as np
import torchvision.transforms as transforms
from .general_utils import download_file_with_checksum
from leres.lib.multi_depth_model_woauxi import RelDepthModel
from leres.lib.net_tools import load_ckpt
    
class LeReSDepth:
    def __init__(self, width=448, height=448, models_path=None, checkpoint_name='res101.pth', backbone='resnext101'):
        self.width = width
        self.height = height
        self.models_path = models_path
        self.checkpoint_name = checkpoint_name
        self.backbone = backbone

        download_file_with_checksum(url='https://cloudstor.aarnet.edu.au/plus/s/lTIJF4vrvHCAI31/download', expected_checksum='7fdc870ae6568cb28d56700d0be8fc45541e09cea7c4f84f01ab47de434cfb7463cacae699ad19fe40ee921849f9760dedf5e0dec04a62db94e169cf203f55b1', dest_folder=models_path, dest_filename=self.checkpoint_name)

        self.depth_model = RelDepthModel(backbone=self.backbone)
        self.depth_model.eval()
        self.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        self.depth_model.to(self.DEVICE)
        load_ckpt(os.path.join(self.models_path, self.checkpoint_name), self.depth_model, None, None)

    @staticmethod
    def scale_torch(img):
        if len(img.shape) == 2:
            img = img[np.newaxis, :, :]
        if img.shape[2] == 3:
            transform = transforms.Compose([transforms.ToTensor(),
                                            transforms.Normalize((0.485, 0.456, 0.406) , (0.229, 0.224, 0.225))])
            img = transform(img)
        else:
            img = img.astype(np.float32)
            img = torch.from_numpy(img)
        return img

    def predict(self, image):
        resized_image = cv2.resize(image, (self.width, self.height))
        img_torch = self.scale_torch(resized_image)[None, :, :, :]
        pred_depth = self.depth_model.inference(img_torch).cpu().numpy().squeeze()
        pred_depth_ori = cv2.resize(pred_depth, (image.shape[1], image.shape[0]))
        return torch.from_numpy(pred_depth_ori).unsqueeze(0).to(self.DEVICE)

    def save_raw_depth(self, depth, filepath):
        depth_normalized = (depth / depth.max() * 60000).astype(np.uint16)
        cv2.imwrite(filepath, depth_normalized)
        
    def to(self, device):
        self.DEVICE = device
        self.depth_model = self.depth_model.to(device)

    def delete(self):
        del self.depth_model