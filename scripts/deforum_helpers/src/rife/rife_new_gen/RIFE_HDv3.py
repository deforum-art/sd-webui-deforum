import os
import torch
import torch.nn as nn
import numpy as np
from torch.optim import AdamW
import torch.optim as optim
import itertools
from ..model.warplayer import warp
from torch.nn.parallel import DistributedDataParallel as DDP
from .IFNet_HDv3 import *
import torch.nn.functional as F
from ..model.loss import *
from ..model.checksum import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
class Model:
    def __init__(self, local_rank=-1):
        self.flownet = IFNet()
        self.device()
        self.optimG = AdamW(self.flownet.parameters(), lr=1e-6, weight_decay=1e-4)
        self.epe = EPE()
        self.version = 3.9
        # self.vgg = VGGPerceptualLoss().to(device)
        self.sobel = SOBEL()
        if local_rank != -1:
            self.flownet = DDP(self.flownet, device_ids=[local_rank], output_device=local_rank)

    def train(self):
        self.flownet.train()

    def eval(self):
        self.flownet.eval()

    def device(self):
        self.flownet.to(device)
        
         
    def load_model(self, path, rank, deforum_models_path):
        
        if path == 'RIFE46':
            if not os.path.exists(os.path.join(deforum_models_path,'RIFE46.pkl')):
                from basicsr.utils.download_util import load_file_from_url
                load_file_from_url(r"https://github.com/hithereai/Practical-RIFE/releases/download/rife46/RIFE46.pkl", deforum_models_path)
                if checksum(os.path.join(deforum_models_path,'RIFE46.pkl')) != 'af6f0b4bed96dea2c9f0624b449216c7adfaf7f0b722fba0c8f5c6e20b2ec39559cf33f3d238d53b160c22f00c6eaa47dc54a6e4f8aa4f59a6e4a9e90e1a808a':
                    raise Exception(r"Error while downloading RIFE46.pkl. Please download from here: https://github.com/hithereai/Practical-RIFE/releases/download/rife46/RIFE46.pkl and place in: " + deforum_models_path)
        elif path == 'RIFE43':
            if not os.path.exists(os.path.join(deforum_models_path,'RIFE43.pkl')):
                from basicsr.utils.download_util import load_file_from_url
                load_file_from_url(r"https://github.com/hithereai/Practical-RIFE/releases/download/rife43/RIFE43.pkl", deforum_models_path)
                if checksum(os.path.join(deforum_models_path,'RIFE43.pkl')) != 'ed660f58708ee369a0b3855f64d2d07a6997d949f33067faae51d740123c5ee015901cc57553594f2df8ec08131a1c5f7c883c481eac0f9addd84379acea90c8':
                    raise Exception(r"Error while downloading RIFE43.pkl. Please download from here: https://github.com/hithereai/Practical-RIFE/releases/download/rife43/RIFE43.pkl and place in: " + deforum_models_path)
        elif path == 'RIFE40':
            if not os.path.exists(os.path.join(deforum_models_path,'RIFE40.pkl')):
                from basicsr.utils.download_util import load_file_from_url
                load_file_from_url(r"https://github.com/hithereai/Practical-RIFE/releases/download/rife40/RIFE40.pkl", deforum_models_path)
                if checksum(os.path.join(deforum_models_path,'RIFE40.pkl')) != '0baf0bed23597cda402a97a80a7d14c26a9ed797d2fc0790aac93b19ca5b0f50676ba07aa9f8423cf061ed881ece6e67922f001ea402bfced83ef67675142ce7':
                    raise Exception(r"Error while downloading RIFE40.pkl. Please download from here: https://github.com/hithereai/Practical-RIFE/releases/download/rife40/RIFE40.pkl and place in: " + deforum_models_path)
        def convert(param):
            if rank == -1:
                return {
                    k.replace("module.", ""): v
                    for k, v in param.items()
                    if "module." in k
                }
            else:
                return param
        if rank <= 0:
            if torch.cuda.is_available():
                self.flownet.load_state_dict(convert(torch.load(os.path.join(deforum_models_path,'{}.pkl').format(path))), False)
            else:
                self.flownet.load_state_dict(convert(torch.load(os.path.join(deforum_models_path,'{}.pkl').format(path), map_location ='cpu')), False)

    def inference(self, img0, img1, timestep=0.5, scale=1.0):
        imgs = torch.cat((img0, img1), 1)
        scale_list = [8/scale, 4/scale, 2/scale, 1/scale]
        flow, mask, merged = self.flownet(imgs, timestep, scale_list)
        return merged[3]
    
    def update(self, imgs, gt, learning_rate=0, mul=1, training=True, flow_gt=None):
        for param_group in self.optimG.param_groups:
            param_group['lr'] = learning_rate
        img0 = imgs[:, :3]
        img1 = imgs[:, 3:]
        if training:
            self.train()
        else:
            self.eval()
        scale = [8, 4, 2, 1]
        flow, mask, merged = self.flownet(torch.cat((imgs, gt), 1), scale=scale, training=training)
        loss_l1 = (merged[3] - gt).abs().mean()
        loss_smooth = self.sobel(flow[3], flow[3]*0).mean()
        # loss_vgg = self.vgg(merged[2], gt)
        if training:
            self.optimG.zero_grad()
            loss_G = loss_l1 + loss_cons + loss_smooth * 0.1
            loss_G.backward()
            self.optimG.step()
        else:
            flow_teacher = flow[2]
        return merged[3], {
            'mask': mask,
            'flow': flow[3][:, :2],
            'loss_l1': loss_l1,
            'loss_cons': loss_cons,
            'loss_smooth': loss_smooth,
            }
