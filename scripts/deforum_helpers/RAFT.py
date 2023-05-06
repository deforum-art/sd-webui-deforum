import torch
import numpy as np
import torchvision.transforms.functional as F
from torchvision.models.optical_flow import Raft_Large_Weights, raft_large

class RAFT:
    def __init__(self):
        weights = Raft_Large_Weights.DEFAULT
        self.transforms = weights.transforms()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = raft_large(weights=weights, progress=False).to(self.device).eval()

    def predict(self, image1, image2, num_flow_updates:int = 50):
        img1 = F.to_tensor(image1)
        img2 = F.to_tensor(image2)
        img1_batch, img2_batch = img1.unsqueeze(0), img2.unsqueeze(0)
        img1_batch, img2_batch = self.transforms(img1_batch, img2_batch)

        with torch.no_grad():
            flow = self.model(image1=img1_batch.to(self.device), image2=img2_batch.to(self.device), num_flow_updates=num_flow_updates)[-1].cpu().numpy()[0]

        # align the flow array to have the shape (w, h, 2) so it's compatible with the rest of CV2's flow methods
        flow = np.transpose(flow, (1, 2, 0))

        return flow

    def delete_model(self):
        del self.model