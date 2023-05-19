import os
import torch
from PIL import Image
from torchvision import transforms
from torch.nn.functional import interpolate
import cv2

preclipseg_transform = transforms.Compose([
      transforms.ToTensor(),
      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
      transforms.Resize((512, 512)), #TODO: check if the size is hardcoded
])

def find_clipseg():
    basedirs = [os.getcwd()]
    src_basedirs = []
    for basedir in basedirs:
        src_basedirs.append(os.path.join(os.path.sep.join(os.path.abspath(__file__).split(os.path.sep)[:-2]), 'deforum_helpers', 'src'))

    for basedir in src_basedirs:
        pth = os.path.join(basedir, './clipseg/weights/rd64-uni.pth')
        if os.path.exists(pth):
            return pth
    raise Exception('CLIPseg weights not found!')

def setup_clipseg(root):
    from clipseg.models.clipseg import CLIPDensePredT
    model = CLIPDensePredT(version='ViT-B/16', reduce_dim=64)
    model.eval()
    model.load_state_dict(torch.load(find_clipseg(), map_location=root.device), strict=False)

    model.to(root.device)
    root.clipseg_model = model

def get_word_mask(root, frame, word_mask):
    if root.clipseg_model is None:
        setup_clipseg(root)
    img = preclipseg_transform(frame).to(root.device, dtype=torch.float32)
    word_masks = [word_mask]
    with torch.no_grad():
        preds = root.clipseg_model(img.repeat(len(word_masks),1,1,1), word_masks)[0]

    mask = torch.sigmoid(preds[0][0]).unsqueeze(0).unsqueeze(0) # add batch, channels dims
    resized_mask = interpolate(mask, size=(frame.size[1], frame.size[0]), mode='bicubic').squeeze() # rescale mask back to the target resolution
    numpy_array = resized_mask.multiply(255).to(dtype=torch.uint8,device='cpu').numpy()
    return Image.fromarray(cv2.threshold(numpy_array, 32, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1])
