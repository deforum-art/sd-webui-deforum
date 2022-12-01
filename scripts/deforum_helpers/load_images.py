import requests
from PIL import Image
import numpy as np
import torchvision.transforms.functional as TF

def load_img(path : str, shape=None, use_alpha_as_mask=False):
    # use_alpha_as_mask: Read the alpha channel of the image as the mask image
    image = load_image(path)
    if use_alpha_as_mask:
        image = image.convert('RGBA')
    else:
        image = image.convert('RGB')

    if shape is not None:
        image = image.resize(shape, resample=Image.LANCZOS)

    mask_image = None
    if use_alpha_as_mask:
        # Split alpha channel into a mask_image
        red, green, blue, alpha = Image.Image.split(image)
        mask_image = alpha.convert('L')
        image = image.convert('RGB')
        
        # check using init image alpha as mask if mask is not blank
        extrema = mask_image.getextrema()
        if (extrema == (0,0)) or extrema == (255,255):
            print("use_alpha_as_mask==True: Using the alpha channel from the init image as a mask, but the alpha channel is blank.")
            print("ignoring alpha as mask.")
            mask_image = None

    return image, mask_image

def load_image(image_path :str):
    if image_path.startswith('http://') or image_path.startswith('https://'):
        image = Image.open(requests.get(image_path, stream=True).raw).convert('RGB')
    else:
        image = Image.open(image_path).convert('RGB')
    return image

def prepare_mask(mask_input, mask_shape, mask_brightness_adjust=1.0, mask_contrast_adjust=1.0):
    """
    prepares mask for use in webui
    """
    if isinstance(mask_input, Image.Image):
        mask = mask_input
    else :
        mask = load_image(mask_input)
    mask = mask.resize(mask_shape, resample=Image.LANCZOS)
    if mask_brightness_adjust != 1:
        mask = TF.adjust_brightness(mask, mask_brightness_adjust)
    if mask_contrast_adjust != 1:
        mask = TF.adjust_contrast(mask, mask_contrast_adjust)
    mask = mask.convert('L')
    return mask
