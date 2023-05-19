import requests
import os
from PIL import Image
import socket
import torchvision.transforms.functional as TF
from .general_utils import clean_gradio_path_strings

def load_img(path : str, shape=None, use_alpha_as_mask=False):
    # use_alpha_as_mask: Read the alpha channel of the image as the mask image
    image = load_image(path)
    image = image.convert('RGBA') if use_alpha_as_mask else image.convert('RGB')
    image = image.resize(shape, resample=Image.LANCZOS) if shape is not None else image

    mask_image = None
    if use_alpha_as_mask:
        # Split alpha channel into a mask_image
        red, green, blue, alpha = Image.Image.split(image) # not interested in R G or B, just in the alpha channel
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
    image_path = clean_gradio_path_strings(image_path)
    image = None
    if image_path.startswith('http://') or image_path.startswith('https://'):
        try:
            host = socket.gethostbyname("www.google.com")
            s = socket.create_connection((host, 80), 2)
            s.close()
        except:
            raise ConnectionError("There is no active internet connection available (couldn't connect to google.com as a network test) - please use *local* masks and init files only.")
        try:
            response = requests.get(image_path, stream=True)
        except requests.exceptions.RequestException as e:
            raise ConnectionError("Failed to download image due to no internet connection. Error: {}".format(e))
        if response.status_code == 404 or response.status_code != 200:
            raise ConnectionError("Init image url or mask image url is not valid")
        image = Image.open(response.raw).convert('RGB')
    else:
        if not os.path.exists(image_path):
            raise RuntimeError("Init image path or mask image path is not valid")
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

# "check_mask_for_errors" may have prevented errors in composable masks,
# but it CAUSES errors on any frame where it's all black.
# Bypassing the check below until we can fix it even better.
# This may break composable masks, but it makes ACTUAL masks usable.
def check_mask_for_errors(mask_input, invert_mask=False):
    extrema = mask_input.getextrema()
    if (invert_mask):
        if extrema == (255,255): 
            print("after inverting mask will be blank. ignoring mask")  
            return None
    elif extrema == (0,0): 
        print("mask is blank. ignoring mask")  
        return None
    else:
        return mask_input    
 
def get_mask(args):
    return prepare_mask(args.mask_file, (args.W, args.H), args.mask_contrast_adjust, args.mask_brightness_adjust)

def get_mask_from_file(mask_file, args):
    return prepare_mask(mask_file, (args.W, args.H), args.mask_contrast_adjust, args.mask_brightness_adjust)

def blank_if_none(mask, w, h, mode):
    return Image.new(mode, (w, h), (0)) if mask is None else mask

def none_if_blank(mask):
    return None if mask.getextrema() == (0,0) else mask
