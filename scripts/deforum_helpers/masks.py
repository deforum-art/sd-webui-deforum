import os
import cv2
import gc
import numpy as np
from PIL import Image, ImageOps
from .video_audio_utilities import get_frame_name

def do_overlay_mask(args, anim_args, img, frame_idx, is_bgr_array=False):
    if is_bgr_array:
        img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)

    if anim_args.use_mask_video:
        current_mask = Image.open(os.path.join(args.outdir, 'maskframes', get_frame_name(anim_args.video_mask_path) + f"{frame_idx+1:09}.jpg"))
        current_frame = Image.open(os.path.join(args.outdir, 'inputframes', get_frame_name(anim_args.video_init_path) + f"{frame_idx+1:09}.jpg"))
    elif args.use_mask:
        current_mask = Image.open(args.mask_file)
        current_frame = Image.open(args.init_image)

    current_mask = current_mask.resize((args.W, args.H), Image.LANCZOS)
    current_frame = current_frame.resize((args.W, args.H), Image.LANCZOS)
    current_mask = ImageOps.grayscale(current_mask)

    img = Image.composite(img, current_frame, current_mask)

    if is_bgr_array:
        img = np.array(img)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    del(current_mask, current_frame)
    gc.collect()

    return img