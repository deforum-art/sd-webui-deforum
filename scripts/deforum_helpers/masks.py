# Copyright (C) 2023 Deforum LLC
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, version 3 of the License.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

# Contact the authors: https://deforum.github.io/

import os
import cv2
import gc
import numpy as np
from PIL import Image, ImageOps
from .video_audio_utilities import get_frame_name
from .load_images import load_image

def do_overlay_mask(args, anim_args, img, frame_idx, is_bgr_array=False):
    if is_bgr_array:
        img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)

    if anim_args.use_mask_video:
        current_mask = Image.open(os.path.join(args.outdir, 'maskframes', get_frame_name(anim_args.video_mask_path) + f"{frame_idx:09}.jpg"))
        current_frame = Image.open(os.path.join(args.outdir, 'inputframes', get_frame_name(anim_args.video_init_path) + f"{frame_idx:09}.jpg"))
    elif args.use_mask:
        current_mask = args.mask_image if args.mask_image is not None else load_image(args.mask_file, None)
        if args.init_image is None and args.init_image_box is None:
            current_frame = img
        else:
            current_frame = load_image(args.init_image, args.init_image_box)

    current_mask = current_mask.resize((args.W, args.H), Image.LANCZOS)
    current_frame = current_frame.resize((args.W, args.H), Image.LANCZOS)
    current_mask = ImageOps.grayscale(current_mask)
    
    if args.invert_mask:
        current_mask = ImageOps.invert(current_mask)

    img = Image.composite(img, current_frame, current_mask)

    if is_bgr_array:
        img = np.array(img)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    del(current_mask, current_frame)
    gc.collect()

    return img