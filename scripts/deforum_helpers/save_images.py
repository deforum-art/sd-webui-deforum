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
import time

def get_output_folder(output_path, batch_folder):
    out_path = os.path.join(output_path,time.strftime('%Y-%m'))
    if batch_folder != "":
        out_path = os.path.join(out_path, batch_folder)
    os.makedirs(out_path, exist_ok=True)
    return out_path

def save_image(image, image_type, filename, args, video_args, root):
    if video_args.store_frames_in_ram:
        root.frames_cache.append({'path':os.path.join(args.outdir, filename), 'image':image, 'image_type':image_type})
    else:
        image.save(os.path.join(args.outdir, filename))

def reset_frames_cache(root):
    root.frames_cache = []
    gc.collect()

def dump_frames_cache(root):
    for image_cache in root.frames_cache:
        if image_cache['image_type'] == 'cv2':
            cv2.imwrite(image_cache['path'], image_cache['image'])
        elif image_cache['image_type'] == 'PIL':
            image_cache['image'].save(image_cache['path'])
    # do not reset the cache since we're going to add frame erasing later function #TODO 
