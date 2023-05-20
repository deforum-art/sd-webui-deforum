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
