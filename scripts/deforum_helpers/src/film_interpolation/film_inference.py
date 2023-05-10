import os
import gc
from glob import glob
import bisect
from tqdm import tqdm
import torch
import numpy as np
import cv2
from .film_util import load_image
import time
from types import SimpleNamespace
from modules.shared import cmd_opts
import warnings
warnings.filterwarnings("ignore")

def run_film_interp_infer(
    model_path = None,
    input_folder = None,
    save_folder = None,
    inter_frames = None):
    
    args = SimpleNamespace()
    args.model_path = model_path
    args.input_folder = input_folder
    args.save_folder = save_folder
    args.inter_frames = inter_frames
    
    # Check if the folder exists
    if not os.path.exists(args.input_folder):
        print(f"Error: Folder '{args.input_folder}' does not exist.")
        return
    # Check if the folder contains any PNG or JPEG images
    if not any([f.endswith(".png") or f.endswith(".jpg") for f in os.listdir(args.input_folder)]):
        print(f"Error: Folder '{args.input_folder}' does not contain any PNG or JPEG images.")
        return
   
    start_time = time.time() # Timer START
    
    # Sort Jpg/Png images by name
    image_paths = sorted(glob(os.path.join(args.input_folder, "*.[jJ][pP][gG]")) + glob(os.path.join(args.input_folder, "*.[pP][nN][gG]")))
    print(f"Total frames to FILM-interpolate: {len(image_paths)}. Total frame-pairs: {len(image_paths)-1}.")
    
    model = torch.jit.load(args.model_path, map_location='cpu')
    # half precision the model if user didn't pass --no-half/ --precision full cmd arg flags
    if not cmd_opts.no_half:
        model = model.half()
    model = model.cuda()
    model.eval()

    for i in tqdm(range(len(image_paths) - 1), desc='FILM progress'):
        img1 = image_paths[i]
        img2 = image_paths[i+1]
        img_batch_1, crop_region_1 = load_image(img1)
        img_batch_2, crop_region_2 = load_image(img2)
        img_batch_1 = torch.from_numpy(img_batch_1).permute(0, 3, 1, 2)
        img_batch_2 = torch.from_numpy(img_batch_2).permute(0, 3, 1, 2)

        save_path = os.path.join(args.save_folder, f"{i}_to_{i+1}.jpg")

        results = [
            img_batch_1,
            img_batch_2
        ]

        idxes = [0, inter_frames + 1]
        remains = list(range(1, inter_frames + 1))

        splits = torch.linspace(0, 1, inter_frames + 2)

        inner_loop_progress = tqdm(range(len(remains)), leave=False, disable=True)
        for _ in inner_loop_progress:
            starts = splits[idxes[:-1]]
            ends = splits[idxes[1:]]
            distances = ((splits[None, remains] - starts[:, None]) / (ends[:, None] - starts[:, None]) - .5).abs()
            matrix = torch.argmin(distances).item()
            start_i, step = np.unravel_index(matrix, distances.shape)
            end_i = start_i + 1

            x0 = results[start_i]
            x1 = results[end_i]

            x0 = x0.half()
            x1 = x1.half()
            x0 = x0.cuda()
            x1 = x1.cuda()

            dt = x0.new_full((1, 1), (splits[remains[step]] - splits[idxes[start_i]])) / (splits[idxes[end_i]] - splits[idxes[start_i]])

            with torch.no_grad():
                prediction = model(x0, x1, dt)
            insert_position = bisect.bisect_left(idxes, remains[step])
            idxes.insert(insert_position, remains[step])
            results.insert(insert_position, prediction.clamp(0, 1).cpu().float())
            inner_loop_progress.update(1)
            del remains[step]
        inner_loop_progress.close()
        # create output folder for interoplated imgs to live in
        os.makedirs(args.save_folder, exist_ok=True)

        y1, x1, y2, x2 = crop_region_1
        frames = [(tensor[0] * 255).byte().flip(0).permute(1, 2, 0).numpy()[y1:y2, x1:x2].copy() for tensor in results]

        existing_files = os.listdir(args.save_folder)
        if len(existing_files) > 0:
            existing_numbers = [int(file.split("_")[1].split(".")[0]) for file in existing_files]
            next_number = max(existing_numbers) + 1
        else:
            next_number = 0

        outer_loop_count = i
        for i, frame in enumerate(frames):
            frame_path = os.path.join(args.save_folder, f"frame_{next_number:09d}.png") 
            # last pair, save all frames including the last one
            if len(image_paths) - 2 == outer_loop_count:
                cv2.imwrite(frame_path, frame)
            else: # not last pair, don't save the last frame
                if not i == len(frames) - 1:
                    cv2.imwrite(frame_path, frame)
            next_number += 1
    
    # remove FILM model from memory
    if model is not None:
        del model
        torch.cuda.empty_cache()
        gc.collect()
    print(f"Interpolation \033[0;32mdone\033[0m in {time.time()-start_time:.2f} seconds!")