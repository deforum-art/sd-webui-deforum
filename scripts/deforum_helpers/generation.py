import cv2
import random
import gc
import numpy as np
from PIL import Image
from .colors import maintain_colors
from .hybrid_video import get_flow_from_images, image_transform_optical_flow
from .generate import generate

# Optical flow generation, before generation
def optical_flow_generation(prev_img, redo_flow_factor, raft_model, args, keys, anim_args, loop_args, controlnet_args, root, frame_idx, sampler_name):
    print(f"Optical flow generation is working with flow {anim_args.optical_flow_redo_generation} before final generation.")

    # uses random seed for extra generation. args aren't carried back to render.py, so the original seed is intact
    args.seed = random.randint(0, 2 ** 32 - 1)

    disposable_image = generate(args, keys, anim_args, loop_args, controlnet_args, root, frame_idx, sampler_name=sampler_name)
    disposable_image = cv2.cvtColor(np.array(disposable_image), cv2.COLOR_RGB2BGR)
    disposable_flow = get_flow_from_images(prev_img, disposable_image, anim_args.optical_flow_redo_generation, raft_model)
    disposable_image = cv2.cvtColor(disposable_image, cv2.COLOR_BGR2RGB)
    disposable_image = image_transform_optical_flow(disposable_image, disposable_flow, redo_flow_factor)    
    init_sample = Image.fromarray(disposable_image)

    del (disposable_image, disposable_flow)
    gc.collect()

    return init_sample

# Redo generation, before generation
def redo_generation(prev_img, args, keys, anim_args, loop_args, controlnet_args, root, frame_idx, sampler_name):
    for n in range(0, int(anim_args.diffusion_redo)):
        print(f"Redo generation {n + 1} of {int(anim_args.diffusion_redo)} before final generation")

        # uses random seed for extra generation. args aren't carried back to render.py, so the original seed is intact
        args.seed = random.randint(0, 2 ** 32 - 1)

        disposable_image = generate(args, keys, anim_args, loop_args, controlnet_args, root, frame_idx, sampler_name=sampler_name)
        disposable_image = cv2.cvtColor(np.array(disposable_image), cv2.COLOR_RGB2BGR)

        # color match on last one only
        if n == int(anim_args.diffusion_redo):
            disposable_image = maintain_colors(prev_img, color_match_sample, anim_args.color_coherence)

        init_sample = Image.fromarray(cv2.cvtColor(disposable_image, cv2.COLOR_BGR2RGB))
        del (disposable_image)

    gc.collect()
    return init_sample
