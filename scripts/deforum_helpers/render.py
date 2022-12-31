import os
import json
import pandas as pd
import cv2
import numpy as np
from PIL import Image

from .generate import generate
from .noise import add_noise
from .animation import sample_from_cv2, sample_to_cv2, anim_frame_warp
from .animation_key_frames import DeformAnimKeys
from .vid2frames import get_frame_name
from .depth import DepthModel
from .colors import maintain_colors
from .parseq_adapter import ParseqAnimKeys
from .seed import next_seed
from .blank_frame_reroll import blank_frame_reroll
from .image_sharpening import unsharp_mask
# Webui
from modules.shared import opts, cmd_opts, state


def render_animation(args, anim_args, parseq_args, animation_prompts, root):

    # use parseq if manifest is provided
    use_parseq = parseq_args.parseq_manifest != None and parseq_args.parseq_manifest.strip()

    # expand key frame strings to values
    keys = DeformAnimKeys(anim_args) if not use_parseq else ParseqAnimKeys(parseq_args, anim_args)

    # resume animation
    start_frame = 0
    if anim_args.resume_from_timestring:
        for tmp in os.listdir(args.outdir):
            if ".txt" in tmp : 
                pass
            else:
                filename = tmp.split("_")
                # don't use saved depth maps to count number of frames
                if anim_args.resume_timestring in filename and "depth" not in filename:
                    start_frame += 1
        #start_frame = start_frame - 1

    # create output folder for the batch
    os.makedirs(args.outdir, exist_ok=True)
    print(f"Saving animation frames to {args.outdir}")

    # save settings for the batch
    settings_filename = os.path.join(args.outdir, f"{args.timestring}_settings.txt")
    with open(settings_filename, "w+", encoding="utf-8") as f:
        args.__dict__["prompts"] = animation_prompts
        s = {**dict(args.__dict__), **dict(anim_args.__dict__)}
        json.dump(s, f, ensure_ascii=False, indent=4)
        
    # resume from timestring
    if anim_args.resume_from_timestring:
        args.timestring = anim_args.resume_timestring

    # Always enable pseudo-3d with parseq. No need for an extra toggle:
    # Whether it's used or not in practice is defined by the schedules
    if use_parseq:
        anim_args.flip_2d_perspective = True        

    # expand prompts out to per-frame
    if use_parseq:
        prompt_series = keys.prompts
    else:
        prompt_series = pd.Series([np.nan for a in range(anim_args.max_frames)])
        for i, prompt in animation_prompts.items():
            prompt_series[int(i)] = prompt
        prompt_series = prompt_series.ffill().bfill()

    # check for video inits
    using_vid_init = anim_args.animation_mode == 'Video Input'

    # load depth model for 3D
    predict_depths = (anim_args.animation_mode == '3D' and anim_args.use_depth_warping) or anim_args.save_depth_maps
    if predict_depths:
        depth_model = DepthModel(root.device)
        depth_model.load_midas(root.models_path, root.half_precision)
        if anim_args.midas_weight < 1.0:
            depth_model.load_adabins(root.models_path)
    else:
        depth_model = None
        anim_args.save_depth_maps = False

    # state for interpolating between diffusion steps
    turbo_steps = 1 if using_vid_init else int(anim_args.diffusion_cadence)
    turbo_prev_image, turbo_prev_frame_idx = None, 0
    turbo_next_image, turbo_next_frame_idx = None, 0

    # resume animation
    prev_sample = None
    color_match_sample = None
    if anim_args.resume_from_timestring:
        last_frame = start_frame-1
        if turbo_steps > 1:
            last_frame -= last_frame%turbo_steps
        path = os.path.join(args.outdir,f"{args.timestring}_{last_frame:05}.png")
        img = cv2.imread(path)
        #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # Changed the colors on resume
        prev_sample = sample_from_cv2(img)
        if anim_args.color_coherence != 'None':
            color_match_sample = img
        if turbo_steps > 1:
            turbo_next_image, turbo_next_frame_idx = sample_to_cv2(prev_sample, type=np.float32), last_frame
            turbo_prev_image, turbo_prev_frame_idx = turbo_next_image, turbo_next_frame_idx
            start_frame = last_frame+turbo_steps

    args.n_samples = 1
    frame_idx = start_frame
    
    #Webui
    state.job_count = anim_args.max_frames
    
    while frame_idx < anim_args.max_frames:
        #Webui
        state.job = f"frame {frame_idx + 1}/{anim_args.max_frames}"
        state.job_no = frame_idx + 1
        if state.interrupted:
            break
        
        print(f"Rendering animation frame {frame_idx} of {anim_args.max_frames}")
        noise = keys.noise_schedule_series[frame_idx]
        strength = keys.strength_schedule_series[frame_idx]
        scale = keys.cfg_scale_schedule_series[frame_idx]
        contrast = keys.contrast_schedule_series[frame_idx]
        kernel = int(keys.kernel_schedule_series[frame_idx])
        sigma = keys.sigma_schedule_series[frame_idx]
        amount = keys.amount_schedule_series[frame_idx]
        threshold = keys.threshold_schedule_series[frame_idx]
        depth = None
        
        # emit in-between frames
        if turbo_steps > 1:
            tween_frame_start_idx = max(0, frame_idx-turbo_steps)
            for tween_frame_idx in range(tween_frame_start_idx, frame_idx):
                tween = float(tween_frame_idx - tween_frame_start_idx + 1) / float(frame_idx - tween_frame_start_idx)
                print(f"  creating in between frame {tween_frame_idx} tween:{tween:0.2f}")

                advance_prev = turbo_prev_image is not None and tween_frame_idx > turbo_prev_frame_idx
                advance_next = tween_frame_idx > turbo_next_frame_idx

                if depth_model is not None:
                    assert(turbo_next_image is not None)
                    depth = depth_model.predict(turbo_next_image, anim_args, root.half_precision)
                
                if advance_prev:
                    turbo_prev_image, _ = anim_frame_warp(turbo_prev_image, args, anim_args, keys, tween_frame_idx, depth_model, depth=depth, device=root.device, half_precision=root.half_precision)
                if advance_next:
                    turbo_next_image, _ = anim_frame_warp(turbo_next_image, args, anim_args, keys, tween_frame_idx, depth_model, depth=depth, device=root.device, half_precision=root.half_precision)

                turbo_prev_frame_idx = turbo_next_frame_idx = tween_frame_idx

                if turbo_prev_image is not None and tween < 1.0:
                    img = turbo_prev_image*(1.0-tween) + turbo_next_image*tween
                else:
                    img = turbo_next_image

                filename = f"{args.timestring}_{tween_frame_idx:05}.png"
                cv2.imwrite(os.path.join(args.outdir, filename), img)
                if anim_args.save_depth_maps:
                    depth_model.save(os.path.join(args.outdir, f"{args.timestring}_depth_{tween_frame_idx:05}.png"), depth)
            if turbo_next_image is not None:
                prev_sample = sample_from_cv2(turbo_next_image)

        # apply transforms to previous frame
        if prev_sample is not None:
            prev_img, depth = anim_frame_warp(prev_sample, args, anim_args, keys, frame_idx, depth_model, depth=None, device=root.device, half_precision=root.half_precision)

            # apply color matching
            if anim_args.color_coherence != 'None':
                if color_match_sample is None:
                    color_match_sample = prev_img.copy()
                else:
                    prev_img = maintain_colors(prev_img, color_match_sample, anim_args.color_coherence)

            # apply scaling
            contrast_sample = prev_img * contrast
            # anti-blur
            contrast_sample = unsharp_mask(contrast_sample, (kernel, kernel), sigma, amount, threshold)
            # apply frame noising
            noised_sample = add_noise(sample_from_cv2(contrast_sample), noise, args.seed, anim_args.noise_type,
                            (anim_args.perlin_w, anim_args.perlin_h, anim_args.perlin_octaves, anim_args.perlin_persistence),
                             args.noise_mask, args.invert_mask)

            # use transformed previous frame as init for current
            args.use_init = True
            if root.half_precision:
                args.init_sample = noised_sample.half().to(root.device)
            else:
                args.init_sample = noised_sample.to(root.device)
            args.strength = max(0.0, min(1.0, strength))
        
        args.scale = scale

        # grab prompt for current frame
        args.prompt = prompt_series[frame_idx]
        
        if args.seed_behavior == 'schedule' or use_parseq:
            args.seed = int(keys.seed_schedule_series[frame_idx])

        if use_parseq:
            args.seed_enable_extras = True
            args.subseed = int(keys.subseed_series[frame_idx])
            args.subseed_strength = keys.subseed_strength_series[frame_idx]

        print(f"{args.prompt} {args.seed}")
        if not using_vid_init:
            print(f"Angle: {keys.angle_series[frame_idx]} Zoom: {keys.zoom_series[frame_idx]}")
            print(f"Tx: {keys.translation_x_series[frame_idx]} Ty: {keys.translation_y_series[frame_idx]} Tz: {keys.translation_z_series[frame_idx]}")
            print(f"Rx: {keys.rotation_3d_x_series[frame_idx]} Ry: {keys.rotation_3d_y_series[frame_idx]} Rz: {keys.rotation_3d_z_series[frame_idx]}")
            if anim_args.use_mask_video:
                mask_frame = os.path.join(args.outdir, 'maskframes', get_frame_name(anim_args.video_mask_path) + f"{frame_idx+1:05}.jpg")
                args.mask_file = mask_frame

        # grab init image for current frame
        if using_vid_init:
            init_frame = os.path.join(args.outdir, 'inputframes', get_frame_name(anim_args.video_init_path) + f"{frame_idx+1:05}.jpg")            
            print(f"Using video init frame {init_frame}")
            args.init_image = init_frame
            if anim_args.use_mask_video:
                mask_frame = os.path.join(args.outdir, 'maskframes', get_frame_name(anim_args.video_mask_path) + f"{frame_idx+1:05}.jpg")
                args.mask_file = mask_frame
                
        # sample the diffusion model
        sample, image = generate(args, anim_args, root, frame_idx, return_sample=True)

        # reroll blank frame 
        if not image.getbbox():
            image = blank_frame_reroll(image, args, root, frame_idx)
            if image == None:
                return

        if not using_vid_init:
            prev_sample = sample

        if turbo_steps > 1:
            turbo_prev_image, turbo_prev_frame_idx = turbo_next_image, turbo_next_frame_idx
            turbo_next_image, turbo_next_frame_idx = sample_to_cv2(sample, type=np.float32), frame_idx
            frame_idx += turbo_steps
        else:    
            filename = f"{args.timestring}_{frame_idx:05}.png"
            image.save(os.path.join(args.outdir, filename))
            if anim_args.save_depth_maps:
                depth = depth_model.predict(sample_to_cv2(sample), anim_args, root.half_precision)
                depth_model.save(os.path.join(args.outdir, f"{args.timestring}_depth_{frame_idx:05}.png"), depth)
            frame_idx += 1

        state.current_image = image

        args.seed = next_seed(args)

