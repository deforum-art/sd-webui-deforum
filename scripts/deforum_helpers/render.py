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
import pandas as pd
import cv2
import numpy as np
import numexpr
import gc
import random
import PIL
import time
import math
from PIL import Image, ImageOps
from .generate import generate, isJson
from .noise import add_noise
from .animation import anim_frame_warp
from .animation_key_frames import DeformAnimKeys, LooperAnimKeys
from .video_audio_utilities import get_frame_name, get_next_frame, render_preview
from .depth import DepthModel
from .colors import maintain_colors
from .parseq_adapter import ParseqAdapter
from .seed import next_seed
from .image_sharpening import unsharp_mask
from .load_images import get_mask, load_img, load_image, get_mask_from_file
from .hybrid_video import hybrid_generation, hybrid_composite, image_transform_optical_flow, get_flow_from_images
from .hybrid_render import hybrid_motion_for_cadence, hybrid_motion_before_generation, hybrid_motion_after_generation
from .cadence_flow import cadence_optflow_setup, cadence_optflow_warp_blend
from .cadence import get_cadence_keys, get_cadence_tweens
from .generation import optical_flow_generation, redo_generation
from .save_images import save_image
from .composable_masks import compose_mask_with_check
from .settings import save_settings_from_animation_run
from .deforum_controlnet import unpack_controlnet_vids, is_controlnet_enabled
from .subtitle_handler import init_srt_file, write_frame_subtitle, format_animation_params
from .resume import get_resume_vars
from .masks import do_overlay_mask
from .prompt import prepare_prompt
from modules.shared import opts, cmd_opts, state, sd_model
from modules import lowvram, devices, sd_hijack
from .RAFT import RAFT

from deforum_api import JobStatusTracker

def render_animation(args, anim_args, video_args, parseq_args, loop_args, controlnet_args, root):
    if opts.data.get("deforum_save_gen_info_as_srt", False):  # create .srt file and set timeframe mechanism using FPS
        srt_filename = os.path.join(args.outdir, f"{root.timestring}.srt")
        srt_frame_duration = init_srt_file(srt_filename, video_args.fps)

    # needs to be setup outside of 2D/3D conditional
    hybrid_available = anim_args.hybrid_composite != 'None' or anim_args.hybrid_motion != 'None'

    if anim_args.animation_mode in ['2D', '3D']:
        # handle hybrid video frame generation and setup vars used later
        if hybrid_available:
            args, anim_args, inputfiles = hybrid_generation(args, anim_args, root)
            # path required by hybrid functions, even if hybrid_comp_save_extra_frames is False
            hybrid_frame_path = os.path.join(args.outdir, 'hybridframes')

        # initialize prev_flow
        if anim_args.hybrid_motion != 'None':
            prev_flow = None

        if loop_args.use_looper:
            print("Using Guided Images mode: seed_behavior will be set to 'schedule' and 'strength_0_no_init' to False")
            if args.strength == 0:
                raise RuntimeError("Strength needs to be greater than 0 in Init tab")
            args.strength_0_no_init = False
            args.seed_behavior = "schedule"
            if not isJson(loop_args.init_images):
                raise RuntimeError("The images set for use with keyframe-guidance are not in a proper JSON format")

    # handle controlnet video input frames generation
    if is_controlnet_enabled(controlnet_args):
        unpack_controlnet_vids(args, anim_args, controlnet_args)

    # initialise Parseq adapter
    parseq_adapter = ParseqAdapter(parseq_args, anim_args, video_args, controlnet_args, loop_args)

    # expand key frame strings to values
    keys = DeformAnimKeys(anim_args, args.seed) if not parseq_adapter.use_parseq else parseq_adapter.anim_keys
    loopSchedulesAndData = LooperAnimKeys(loop_args, anim_args, args.seed) if not parseq_adapter.use_parseq else parseq_adapter.looper_keys

    # create output folder for the batch
    os.makedirs(args.outdir, exist_ok=True)
    print(f"Saving animation frames to:\n{args.outdir}")

    # save settings.txt file for the current run
    save_settings_from_animation_run(args, anim_args, parseq_args, loop_args, controlnet_args, video_args, root)

    # resume from timestring
    if anim_args.resume_from_timestring:
        root.timestring = anim_args.resume_timestring

    # Always enable pseudo-3d with parseq. No need for an extra toggle:
    # Whether it's used or not in practice is defined by the schedules
    if parseq_adapter.use_parseq:
        anim_args.flip_2d_perspective = True

    # expand prompts out to per-frame
    if parseq_adapter.manages_prompts():
        prompt_series = keys.prompts
    else:
        prompt_series = pd.Series([np.nan for a in range(anim_args.max_frames + 1)])
        for i, prompt in root.animation_prompts.items():
            if str(i).isdigit():
                prompt_series[int(i)] = prompt
            else:
                prompt_series[int(numexpr.evaluate(i))] = prompt
        prompt_series = prompt_series.ffill().bfill()

    # check for video inits
    using_vid_init = anim_args.animation_mode == 'Video Input'

    # load depth model for 3D
    predict_depths = (anim_args.animation_mode == '3D' and anim_args.use_depth_warping) or anim_args.save_depth_maps
    predict_depths_for_hybrid = anim_args.hybrid_composite != 'None' and anim_args.hybrid_comp_mask_type in ['Depth', 'Video Depth']
    if predict_depths or predict_depths_for_hybrid:
        keep_in_vram = opts.data.get("deforum_keep_3d_models_in_vram")

        device = ('cpu' if cmd_opts.lowvram or cmd_opts.medvram else root.device)
        depth_model = DepthModel(root.models_path, device, root.half_precision, keep_in_vram=keep_in_vram, depth_algorithm=anim_args.depth_algorithm, Width=args.W, Height=args.H,
                                 midas_weight=anim_args.midas_weight)

        # the 'Depth' hybrid composite mask requires the saved depth maps because it refers to the previous frame's depth
        if predict_depths_for_hybrid and anim_args.hybrid_comp_mask_type == 'Depth':
            anim_args.save_depth_maps = True
    else:
        depth_model = None
        anim_args.save_depth_maps = False

    raft_model = None
    load_raft = (anim_args.optical_flow_cadence == "RAFT" and int(anim_args.diffusion_cadence) > 1) or \
                (anim_args.hybrid_motion == "Optical Flow" and anim_args.hybrid_flow_method == "RAFT") or \
                (anim_args.previous_image_flow == "RAFT") or \
                (anim_args.optical_flow_redo_generation == "RAFT")
    if load_raft:
        print("Loading RAFT model...")
        raft_model = RAFT()

    # state for interpolating between diffusion steps
    turbo_steps = 1 if using_vid_init else int(anim_args.diffusion_cadence)
    turbo_prev_image, turbo_prev_frame_idx = None, 0
    turbo_next_image, turbo_next_frame_idx = None, 0

    # initialize crucial vars, set master frame index
    prev_img = None
    prev_imgs = []
    prev_imgs_max = 2
    color_match_sample = None
    frame_idx = 0
    resume_skip_cadence = False

    # resume animation (requires at least two "actual" frames - see function)
    if anim_args.resume_from_timestring:
        prev_frame_idx, next_frame_idx, prev_img, next_img, frame_idx, resume_skip_cadence = get_resume_vars(
            folder=args.outdir,
            timestring=anim_args.resume_timestring,
            cadence=turbo_steps
        )

        # set up turbo step vars
        if turbo_steps > 1:
            turbo_prev_image, turbo_prev_frame_idx = prev_img, prev_frame_idx
            turbo_next_image, turbo_next_frame_idx = next_img, next_frame_idx

    # reset the mask vals as they are overwritten in the compose_mask algorithm
    mask_vals = {}
    noise_mask_vals = {}

    mask_vals['everywhere'] = Image.new('1', (args.W, args.H), 1)
    noise_mask_vals['everywhere'] = Image.new('1', (args.W, args.H), 1)

    mask_image = None

    if args.use_init and ((args.init_image != None and args.init_image != '') or args.init_image_box != None):
        _, mask_image = load_img(args.init_image,
                                 args.init_image_box,
                                 shape=(args.W, args.H),
                                 use_alpha_as_mask=args.use_alpha_as_mask)
        mask_vals['video_mask'] = mask_image
        noise_mask_vals['video_mask'] = mask_image

    # Grab the first frame masks since they wont be provided until next frame
    # Video mask overrides the init image mask, also, won't be searching for init_mask if use_mask_video is set
    # Made to solve https://github.com/deforum-art/deforum-for-automatic1111-webui/issues/386
    if anim_args.use_mask_video:

        args.mask_file = get_mask_from_file(get_next_frame(args.outdir, anim_args.video_mask_path, frame_idx, True), args)
        root.noise_mask = get_mask_from_file(get_next_frame(args.outdir, anim_args.video_mask_path, frame_idx, True), args)

        mask_vals['video_mask'] = get_mask_from_file(get_next_frame(args.outdir, anim_args.video_mask_path, frame_idx, True), args)
        noise_mask_vals['video_mask'] = get_mask_from_file(get_next_frame(args.outdir, anim_args.video_mask_path, frame_idx, True), args)
    elif mask_image is None and args.use_mask:
        mask_vals['video_mask'] = get_mask(args)
        noise_mask_vals['video_mask'] = get_mask(args)  # TODO?: add a different default noisc mask

    # get color match for 'Image' color coherence only once, before loop
    if anim_args.color_coherence == 'Image':
        color_match_sample = load_image(anim_args.color_coherence_image_path, None)
        color_match_sample = color_match_sample.resize((args.W, args.H), PIL.Image.LANCZOS)
        color_match_sample = cv2.cvtColor(np.array(color_match_sample), cv2.COLOR_RGB2BGR)

    # Webui
    state.job_count = anim_args.max_frames
    last_preview_frame = 0

    while frame_idx <= anim_args.max_frames:
        # Webui

        state.job = f"frame {frame_idx + 1}/{anim_args.max_frames}"
        state.job_no = frame_idx + 1

        if state.skipped:
            print("\n** PAUSED **")
            state.skipped = False
            while not state.skipped:
                time.sleep(0.1)
            print("** RESUMING **")

        print(f"Animation frame index in progress: {frame_idx}/{anim_args.max_frames} ")

        noise = keys.noise_schedule_series[frame_idx]
        strength = keys.strength_schedule_series[frame_idx]
        scale = keys.cfg_scale_schedule_series[frame_idx]
        contrast = keys.contrast_schedule_series[frame_idx]
        kernel = int(keys.kernel_schedule_series[frame_idx])
        sigma = keys.sigma_schedule_series[frame_idx]
        amount = keys.amount_schedule_series[frame_idx]
        threshold = keys.threshold_schedule_series[frame_idx]
        hybrid_flow_factor = keys.hybrid_flow_factor_schedule_series[frame_idx]
        previous_image_flow_factor = keys.previous_image_flow_factor_schedule_series[frame_idx]
        redo_flow_factor = keys.redo_flow_factor_schedule_series[frame_idx]
        hybrid_comp_schedules = {
            "alpha": keys.hybrid_comp_alpha_schedule_series[frame_idx],
            "mask_blend_alpha": keys.hybrid_comp_mask_blend_alpha_schedule_series[frame_idx],
            "mask_contrast": keys.hybrid_comp_mask_contrast_schedule_series[frame_idx],
            "mask_auto_contrast_cutoff_low": int(keys.hybrid_comp_mask_auto_contrast_cutoff_low_schedule_series[frame_idx]),
            "mask_auto_contrast_cutoff_high": int(keys.hybrid_comp_mask_auto_contrast_cutoff_high_schedule_series[frame_idx])
        }
        scheduled_sampler_name = None
        scheduled_clipskip = None
        scheduled_noise_multiplier = None
        scheduled_ddim_eta = None
        scheduled_ancestral_eta = None

        mask_seq = None
        noise_mask_seq = None
        if anim_args.enable_steps_scheduling and keys.steps_schedule_series[frame_idx] is not None:
            args.steps = int(keys.steps_schedule_series[frame_idx])
        if anim_args.enable_sampler_scheduling and keys.sampler_schedule_series[frame_idx] is not None:
            scheduled_sampler_name = keys.sampler_schedule_series[frame_idx].casefold()
        if anim_args.enable_clipskip_scheduling and keys.clipskip_schedule_series[frame_idx] is not None:
            scheduled_clipskip = int(keys.clipskip_schedule_series[frame_idx])
        if anim_args.enable_noise_multiplier_scheduling and keys.noise_multiplier_schedule_series[frame_idx] is not None:
            scheduled_noise_multiplier = float(keys.noise_multiplier_schedule_series[frame_idx])
        if anim_args.enable_ddim_eta_scheduling and keys.ddim_eta_schedule_series[frame_idx] is not None:
            scheduled_ddim_eta = float(keys.ddim_eta_schedule_series[frame_idx])
        if anim_args.enable_ancestral_eta_scheduling and keys.ancestral_eta_schedule_series[frame_idx] is not None:
            scheduled_ancestral_eta = float(keys.ancestral_eta_schedule_series[frame_idx])
        if args.use_mask and keys.mask_schedule_series[frame_idx] is not None:
            mask_seq = keys.mask_schedule_series[frame_idx]
        if anim_args.use_noise_mask and keys.noise_mask_schedule_series[frame_idx] is not None:
            noise_mask_seq = keys.noise_mask_schedule_series[frame_idx]

        if args.use_mask and not anim_args.use_noise_mask:
            noise_mask_seq = mask_seq

        depth = None

        if anim_args.animation_mode == '3D' and (cmd_opts.lowvram or cmd_opts.medvram):
            # Unload the main checkpoint and load the depth model
            lowvram.send_everything_to_cpu()
            sd_hijack.model_hijack.undo_hijack(sd_model)
            devices.torch_gc()
            if predict_depths: depth_model.to(root.device)

        if turbo_steps == 1 and opts.data.get("deforum_save_gen_info_as_srt"):
            params_string = format_animation_params(keys, prompt_series, frame_idx)
            write_frame_subtitle(srt_filename, frame_idx, srt_frame_duration, f"F#: {frame_idx}; Cadence: false; Seed: {args.seed}; {params_string}")
            params_string = None

        # emit in-between frames - at cadence > 1 frames all saved in this loop, at cadence 1 frames are saved after generation
        if turbo_steps > 1 and not resume_skip_cadence:
            # setup vars for the start of each cadence loop
            tween_frame_start_idx = max(0, frame_idx - turbo_steps)
            cadence_flow = None

            for tween_frame_idx in range(tween_frame_start_idx, frame_idx):
                # update ui progress during cadence
                state.job = f"frame {tween_frame_idx + 1}/{anim_args.max_frames}"
                state.job_no = tween_frame_idx + 1

                # vars that need refreshing within cadence loop
                advance_prev = turbo_prev_image is not None and tween_frame_idx > turbo_prev_frame_idx
                advance_next = tween_frame_idx > turbo_next_frame_idx
                diffusion_cadence_easing, cadence_flow_easing, hybrid_flow_factor, cadence_flow_factor = get_cadence_keys(keys, tween_frame_idx)

                # eased tweens locked with master tween
                tween, tween_images, tween_flow = get_cadence_tweens(tween_frame_idx, tween_frame_start_idx, frame_idx, diffusion_cadence_easing, cadence_flow_easing)

                # load depth model
                if depth_model is not None:
                    assert (turbo_next_image is not None)
                    depth = depth_model.predict(turbo_next_image, anim_args.midas_weight, root.half_precision)

                # srt files
                if opts.data.get("deforum_save_gen_info_as_srt"):
                    params_string = format_animation_params(keys, prompt_series, tween_frame_idx)
                    write_frame_subtitle(srt_filename, tween_frame_idx, srt_frame_duration, f"F#: {tween_frame_idx}; Cadence: {tween < 1.0}; Seed: {args.seed}; {params_string}")
                    params_string = None

                # setup cadence_flow once for whole cadence cycle if turbo prev/next are provided, get cadence flow and turbo_next_image warped back half-way
                if anim_args.optical_flow_cadence != 'None' and cadence_flow is None and turbo_prev_image is not None and turbo_next_image is not None:
                    cadence_flow, turbo_next_image = cadence_optflow_setup(anim_args, cadence_flow, turbo_prev_image, turbo_next_image, raft_model)

                # setup done, report cadence frame creation in console
                print(f"Creating in-between {'' if cadence_flow is None else anim_args.optical_flow_cadence + ' optical flow '}cadence frame: {tween_frame_idx}; tween:{tween:0.2f}; tween_images:{tween_images:0.2f}; tween_flow:{tween_flow:0.2f};")

                # cadence flow is warped to match animation warp & divided into increments each cadence step
                if cadence_flow is not None:
                    cadence_flow_inc = cadence_optflow_warp_blend(cadence_flow, tween_flow, args, anim_args, keys, tween_frame_idx, depth_model, depth=depth, device=root.device, half_precision=root.half_precision)

                # animation warping of both images
                if advance_prev:
                    turbo_prev_image, _ = anim_frame_warp(turbo_prev_image, args, anim_args, keys, tween_frame_idx, depth_model, depth=depth, device=root.device, half_precision=root.half_precision)
                if advance_next:
                    turbo_next_image, _ = anim_frame_warp(turbo_next_image, args, anim_args, keys, tween_frame_idx, depth_model, depth=depth, device=root.device, half_precision=root.half_precision)

                # hybrid motion for cadence - after animation warping, perspective/affine/optflow from last frame to current frame
                if anim_args.hybrid_motion != 'None' and tween_frame_idx > 0 and tween < 1:
                    turbo_prev_image, turbo_next_image, prev_flow = hybrid_motion_for_cadence(turbo_prev_image, turbo_next_image, advance_prev, advance_next, args, anim_args,
                                                                                              tween_frame_idx, prev_img, prev_flow, hybrid_flow_factor, inputfiles, hybrid_frame_path, raft_model)

                # previous image flow - uses flow from a past 'prev_imgs array image to the prev_img to warp the prev_img.
                if anim_args.previous_image_flow != 'None' and len(prev_imgs) > 1 and previous_image_flow_factor != 0:
                    flow = get_flow_from_images(prev_imgs[-1], prev_imgs[0], anim_args.previous_image_flow, raft_model)
                    turbo_next_image = image_transform_optical_flow(turbo_next_image, flow, previous_image_flow_factor)
                    print(f"Previous image flow {anim_args.previous_image_flow} frame {tween_frame_idx-len(prev_imgs)} to {tween_frame_idx-1} applied to turbo_next_image at flow factor {previous_image_flow_factor}")

                # blend cadence images with eased tweening
                if turbo_prev_image is not None and tween < 1.0:
                    img = turbo_prev_image * (1.0 - tween_images) + turbo_next_image * tween_images
                else:
                    img = turbo_next_image

                # Optical flow cadence application directly after blend of cadence images
                if cadence_flow is not None and cadence_flow_factor != 0:
                    img = image_transform_optical_flow(img, cadence_flow_inc, cadence_flow_factor)

                # intercept and override to grayscale before overlay
                if anim_args.color_force_grayscale:
                    img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2GRAY)
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

                # overlay mask goes last
                if args.overlay_mask and (anim_args.use_mask_video or args.use_mask):
                    img = do_overlay_mask(args, anim_args, img, tween_frame_idx, True)

                # update cadence frame index now that all frame-dependent routines are done
                turbo_prev_frame_idx = turbo_next_frame_idx = tween_frame_idx

                # cadence keeps a prev_img too
                prev_img = img

                # prev_imgs array keeps prev_img also (will replace prev_img)
                prev_imgs.insert(0, img.astype(np.uint8))
                while len(prev_imgs) > prev_imgs_max:
                    prev_imgs.pop()

                # current image update for cadence frames (commented because it doesn't currently update the preview)
                # state.current_image = Image.fromarray(cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2RGB))

                # saving cadence frames
                filename = f"{root.timestring}_{tween_frame_idx:09}.png"
                cv2.imwrite(os.path.join(args.outdir, filename), img)
                if anim_args.save_depth_maps:
                    depth_model.save(os.path.join(args.outdir, f"{root.timestring}_depth_{tween_frame_idx:09}.png"), depth)

        # END CADENCE LOOP

        # if reached end of first cadence loop, the skip cadence trigger can be turned off
        if resume_skip_cadence:
            resume_skip_cadence = False

        # get color match for video outside of prev_img conditional
        if anim_args.color_coherence == 'Video Input' and hybrid_available:
            if int(frame_idx) % int(anim_args.color_coherence_video_every_N_frames) == 0:
                prev_vid_img = Image.open(os.path.join(args.outdir, 'inputframes', get_frame_name(anim_args.video_init_path) + f"{frame_idx:09}.jpg"))
                prev_vid_img = prev_vid_img.resize((args.W, args.H), PIL.Image.LANCZOS)
                color_match_sample = np.asarray(prev_vid_img)
                color_match_sample = cv2.cvtColor(color_match_sample, cv2.COLOR_RGB2BGR)

        # after 1st frame, prev_img exists
        if prev_img is not None:
            # apply transforms to previous frame
            prev_img, depth = anim_frame_warp(prev_img, args, anim_args, keys, frame_idx, depth_model, depth=None, device=root.device, half_precision=root.half_precision)

            # do hybrid compositing before motion
            if anim_args.hybrid_composite == 'Before Motion':
                args, prev_img = hybrid_composite(args, anim_args, frame_idx, prev_img, depth_model, hybrid_comp_schedules, root)

            # hybrid video motion: Before Generation (normal) warps prev_img to match motion from last frame to current frame
            if anim_args.hybrid_motion != 'None' and anim_args.hybrid_motion_behavior == 'Before Generation':
                prev_img, prev_flow = hybrid_motion_before_generation(prev_img, prev_flow, frame_idx, args, anim_args, inputfiles, hybrid_frame_path, hybrid_flow_factor, raft_model)

            # do hybrid compositing after motion (normal)
            if anim_args.hybrid_composite == 'Normal':
                args, prev_img = hybrid_composite(args, anim_args, frame_idx, prev_img, depth_model, hybrid_comp_schedules, root)

            # previous image flow - uses flow from 'prev_imgs array images to prev_img' to warp the prev_img. Doesn't require extra generation!
            if turbo_steps == 1 and anim_args.previous_image_flow != 'None' and prev_img is not None and len(prev_imgs) > 1 and previous_image_flow_factor != 0:
                flow = get_flow_from_images(prev_imgs[-1], prev_imgs[0], anim_args.previous_image_flow, raft_model)
                prev_img = image_transform_optical_flow(prev_img, flow, previous_image_flow_factor)
                print(f"Previous image flow {anim_args.previous_image_flow} frame {frame_idx-len(prev_imgs)} to {frame_idx-1} applied at flow factor {previous_image_flow_factor}")

            # apply color matching - if no color match, make one for next frame
            if anim_args.color_coherence != 'None':
                if color_match_sample is None:
                    color_match_sample = prev_img.copy()
                else:
                    prev_img = maintain_colors(prev_img, color_match_sample, anim_args.color_coherence)

            # intercept and override to grayscale
            if anim_args.color_force_grayscale:
                prev_img = cv2.cvtColor(prev_img, cv2.COLOR_BGR2GRAY)
                prev_img = cv2.cvtColor(prev_img, cv2.COLOR_GRAY2BGR)

            # apply scaling
            contrast_image = (prev_img * contrast).round().astype(np.uint8)
            # anti-blur
            if amount > 0:
                contrast_image = unsharp_mask(contrast_image, (kernel, kernel), sigma, amount, threshold, mask_image if args.use_mask else None)
            # apply frame noising
            if args.use_mask or anim_args.use_noise_mask:
                root.noise_mask = compose_mask_with_check(root, args, noise_mask_seq, noise_mask_vals, Image.fromarray(cv2.cvtColor(contrast_image, cv2.COLOR_BGR2RGB)))
            noised_image = add_noise(contrast_image, noise, args.seed, anim_args.noise_type,
                                     (anim_args.perlin_w, anim_args.perlin_h, anim_args.perlin_octaves, anim_args.perlin_persistence),
                                     root.noise_mask, args.invert_mask)

            # use transformed previous frame as init for current
            args.use_init = True
            root.init_sample = Image.fromarray(cv2.cvtColor(noised_image, cv2.COLOR_BGR2RGB))
            args.strength = max(0.0, min(1.0, strength))

        args.scale = scale

        # Pix2Pix Image CFG Scale - does *nothing* with non pix2pix checkpoints
        args.pix2pix_img_cfg_scale = float(keys.pix2pix_img_cfg_scale_series[frame_idx])

        # grab prompt for current frame
        args.prompt = prompt_series[frame_idx]

        if args.seed_behavior == 'schedule' or parseq_adapter.manages_seed():
            args.seed = int(keys.seed_schedule_series[frame_idx])

        if anim_args.enable_checkpoint_scheduling:
            args.checkpoint = keys.checkpoint_schedule_series[frame_idx]
        else:
            args.checkpoint = None

        # SubSeed scheduling
        if anim_args.enable_subseed_scheduling:
            root.subseed = int(keys.subseed_schedule_series[frame_idx])
            root.subseed_strength = float(keys.subseed_strength_schedule_series[frame_idx])

        if parseq_adapter.manages_seed():
            anim_args.enable_subseed_scheduling = True
            root.subseed = int(keys.subseed_schedule_series[frame_idx])
            root.subseed_strength = keys.subseed_strength_schedule_series[frame_idx]

        # set value back into the prompt - prepare and report prompt and seed
        args.prompt = prepare_prompt(args.prompt, anim_args.max_frames, args.seed, frame_idx)

        # grab init image for current frame
        if using_vid_init:
            init_frame = get_next_frame(args.outdir, anim_args.video_init_path, frame_idx, False)
            print(f"Using video init frame {init_frame}")
            args.init_image = init_frame
            args.init_image_box = None  # init_image_box not used in this case
            args.strength = max(0.0, min(1.0, strength))
        if anim_args.use_mask_video:
            args.mask_file = get_mask_from_file(get_next_frame(args.outdir, anim_args.video_mask_path, frame_idx, True), args)
            root.noise_mask = get_mask_from_file(get_next_frame(args.outdir, anim_args.video_mask_path, frame_idx, True), args)

            mask_vals['video_mask'] = get_mask_from_file(get_next_frame(args.outdir, anim_args.video_mask_path, frame_idx, True), args)

        if args.use_mask:
            args.mask_image = compose_mask_with_check(root, args, mask_seq, mask_vals, root.init_sample) if root.init_sample is not None else None  # we need it only after the first frame anyway

        # setting up some arguments for the looper
        loop_args.imageStrength = loopSchedulesAndData.image_strength_schedule_series[frame_idx]
        loop_args.blendFactorMax = loopSchedulesAndData.blendFactorMax_series[frame_idx]
        loop_args.blendFactorSlope = loopSchedulesAndData.blendFactorSlope_series[frame_idx]
        loop_args.tweeningFrameSchedule = loopSchedulesAndData.tweening_frames_schedule_series[frame_idx]
        loop_args.colorCorrectionFactor = loopSchedulesAndData.color_correction_factor_series[frame_idx]
        loop_args.use_looper = loopSchedulesAndData.use_looper
        loop_args.imagesToKeyframe = loopSchedulesAndData.imagesToKeyframe

        if 'img2img_fix_steps' in opts.data and opts.data["img2img_fix_steps"]:  # disable "with img2img do exactly x steps" from general setting, as it *ruins* deforum animations
            opts.data["img2img_fix_steps"] = False
        if scheduled_clipskip is not None:
            opts.data["CLIP_stop_at_last_layers"] = scheduled_clipskip
        if scheduled_noise_multiplier is not None:
            opts.data["initial_noise_multiplier"] = scheduled_noise_multiplier
        if scheduled_ddim_eta is not None:
            opts.data["eta_ddim"] = scheduled_ddim_eta
        if scheduled_ancestral_eta is not None:
            opts.data["eta_ancestral"] = scheduled_ancestral_eta

        if anim_args.animation_mode == '3D' and (cmd_opts.lowvram or cmd_opts.medvram):
            if predict_depths: depth_model.to('cpu')
            devices.torch_gc()
            lowvram.setup_for_low_vram(sd_model, cmd_opts.medvram)
            sd_hijack.model_hijack.hijack(sd_model)

        # Optical flow generation, before generation
        if anim_args.optical_flow_redo_generation != 'None' and prev_img is not None:
            root.init_sample = optical_flow_generation(prev_img, redo_flow_factor, raft_model, args, keys, anim_args, loop_args, controlnet_args, root, frame_idx, sampler_name=scheduled_sampler_name)

        # Redo generation, before generation
        if int(anim_args.diffusion_redo) > 0 and prev_img is not None:
            root.init_sample = redo_generation(prev_img, args, keys, anim_args, loop_args, controlnet_args, root, frame_idx, sampler_name=scheduled_sampler_name)

        # generation
        image = generate(args, keys, anim_args, loop_args, controlnet_args, root, frame_idx, sampler_name=scheduled_sampler_name)

        if image is None:
            break

        # RGB to BGR for this section ---------------------------------------
        image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        # hybrid video compositing: After generation
        if anim_args.hybrid_composite == 'After Generation':
            args, image = hybrid_composite(args, anim_args, frame_idx, image, depth_model, hybrid_comp_schedules, root)

        # hybrid video motion: After generation, flow from current frame to next frame (skips last frame)
        if anim_args.hybrid_motion != 'None' and anim_args.hybrid_motion_behavior == 'After Generation' and frame_idx < anim_args.max_frames:
            image, prev_flow = hybrid_motion_after_generation(image, prev_img, prev_flow, frame_idx, args, anim_args, inputfiles, hybrid_frame_path, hybrid_flow_factor, raft_model)

        # color matching for Image/Video on 1st frame is after generation, legacy or not
        # non-legacy color matching also happens after generation every frame
        color_match_image_first_frame = (frame_idx == 0) and (anim_args.color_coherence == 'Image' or (anim_args.color_coherence == 'Video Input' and hybrid_available))
        color_not_legacy = not anim_args.legacy_colormatch and color_match_sample is not None and anim_args.color_coherence != 'None'
        if color_match_image_first_frame or color_not_legacy:
            image = maintain_colors(image, color_match_sample, anim_args.color_coherence)

        # on 1st frame, get a color match after 1st generation, unless Image or Video Input (color sample done elsewhere for those)
        if frame_idx == 0 and not anim_args.color_coherence in ['Image', 'Video Input']:
            color_match_sample = image

        # BGR back to RGB ---------------------------------------
        image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        # intercept and override to grayscale before overlay (PIL RGB functions)
        if anim_args.color_force_grayscale:
            image = ImageOps.grayscale(image)
            image = ImageOps.colorize(image, black="black", white="white")

        # overlay mask last before opencv_image/prev_img are captured
        if args.overlay_mask and (anim_args.use_mask_video or args.use_mask):
            image = do_overlay_mask(args, anim_args, image, frame_idx)

        # convert to opencv for end routines - for prev_img capture, and for file saving of cadence 1 frames 
        opencv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        # important: cadence makes it's own prev_img. If you also make one here, the cadence prev_img can't work the same
        # 'Video Input' animation mode just doesn't use prev_img
        # if turbo_steps == 1 or not using_vid_init:
        if not using_vid_init:
            prev_img = opencv_image

            # prev_imgs array for 'Previous image flow'
            prev_imgs.insert(0, opencv_image.astype(np.uint8))
            while len(prev_imgs) > prev_imgs_max:
                prev_imgs.pop()

        # if turbo_steps above 1, it saves all files within cadence loop in next cycle using these vars
        if turbo_steps > 1:
            turbo_prev_image, turbo_prev_frame_idx = turbo_next_image, turbo_next_frame_idx
            turbo_next_image, turbo_next_frame_idx = opencv_image, frame_idx
        else: # if turbo_steps is 1, saves files here
            filename = f"{root.timestring}_{frame_idx:09}.png"
            save_image(image, 'PIL', filename, args, video_args, root)

            if anim_args.save_depth_maps:
                if cmd_opts.lowvram or cmd_opts.medvram:
                    lowvram.send_everything_to_cpu()
                    sd_hijack.model_hijack.undo_hijack(sd_model)
                    devices.torch_gc()
                    depth_model.to(root.device)
                depth = depth_model.predict(opencv_image, anim_args.midas_weight, root.half_precision)
                depth_model.save(os.path.join(args.outdir, f"{root.timestring}_depth_{frame_idx:09}.png"), depth)
                if cmd_opts.lowvram or cmd_opts.medvram:
                    depth_model.to('cpu')
                    devices.torch_gc()
                    lowvram.setup_for_low_vram(sd_model, cmd_opts.medvram)
                    sd_hijack.model_hijack.hijack(sd_model)
        state.assign_current_image(image)

        print(f"\033[36mAnimation frame index complete:\033[0m {frame_idx}/{anim_args.max_frames} ")

        last_preview_frame = render_preview(args, anim_args, video_args, root, frame_idx, last_preview_frame)            

        JobStatusTracker().update_phase(root.job_id, phase="GENERATING", progress=frame_idx/anim_args.max_frames)

        # advance the seed
        args.seed = next_seed(args, root)

    # remove things from vram
    if predict_depths and not keep_in_vram:
        depth_model.delete_model()  # handles adabins too

    if load_raft:
        raft_model.delete_model()
