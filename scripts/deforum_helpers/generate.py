import numpy as np
import cv2
from PIL import Image
from .prompt import split_weighted_subprompts
from .load_images import load_img, prepare_mask, check_mask_for_errors
from .webui_sd_pipeline import get_webui_sd_pipeline
from .animation import sample_from_cv2, sample_to_cv2
from .rich import console
#Webui
import cv2
from .animation import sample_from_cv2, sample_to_cv2
from modules import processing, sd_models
from modules.shared import opts, sd_model
from modules.processing import process_images, StableDiffusionProcessingTxt2Img
from .deforum_controlnet import is_controlnet_enabled, process_txt2img_with_controlnet, process_img2img_with_controlnet

import math, json, itertools
import requests

def load_mask_latent(mask_input, shape):
    # mask_input (str or PIL Image.Image): Path to the mask image or a PIL Image object
    # shape (list-like len(4)): shape of the image to match, usually latent_image.shape
    
    if isinstance(mask_input, str): # mask input is probably a file name
        if mask_input.startswith('http://') or mask_input.startswith('https://'):
            mask_image = Image.open(requests.get(mask_input, stream=True).raw).convert('RGBA')
        else:
            mask_image = Image.open(mask_input).convert('RGBA')
    elif isinstance(mask_input, Image.Image):
        mask_image = mask_input
    else:
        raise Exception("mask_input must be a PIL image or a file name")

    mask_w_h = (shape[-1], shape[-2])
    mask = mask_image.resize(mask_w_h, resample=Image.LANCZOS)
    mask = mask.convert("L")
    return mask

def isJson(myjson):
    try:
        json.loads(myjson)
    except ValueError as e:
        return False
    return True

# Add pairwise implementation here not to upgrade
# the whole python to 3.10 just for one function
def pairwise_repl(iterable):
    a, b = itertools.tee(iterable)
    next(b, None)
    return zip(a, b)

def generate(args, keys, anim_args, loop_args, controlnet_args, root, frame = 0, return_sample=False, sampler_name=None):
    assert args.prompt is not None

    # Setup the pipeline
    p = get_webui_sd_pipeline(args, root, frame)
    p.prompt, p.negative_prompt = split_weighted_subprompts(args.prompt, frame)
    
    if not args.use_init and args.strength > 0 and args.strength_0_no_init:
        print("\nNo init image, but strength > 0. Strength has been auto set to 0, since use_init is False.")
        print("If you want to force strength > 0 with no init, please set strength_0_no_init to False.\n")
        args.strength = 0
    processed = None
    mask_image = None
    init_image = None
    image_init0 = None

    if loop_args.use_looper and anim_args.animation_mode in ['2D','3D']:
        # TODO find out why we need to set this in the init tab
        if args.strength == 0:
            raise RuntimeError("Strength needs to be greater than 0 in Init tab and strength_0_no_init should *not* be checked")
        if args.seed_behavior != "schedule":
            raise RuntimeError("seed_behavior needs to be set to schedule in under 'Keyframes' tab --> 'Seed scheduling'")
        if not isJson(loop_args.imagesToKeyframe):
            raise RuntimeError("The images set for use with keyframe-guidance are not in a proper JSON format")
        args.strength = loop_args.imageStrength
        tweeningFrames = loop_args.tweeningFrameSchedule
        blendFactor = .07
        colorCorrectionFactor = loop_args.colorCorrectionFactor
        jsonImages = json.loads(loop_args.imagesToKeyframe)
        framesToImageSwapOn = list(map(int, list(jsonImages.keys())))
        # find which image to show
        frameToChoose = 0
        for swappingFrame in framesToImageSwapOn[1:]:
            frameToChoose += (frame >= int(swappingFrame))
        
        #find which frame to do our swapping on for tweening
        skipFrame = 25
        for fs, fe in pairwise_repl(framesToImageSwapOn):
            if fs <= frame <= fe:
                skipFrame = fe - fs

        if frame % skipFrame <= tweeningFrames: # number of tweening frames
            blendFactor = loop_args.blendFactorMax - loop_args.blendFactorSlope*math.cos((frame % tweeningFrames) / (tweeningFrames / 2))
        init_image2, _ = load_img(list(jsonImages.values())[frameToChoose],
                                shape=(args.W, args.H),
                                use_alpha_as_mask=args.use_alpha_as_mask)
        image_init0 = list(jsonImages.values())[0]
            
    else: # they passed in a single init image
        image_init0 = args.init_image


    available_samplers = { 
        'euler a':'Euler a',
        'euler':'Euler',
        'lms':'LMS',
        'heun':'Heun',
        'dpm2':'DPM2',
        'dpm2 a':'DPM2 a',
        'dpm++ 2s a':'DPM++ 2S a',
        'dpm++ 2m':'DPM++ 2M',
        'dpm++ sde':'DPM++ SDE',
        'dpm fast':'DPM fast',
        'dpm adaptive':'DPM adaptive',
        'lms karras':'LMS Karras' ,
        'dpm2 karras':'DPM2 Karras',
        'dpm2 a karras':'DPM2 a Karras',
        'dpm++ 2s a karras':'DPM++ 2S a Karras',
        'dpm++ 2m karras':'DPM++ 2M Karras',
        'dpm++ sde karras':'DPM++ SDE Karras'
    }
    if sampler_name is not None:
        if sampler_name in available_samplers.keys():
            p.sampler_name = available_samplers[sampler_name]
        else:
            raise RuntimeError(f"Sampler name '{sampler_name}' is invalid. Please check the available sampler list in the 'Run' tab")

    if args.checkpoint is not None:
        info = sd_models.get_closet_checkpoint_match(args.checkpoint)
        if info is None:
            raise RuntimeError(f"Unknown checkpoint: {args.checkpoint}")
        sd_models.reload_model_weights(info=info)
    
    if args.init_sample is not None:
        # TODO: cleanup init_sample remains later
        img = args.init_sample
        init_image = img
        image_init0 = img
        if loop_args.use_looper and isJson(loop_args.imagesToKeyframe) and anim_args.animation_mode in ['2D','3D']:
            init_image = Image.blend(init_image, init_image2, blendFactor)
            correction_colors = Image.blend(init_image, init_image2, colorCorrectionFactor)
            p.color_corrections = [processing.setup_color_correction(correction_colors)]

    # this is the first pass
    elif (loop_args.use_looper and anim_args.animation_mode in ['2D','3D']) or (args.use_init and ((args.init_image != None and args.init_image != ''))):
        init_image, mask_image = load_img(image_init0, # initial init image
                                          shape=(args.W, args.H),  
                                          use_alpha_as_mask=args.use_alpha_as_mask)
                                          
    else:
        
        if anim_args.animation_mode != 'Interpolation':
            print(f"Not using an init image (doing pure txt2img)")
        p_txt = StableDiffusionProcessingTxt2Img(
                sd_model=sd_model,
                outpath_samples=root.tmp_deforum_run_duplicated_folder,
                outpath_grids=root.tmp_deforum_run_duplicated_folder,
                prompt=p.prompt,
                styles=p.styles,
                negative_prompt=p.negative_prompt,
                seed=p.seed,
                subseed=p.subseed,
                subseed_strength=p.subseed_strength,
                seed_resize_from_h=p.seed_resize_from_h,
                seed_resize_from_w=p.seed_resize_from_w,
                sampler_name=p.sampler_name,
                batch_size=p.batch_size,
                n_iter=p.n_iter,
                steps=p.steps,
                cfg_scale=p.cfg_scale,
                width=p.width,
                height=p.height,
                restore_faces=p.restore_faces,
                tiling=p.tiling,
                enable_hr=None,
                denoising_strength=None,
            )
            
        print_combined_table(args, anim_args, p_txt, keys, frame) # print dynamic table to cli

        if is_controlnet_enabled(controlnet_args):
            processed = process_txt2img_with_controlnet(p, args, anim_args, loop_args, controlnet_args, root, frame)
        else:
            processed = processing.process_images(p_txt)

    if processed is None:
        # Mask functions
        if args.use_mask:
            mask = args.mask_image
            #assign masking options to pipeline
            if mask is not None:
                p.inpainting_mask_invert = args.invert_mask
                p.inpainting_fill = args.fill 
                p.inpaint_full_res= args.full_res_mask 
                p.inpaint_full_res_padding = args.full_res_mask_padding
        else:
            mask = None

        assert not ( (mask is not None and args.use_mask and args.overlay_mask) and (args.init_sample is None and init_image is None)), "Need an init image when use_mask == True and overlay_mask == True"
        
        p.init_images = [init_image]
        p.image_mask = mask
        p.image_cfg_scale = args.pix2pix_img_cfg_scale
        
        print_combined_table(args, anim_args, p, keys, frame) # print dynamic table to cli
       
        if is_controlnet_enabled(controlnet_args):
            processed = process_img2img_with_controlnet(p, args, anim_args, loop_args, controlnet_args, root, frame)
        else:
            processed = processing.process_images(p)
    
    if root.initial_info == None:
        root.initial_seed = processed.seed
        root.initial_info = processed.info
        
    if root.first_frame == None:
        root.first_frame = processed.images[0]
    
    results = processed.images[0]
    
    return results
    
def print_combined_table(args, anim_args, p, keys, frame_idx):
    from rich.table import Table
    from rich import box

    table = Table(padding=0, box=box.ROUNDED)
    
    field_names1 = ["Steps", "CFG"]
    if anim_args.animation_mode != 'Interpolation':
        field_names1.append("Denoise")
    field_names1 += ["Subseed", "Subs. str"] * (anim_args.enable_subseed_scheduling)
    field_names1 += ["Sampler"] * anim_args.enable_sampler_scheduling
    field_names1 += ["Checkpoint"] * anim_args.enable_checkpoint_scheduling
    
    for field_name in field_names1:
        table.add_column(field_name, justify="center")

    rows1 = [str(p.steps), str(p.cfg_scale)]
    if anim_args.animation_mode != 'Interpolation':
        rows1.append(f"{p.denoising_strength:.5g}" if p.denoising_strength is not None else "None")

    rows1 += [str(p.subseed), f"{p.subseed_strength:.5g}"] * (anim_args.enable_subseed_scheduling)
    rows1 += [p.sampler_name] * anim_args.enable_sampler_scheduling
    rows1 += [str(args.checkpoint)] * anim_args.enable_checkpoint_scheduling
    
    rows2 = []
    if anim_args.animation_mode not in ['Video Input', 'Interpolation']:
        if anim_args.animation_mode == '2D':
            field_names2 = ["Angle", "Zoom"]
        else:
            field_names2 = []
        field_names2 += ["Tr X", "Tr Y"]
        if anim_args.animation_mode == '3D':
            field_names2 += ["Tr Z", "Ro X", "Ro Y", "Ro Z"]
            if anim_args.aspect_ratio_schedule.replace(" ", "") != '0:(1)':
                field_names2 += ["Asp. Ratio"]
        if anim_args.enable_perspective_flip:
            field_names2 += ["Pf T", "Pf P", "Pf G", "Pf F"]
        
        for field_name in field_names2:
            table.add_column(field_name, justify="center")

        if anim_args.animation_mode == '2D':
            rows2 += [f"{keys.angle_series[frame_idx]:.5g}", f"{keys.zoom_series[frame_idx]:.5g}"]
        rows2 += [f"{keys.translation_x_series[frame_idx]:.5g}", f"{keys.translation_y_series[frame_idx]:.5g}"]
 
        if anim_args.animation_mode == '3D':
            rows2 += [f"{keys.translation_z_series[frame_idx]:.5g}", f"{keys.rotation_3d_x_series[frame_idx]:.5g}",
                      f"{keys.rotation_3d_y_series[frame_idx]:.5g}", f"{keys.rotation_3d_z_series[frame_idx]:.5g}"]
            if anim_args.aspect_ratio_schedule.replace(" ", "") != '0:(1)':
                rows2 += [f"{keys.aspect_ratio_series[frame_idx]:.5g}"]
        if anim_args.enable_perspective_flip:
            rows2 += [f"{keys.perspective_flip_theta_series[frame_idx]:.5g}", f"{keys.perspective_flip_phi_series[frame_idx]:.5g}",
                      f"{keys.perspective_flip_gamma_series[frame_idx]:.5g}", f"{keys.perspective_flip_fv_series[frame_idx]:.5g}"]

    table.add_row(*rows1, *rows2)
    console.print(table)