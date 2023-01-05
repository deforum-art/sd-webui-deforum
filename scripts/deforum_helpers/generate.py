import numpy as np
import torchvision.transforms.functional as TF
from pytorch_lightning import seed_everything
import os
from ldm.models.diffusion.plms import PLMSSampler
from ldm.models.diffusion.ddim import DDIMSampler
from k_diffusion.external import CompVisDenoiser
from torch import autocast
from contextlib import nullcontext
from einops import rearrange
import random
from .prompt import get_uc_and_c, parse_weight
from .k_samplers import sampler_fn
from scipy.ndimage import gaussian_filter

from .callback import SamplerCallback

#Webui
import cv2
from PIL import Image
from .prompt import split_weighted_subprompts
from .load_images import load_img, prepare_mask, check_mask_for_errors
from .webui_sd_pipeline import get_webui_sd_pipeline
from .animation import sample_from_cv2, sample_to_cv2

#Webui
from modules import processing
from modules.shared import sd_model
from modules.processing import StableDiffusionProcessingTxt2Img
    
def generate(args, anim_args, root, frame = 0, return_sample=False, sampler_name=None):
    import re
    assert args.prompt is not None
    
    # Setup the pipeline
    p = get_webui_sd_pipeline(args, root, frame)
    p.prompt, p.negative_prompt = split_weighted_subprompts(args.prompt, frame)
    p = root.p
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
        'dpm fase':'DPM fast',
        'dpm adaptive':'DPM adaptive',
        'lms karras':'LMS Karras' ,
        'dpm2 karras':'DPM2 Karras',
        'dpm2 a karras':'DPM2 a Karras',
        'dpm++ 2s a karras':'DPM++ 2S a Karras',
        'dpm++ 2m karras':'DPM++ 2M Karras',
        'dpm++ sde karras':'DPM++ SDE Karras'
    }

    if sampler_name in available_samplers.keys():
        args.sampler = available_samplers[sampler_name]

    os.makedirs(args.outdir, exist_ok=True)
    p.batch_size = args.n_samples
    p.width = args.W
    p.height = args.H
    p.seed = args.seed
    p.subseed=args.subseed
    p.subseed_strength=args.subseed_strength
    p.do_not_save_samples = not args.save_sample_per_step
    p.do_not_save_grid = not args.make_grid
    p.sd_model=sd_model
    p.sampler_name = args.sampler
    p.mask_blur = args.mask_overlay_blur
    p.extra_generation_params["Mask blur"] = args.mask_overlay_blur
    p.n_iter = 1
    p.steps = args.steps
    p.denoising_strength = 1 - args.strength
    p.cfg_scale = args.scale
    p.seed_enable_extras = args.seed_enable_extras

    # FIXME better color corrections as match histograms doesn't seem to be fully working
    if root.color_corrections is not None:
        p.color_corrections = root.color_corrections
    p.outpath_samples = root.outpath_samples
    p.outpath_grids = root.outpath_samples

    prompt_split = parsed_prompt.split("--neg")
    if len(prompt_split) > 1:
        p.prompt, p.negative_prompt = parsed_prompt.split("--neg") #TODO: add --neg to vanilla Deforum for compat
        print(f'Positive prompt:{p.prompt}')
        print(f'Negative prompt:{p.negative_prompt}')
    else:
        p.prompt = prompt_split[0]
        print(f'Positive prompt:{p.prompt}')
        p.negative_prompt = ""
    
    if not args.use_init and args.strength > 0 and args.strength_0_no_init:
        print("\nNo init image, but strength > 0. Strength has been auto set to 0, since use_init is False.")
        print("If you want to force strength > 0 with no init, please set strength_0_no_init to False.\n")
        args.strength = 0

    mask_image = None
    init_image = None
    processed = None
    
    if args.init_sample is not None:
        open_cv_image = sample_to_cv2(args.init_sample)
        img = cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2RGB)
        init_image = Image.fromarray(img)

    elif args.use_init and args.init_image != None and args.init_image != '':
        init_image, mask_image = load_img(args.init_image, 
                                          shape=(args.W, args.H),  
                                          use_alpha_as_mask=args.use_alpha_as_mask)
                                          
    else:
        print(f"Not using an init image (doing pure txt2img) - seed:{p.seed}; subseed:{p.subseed}; subseed_strength:{p.subseed_strength}; cfg_scale:{p.cfg_scale}; steps:{p.steps}")
        p_txt = StableDiffusionProcessingTxt2Img(
                sd_model=sd_model,
                outpath_samples=p.outpath_samples,
                outpath_grids=p.outpath_samples,
                prompt=p.prompt,
                styles=p.styles,
                negative_prompt=p.negative_prompt,
                seed=p.seed,
                subseed=p.subseed,
                subseed_strength=p.subseed_strength,
                seed_resize_from_h=p.seed_resize_from_h,
                seed_resize_from_w=p.seed_resize_from_w,
                seed_enable_extras=p.seed_enable_extras,
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
        processed = processing.process_images(p_txt)
    
    if processed is None:
        # Mask functions
        if args.use_mask:
            assert args.mask_file is not None or mask_image is not None, "use_mask==True: An mask image is required for a mask. Please enter a mask_file or use an init image with an alpha channel"
            assert args.use_init, "use_mask==True: use_init is required for a mask"
            mask = prepare_mask(args.mask_file if mask_image is None else mask_image, 
                                (args.W, args.H),
                                args.mask_contrast_adjust, 
                                args.mask_brightness_adjust)
            #prevent loaded mask from throwing errors in Image operations if completely black and crop and resize in webui pipeline
            #doing this after contrast and brightness adjustments to ensure that mask is not passed as black or blank
            mask = check_mask_for_errors(mask, args.invert_mask)
            args.noise_mask = mask
            #assign masking options to pipeline
            if mask is not None:
                p.inpainting_mask_invert = args.invert_mask
                p.inpainting_fill = args.fill 
                p.inpaint_full_res= args.full_res_mask 
                p.inpaint_full_res_padding = args.full_res_mask_padding
        else:
            mask = None

        assert not ( (args.use_mask and args.overlay_mask) and (args.init_sample is None and init_image is None)), "Need an init image when use_mask == True and overlay_mask == True"
        
        p.init_images = [init_image]
        p.image_mask = mask

        print(f"seed={p.seed}; subseed={p.subseed}; subseed_strength={p.subseed_strength}; denoising_strength={p.denoising_strength}; steps={p.steps}; cfg_scale={p.cfg_scale}; sampler={p.sampler_name}")
        processed = processing.process_images(p)
        p.sd_model=sd_model
    
    if root.initial_info == None:
        root.initial_seed = processed.seed
        root.initial_info = processed.info
    
    if root.first_frame == None:
        root.first_frame = processed.images[0]
        ### TODO: put the correct arg here.
        if anim_args.histogram_matching:
            root.color_corrections = [processing.setup_color_correction(root.first_frame)]
    
    if return_sample:
        pil_image = processed.images[0].convert('RGB') 
        open_cv_image = np.array(pil_image) 
        # Convert RGB to BGR 
        open_cv_image = open_cv_image[:, :, ::-1].copy() 
        image = sample_from_cv2(open_cv_image)
        results = [image, processed.images[0]]
    else:
        results = [processed.images[0]]
    
    return results
