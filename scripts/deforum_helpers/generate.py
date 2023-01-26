import numpy as np
import cv2
from PIL import Image
from .prompt import split_weighted_subprompts
from .load_images import load_img, prepare_mask, check_mask_for_errors
from .webui_sd_pipeline import get_webui_sd_pipeline
from .animation import sample_from_cv2, sample_to_cv2

#Webui
import cv2
from .animation import sample_from_cv2, sample_to_cv2
from modules import processing
from modules.shared import opts, sd_model
from modules.processing import process_images, StableDiffusionProcessingTxt2Img

from modules import processing, sd_models
from modules.shared import sd_model
from modules.processing import StableDiffusionProcessingTxt2Img

import math, json, itertools
import requests

def load_img(path, shape, use_alpha_as_mask=False):
    # use_alpha_as_mask: Read the alpha channel of the image as the mask image
    if path.startswith('http://') or path.startswith('https://'):
        image = Image.open(requests.get(path, stream=True).raw)
    else:
        image = Image.open(path)

    if use_alpha_as_mask:
        image = image.convert('RGBA')
    else:
        image = image.convert('RGB')

    image = image.resize(shape, resample=Image.LANCZOS)

    mask_image = None
    if use_alpha_as_mask:
        # Split alpha channel into a mask_image
        red, green, blue, alpha = Image.Image.split(image)
        mask_image = alpha.convert('L')
        image = image.convert('RGB')

    return image, mask_image #PIL image for auto's pipeline

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

def prepare_mask(mask_input, mask_shape, mask_brightness_adjust=1.0, mask_contrast_adjust=1.0, invert_mask=False):
    # mask_input (str or PIL Image.Image): Path to the mask image or a PIL Image object
    # shape (list-like len(4)): shape of the image to match, usually latent_image.shape
    # mask_brightness_adjust (non-negative float): amount to adjust brightness of the iamge, 
    #     0 is black, 1 is no adjustment, >1 is brighter
    # mask_contrast_adjust (non-negative float): amount to adjust contrast of the image, 
    #     0 is a flat grey image, 1 is no adjustment, >1 is more contrast
    
    mask = load_mask_latent(mask_input, mask_shape)

    # Mask brightness/contrast adjustments
    if mask_brightness_adjust != 1:
        mask = TF.adjust_brightness(mask, mask_brightness_adjust)
    if mask_contrast_adjust != 1:
        mask = TF.adjust_contrast(mask, mask_contrast_adjust)

    if invert_mask:
        mask = PIL.ImageOps.invert(mask)
    
    return mask

def isJson(myjson):
    try:
        json.loads(myjson)
    except ValueError as e:
        return False
    return True

def generate(args, anim_args, loop_args, root, frame = 0, return_sample=False, sampler_name=None):
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
    print("in the generate loop")
    print(loop_args)
    # some setup variables that should be broken out later
    if loop_args.useLooper and isJson(loop_args.imagesToKeyframe):
        tweeningFrames = loop_args.tweening_frames_schedule
        blendFactor = .07
        colorCorrectionFactor = loop_args.color_correction_factor
        jsonImages = json.loads(loop_args.imagesToKeyframe)
        framesToImageSwapOn = list(map(int, list(jsonImages.keys())))
        
        # find which image to show
        frameToChoose = 0
        for swappingFrame in framesToImageSwapOn[1:]:
            frameToChoose += (frame >= int(swappingFrame))
        
        #find which frame to do our swapping on for tweening
        skipFrame = 25
        for fs, fe in itertools.pairwise(framesToImageSwapOn):
            if fs <= frame <= fe:
                skipFrame = fe - fs

        if frame % skipFrame <= tweeningFrames: # number of tweening frames
            blendFactor = loop_args.blendFactorMax - loop_args.blendFactorSlope*math.cos((frame % tweeningFrames) / (tweeningFrames / 2))
        
        print(f"\nframe: {frame} - blend factor: {blendFactor:.5f} - strength:{args.strength:.5f} - denoising: {p.denoising_strength:.5f} - skipFrame: {skipFrame}\n")
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
            args.sampler = available_samplers[sampler_name]

    if args.checkpoint is not None:
        info = sd_models.get_closet_checkpoint_match(args.checkpoint)
        if info is None:
            raise RuntimeError(f"Unknown checkpoint: {args.checkpoint}")
        sd_models.reload_model_weights(info=info)



    if args.init_sample is not None:
        open_cv_image = sample_to_cv2(args.init_sample)
        img = cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2RGB)
        init_image = Image.fromarray(img)
        image_init0 = Image.fromarray(img)
        if args.init_image is not None and isJson(args.init_image):
            init_image = Image.blend(init_image, init_image2, blendFactor)
            correction_colors = Image.blend(init_image, init_image2, colorCorrectionFactor)
            p.color_corrections = [processing.setup_color_correction(correction_colors)]

    # this is the first pass
    elif args.use_init and ((args.init_image != None and args.init_image != '')):
        init_image, mask_image = load_img(image_init0, # initial init image
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
