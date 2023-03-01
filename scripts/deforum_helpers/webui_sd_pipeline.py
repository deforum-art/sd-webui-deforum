from modules.processing import StableDiffusionProcessingImg2Img
from modules.shared import opts, sd_model
import os

def get_webui_sd_pipeline(args, root, frame):
    import re
    assert args.prompt is not None
    
    # Setup the pipeline
    p = StableDiffusionProcessingImg2Img(
        sd_model=sd_model,
        outpath_samples = opts.outdir_samples or opts.outdir_img2img_samples,
        #we'll setup the rest later
    )
    
    os.makedirs(args.outdir, exist_ok=True)
    p.width, p.height = map(lambda x: x - x % 64, (args.W, args.H))
    p.steps = args.steps
    p.seed = args.seed
    p.sampler_name = args.sampler
    p.batch_size = args.n_batch
    p.tiling = args.tiling
    p.restore_faces = args.restore_faces
    p.subseed = args.subseed
    p.subseed_strength = args.subseed_strength
    p.seed_resize_from_w = args.seed_resize_from_w
    p.seed_resize_from_h = args.seed_resize_from_h
    p.fill = args.fill
    p.ddim_eta = args.ddim_eta
    p.batch_size = args.n_samples
    p.width = args.W
    p.height = args.H
    p.seed = args.seed
    p.do_not_save_samples = not args.save_sample_per_step
    p.sampler_name = args.sampler
    p.mask_blur = args.mask_overlay_blur
    p.extra_generation_params["Mask blur"] = args.mask_overlay_blur
    p.n_iter = 1
    p.steps = args.steps
    if opts.img2img_fix_steps:
        p.denoising_strength = 1 / (1 - args.strength + 1.0/args.steps) #see https://github.com/deforum-art/deforum-for-automatic1111-webui/issues/3
    else:
        p.denoising_strength = 1 - args.strength
    p.cfg_scale = args.scale
    p.image_cfg_scale = args.pix2pix_img_cfg_scale
    p.outpath_samples = root.outpath_samples
    

    return p