from modules.processing import StableDiffusionProcessingImg2Img
from modules.shared import opts, sd_model
import os

def get_webui_sd_pipeline(args, root):
    # Set up the pipeline
    p = StableDiffusionProcessingImg2Img(
        sd_model=sd_model,
        outpath_samples = opts.outdir_samples or opts.outdir_img2img_samples,
    )    # we'll set up the rest later
    
    os.makedirs(args.outdir, exist_ok=True)
    p.width, p.height = map(lambda x: x - x % 8, (args.W, args.H))
    p.steps = args.steps
    p.seed = args.seed
    p.sampler_name = args.sampler
    p.tiling = args.tiling
    p.restore_faces = args.restore_faces
    p.subseed = root.subseed
    p.subseed_strength = root.subseed_strength
    p.seed_resize_from_w = args.seed_resize_from_w
    p.seed_resize_from_h = args.seed_resize_from_h
    p.fill = args.fill
    p.batch_size = 1  # b.size 1 as this is DEFORUM :)
    p.seed = args.seed
    p.do_not_save_samples = True  # Setting this to False will trigger webui's saving mechanism - and we will end up with duplicated files, and another folder within our destination folder - big no no.
    p.sampler_name = args.sampler
    p.mask_blur = args.mask_overlay_blur
    p.extra_generation_params["Mask blur"] = args.mask_overlay_blur
    p.n_iter = 1
    p.steps = args.steps
    p.denoising_strength = 1 - args.strength
    p.cfg_scale = args.scale
    p.image_cfg_scale = args.pix2pix_img_cfg_scale
    p.outpath_samples = args.outdir
    
    return p