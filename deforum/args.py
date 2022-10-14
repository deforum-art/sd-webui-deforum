from modules.shared import cmd_opts
import modules.shared as sh
import modules.paths as ph

def Root():
    device = sh.device
    models_path = ph.models_path + '/Deforum'
    half_precision = not cmd_opts.no_half
    p = None
    initial_seed = None
    initial_info = None
    first_frame = None
    prompts = None
    outpath_samples = ""
    animation_prompts = None
    color_corrections = None
    return locals()

def DeforumAnimArgs():

    #@markdown ####**Animation:**
    animation_mode = '2D' #@param ['None', '2D', '3D', 'Video Input', 'Interpolation'] {type:'string'}
    max_frames = 1000 #@param {type:"number"}
    border = 'replicate' #@param ['wrap', 'replicate'] {type:'string'}

    #@markdown ####**Motion Parameters:**
    angle = "0:(0)"#@param {type:"string"}
    zoom = "0:(1.04)"#@param {type:"string"}
    translation_x = "0:(10*sin(2*3.14*t/10))"#@param {type:"string"}
    translation_y = "0:(0)"#@param {type:"string"}
    translation_z = "0:(10)"#@param {type:"string"}
    rotation_3d_x = "0:(0)"#@param {type:"string"}
    rotation_3d_y = "0:(0)"#@param {type:"string"}
    rotation_3d_z = "0:(0)"#@param {type:"string"}
    flip_2d_perspective = False #@param {type:"boolean"}
    perspective_flip_theta = "0:(0)"#@param {type:"string"}
    perspective_flip_phi = "0:(t%15)"#@param {type:"string"}
    perspective_flip_gamma = "0:(0)"#@param {type:"string"}
    perspective_flip_fv = "0:(53)"#@param {type:"string"}
    noise_schedule = "0: (0.02)"#@param {type:"string"}
    strength_schedule = "0: (0.65)"#@param {type:"string"}
    contrast_schedule = "0: (1.0)"#@param {type:"string"}
#TODO
#    seed_schedule = "0: (-1)"
#    scale_schedule = "0: (7)"

    #@markdown ####**Coherence:**
    color_coherence = 'Match Frame 0 LAB' #@param ['None', 'Match Frame 0 HSV', 'Match Frame 0 LAB', 'Match Frame 0 RGB'] {type:'string'}
    diffusion_cadence = '1' #@param ['1','2','3','4','5','6','7','8'] {type:'string'}

    #@markdown ####**3D Depth Warping:**
    use_depth_warping = True #@param {type:"boolean"}
    midas_weight = 0.3#@param {type:"number"}
    near_plane = 200
    far_plane = 10000
    fov = 40#@param {type:"number"}
    padding_mode = 'border'#@param ['border', 'reflection', 'zeros'] {type:'string'}
    sampling_mode = 'bicubic'#@param ['bicubic', 'bilinear', 'nearest'] {type:'string'}
    save_depth_maps = False #@param {type:"boolean"}

    #@markdown ####**Video Input:**
    video_init_path ='/content/video_in.mp4'#@param {type:"string"}
    extract_nth_frame = 1#@param {type:"number"}
    overwrite_extracted_frames = True #@param {type:"boolean"}
    use_mask_video = False #@param {type:"boolean"}
    video_mask_path ='/content/video_in.mp4'#@param {type:"string"}

    #@markdown ####**Interpolation:**
    interpolate_key_frames = False #@param {type:"boolean"}
    interpolate_x_frames = 4 #@param {type:"number"}
    
    #@markdown ####**Resume Animation:**
    resume_from_timestring = False #@param {type:"boolean"}
    resume_timestring = "20220829210106" #@param {type:"string"}

    return locals()

def DeforumPrompts():
    return r"""[
    "a beautiful forest by Asher Brown Durand, trending on Artstation",
    "a beautiful portrait of a woman by Artgerm, trending on Artstation"
]
"""

def DeforumAnimPrompts():
    return r"""{
    "0": "a beautiful apple, trending on Artstation",
    "20": "a beautiful banana, trending on Artstation",
    "30": "a beautiful coconut, trending on Artstation",
    "40": "a beautiful durian, trending on Artstation"
}
"""

def DeforumArgs():
    #@markdown **Image Settings**
    W = 512 #@param
    H = 512 #@param
    W, H = map(lambda x: x - x % 64, (W, H))  # resize to integer multiple of 64

    #@markdown **Sampling Settings**
    seed = -1 #@param
    sampler = 'klms' #@param ["klms","dpm2","dpm2_ancestral","heun","euler","euler_ancestral","plms", "ddim"]
    steps = 50 #@param
    scale = 7 #@param
    ddim_eta = 0.0 #@param
    dynamic_threshold = None
    static_threshold = None   

    #@markdown **Save & Display Settings**
    save_samples = True #@param {type:"boolean"}
    save_settings = True #@param {type:"boolean"}
    display_samples = True #@param {type:"boolean"}
    save_sample_per_step = False #@param {type:"boolean"}
    show_sample_per_step = False #@param {type:"boolean"}

    #@markdown **Prompt Settings**
    prompt_weighting = False #@param {type:"boolean"}
    normalize_prompt_weights = True #@param {type:"boolean"}
    log_weighted_subprompts = False #@param {type:"boolean"}

    #@markdown **Batch Settings**
    n_batch = 1 #@param
    batch_name = "Deforum" #@param {type:"string"}
    filename_format = "{timestring}_{index}_{prompt}.png" #@param ["{timestring}_{index}_{seed}.png","{timestring}_{index}_{prompt}.png"]
    seed_behavior = "iter" #@param ["iter","fixed","random"]
    make_grid = False #@param {type:"boolean"}
    grid_rows = 2 #@param 
    outdir = ""#get_output_folder(output_path, batch_name)

    #@markdown **Init Settings**
    use_init = False #@param {type:"boolean"}
    strength = 0.0 #@param {type:"number"}
    strength_0_no_init = True # Set the strength to 0 automatically when no init image is used
    init_image = "https://user-images.githubusercontent.com/14872007/195867706-d067cdc6-28cd-450b-a61e-55e25bc67010.png" #@param {type:"string"}
    # Whiter areas of the mask are areas that change more
    use_mask = False #@param {type:"boolean"}
    use_alpha_as_mask = False # use the alpha channel of the init image as the mask
    mask_file = "https://www.filterforge.com/wiki/images/archive/b/b7/20080927223728%21Polygonal_gradient_thumb.jpg" #@param {type:"string"}
    invert_mask = False #@param {type:"boolean"}
    # Adjust mask image, 1.0 is no adjustment. Should be positive numbers.
    mask_brightness_adjust = 1.0  #@param {type:"number"}
    mask_contrast_adjust = 1.0  #@param {type:"number"}
    # Overlay the masked image at the end of the generation so it does not get degraded by encoding and decoding
    overlay_mask = True  # {type:"boolean"}
    # Blur edges of final overlay mask, if used. Minimum = 0 (no blur)
    mask_overlay_blur = 5 # {type:"number"}

    n_samples = 1 # doesnt do anything
    precision = 'autocast' 
    C = 4
    f = 8

    prompt = ""
    timestring = ""
    init_latent = None
    init_sample = None
    init_c = None

    return locals()
    
def DeforumOutputArgs():
    skip_video_for_run_all = True #@param {type: 'boolean'}
    fps = 12 #@param {type:"number"}
    #@markdown **Manual Settings**
    use_manual_settings = False #@param {type:"boolean"}
    image_path = "/content/drive/MyDrive/AI/StableDiffusion/2022-09/20220903000939_%05d.png" #@param {type:"string"}
    mp4_path = "/content/drive/MyDrive/AI/StableDiffusion/content/drive/MyDrive/AI/StableDiffusion/2022-09/sion/2022-09/20220903000939.mp4" #@param {type:"string"}
    render_steps = False  #@param {type: 'boolean'}
    path_name_modifier = "x0_pred" #@param ["x0_pred","x"]
    max_video_frames = 200 #@param {type:"string"}
    return locals()
    
import gradio as gr
import os
import time
from types import SimpleNamespace

def setup_deforum_setting_ui(is_img2img):
    d = SimpleNamespace(**DeforumArgs()) #default args
    da = SimpleNamespace(**DeforumAnimArgs()) #default anim args
    dv = SimpleNamespace(**DeforumOutputArgs()) #default video args
    i1 = gr.HTML("<p style=\"font-weight:bold;margin-bottom:0.75em\">Deforum v0.5-webui-beta</p>")
    i2 = gr.HTML("<p style=\"margin-bottom:0.75em\">Made by deforum.github.io</p>")
    i3 = gr.HTML("<p style=\"margin-bottom:0.75em\">Original Deforum Github repo  github.com/deforum/stable-diffusion</p>")
    i4 = gr.HTML("<p style=\"margin-bottom:0.75em\">This WIP fork for auto1111's webui github.com/kabachuha/stable-diffusion/tree/automatic1111-webui</p>")
    i5 = gr.HTML("<p style=\"margin-bottom:0.75em\">Join the official Deforum Discord discord.gg/deforum to share your creations and suggestions</p>")
    i6 = gr.HTML("<p style=\"margin-bottom:0.75em\">User guide for v0.5 docs.google.com/document/d/1pEobUknMFMkn8F5TMsv8qRzamXX_75BShMMXV8IFslI/edit</p>")
    i7 = gr.HTML("<p style=\"margin-bottom:0.75em\">Math keyframing explanation docs.google.com/document/d/1pfW1PwbDIuW0cv-dnuyYj1UzPqe23BlSLTJsqazffXM/edit?usp=sharing</p>")
    
    
    i8 = gr.HTML("<p style=\"font-weight:bold;margin-bottom:0.75em\">Import settings from file</p>")
    with gr.Row():
        override_settings_with_file = gr.Checkbox(label="Override settings", value=False, interactive=True)
        custom_settings_file = gr.Textbox(label="Custom settings file", lines=1, interactive=True)
        #TODO make a button
        
    # Animation settings START
    #TODO make a some sort of the original dictionary parsing
    i9 = gr.HTML("<p style=\"font-weight:bold;margin-bottom:0.75em\">Animation settings</p>")
    with gr.Row():
        animation_mode = gr.Dropdown(label="animation_mode", choices=['2D', '3D', 'Video Input'], value=da.animation_mode, type="value", elem_id="animation_mode", interactive=True)
        max_frames = gr.Number(label="max_frames", value=da.max_frames, interactive=True, precision=0)
        border = gr.Dropdown(label="border", choices=['replicate', 'wrap'], value=da.border, type="value", elem_id="border", interactive=True)
    
    
    i10 = gr.HTML("<p style=\"margin-bottom:0.75em\">Motion parameters:</p>")
    i11 = gr.HTML("<p style=\"margin-bottom:0.75em\">2D and 3D settings</p>")
    with gr.Row():
        angle = gr.Textbox(label="angle", lines=1, value = da.angle, interactive=True)
    with gr.Row():
        zoom = gr.Textbox(label="zoom", lines=1, value = da.zoom, interactive=True)
    with gr.Row():
        translation_x = gr.Textbox(label="translation_x", lines=1, value = da.translation_x, interactive=True)
    with gr.Row():
        translation_y = gr.Textbox(label="translation_y", lines=1, value = da.translation_y, interactive=True)
    i33 = gr.HTML("<p style=\"margin-bottom:0.75em\">3D settings</p>")
    with gr.Row():
        translation_z = gr.Textbox(label="translation_z", lines=1, value = da.translation_z, interactive=True)
    with gr.Row():
        rotation_3d_x = gr.Textbox(label="rotation_3d_x", lines=1, value = da.rotation_3d_x, interactive=True)
    with gr.Row():
        rotation_3d_y = gr.Textbox(label="rotation_3d_y", lines=1, value = da.rotation_3d_y, interactive=True)
    with gr.Row():
        rotation_3d_z = gr.Textbox(label="rotation_3d_z", lines=1, value = da.rotation_3d_z, interactive=True)
    i12 = gr.HTML("<p style=\"margin-bottom:0.75em\">Prespective flip — Low VRAM pseudo-3D mode:</p>")
    with gr.Row():
        flip_2d_perspective = gr.Checkbox(label="flip_2d_perspective", value=da.flip_2d_perspective, interactive=True)
    with gr.Row():
        perspective_flip_theta = gr.Textbox(label="perspective_flip_theta", lines=1, value = da.perspective_flip_theta, interactive=True)
    with gr.Row():
        perspective_flip_phi = gr.Textbox(label="perspective_flip_phi", lines=1, value = da.perspective_flip_phi, interactive=True)
    with gr.Row():
        perspective_flip_gamma = gr.Textbox(label="perspective_flip_gamma", lines=1, value = da.perspective_flip_gamma, interactive=True)
    with gr.Row():
        perspective_flip_fv = gr.Textbox(label="perspective_flip_fv", lines=1, value = da.perspective_flip_fv, interactive=True)
    i34 = gr.HTML("<p style=\"margin-bottom:0.75em\">Generation settings:</p>")
    with gr.Row():
        noise_schedule = gr.Textbox(label="noise_schedule", lines=1, value = da.noise_schedule, interactive=True)
    with gr.Row():
        strength_schedule = gr.Textbox(label="strength_schedule", lines=1, value = da.strength_schedule, interactive=True)
    with gr.Row():
        contrast_schedule = gr.Textbox(label="contrast_schedule", lines=1, value = da.contrast_schedule, interactive=True)
#TODO
#        with gr.Row():
#            seed_schedule = gr.Textbox(label="seed_schedule", lines=1, value = da.seed_schedule, interactive=True)
#        with gr.Row():
#            scale_schedule = gr.Textbox(label="scale_schedule", lines=1, value = da.scale_schedule, interactive=True)
    
    i13 = gr.HTML("<p style=\"margin-bottom:0.75em\">Coherence:</p>")
    with gr.Row():
        color_coherence = gr.Dropdown(label="color_coherence", choices=['None', 'Match Frame 0 HSV', 'Match Frame 0 LAB', 'Match Frame 0 RGB'], value=da.color_coherence, type="value", elem_id="color_coherence", interactive=True)
        diffusion_cadence = gr.Slider(label="diffusion_cadence", minimum=1, maximum=8, step=1, value=1, interactive=True)
        
    i14 = gr.HTML("<p style=\"margin-bottom:0.75em\">3D Depth Warping:</p>")
    with gr.Row():
        use_depth_warping = gr.Checkbox(label="use_depth_warping", value=da.use_depth_warping, interactive=True)
    with gr.Row():
        midas_weight = gr.Number(label="midas_weight", value=da.midas_weight, interactive=True)
        near_plane = gr.Number(label="near_plane", value=da.near_plane, interactive=True)
        far_plane = gr.Number(label="far_plane", value=da.far_plane, interactive=True)
        fov = gr.Number(label="fov", value=da.fov, interactive=True)
        padding_mode = gr.Dropdown(label="padding_mode", choices=['border', 'reflection', 'zeros'], value=da.padding_mode, type="value", elem_id="padding_mode", interactive=True)
        sampling_mode = gr.Dropdown(label="sampling_mode", choices=['bicubic', 'bilinear', 'nearest'], value=da.sampling_mode, type="value", elem_id="sampling_mode", interactive=True)
        save_depth_maps = gr.Checkbox(label="save_depth_maps", value=da.save_depth_maps, interactive=True)
    
    i15 = gr.HTML("<p style=\"margin-bottom:0.75em\">Video Input:</p>")
    with gr.Row():
        video_init_path = gr.Textbox(label="video_init_path", lines=1, value = da.video_init_path, interactive=True)
    with gr.Row():
        extract_nth_frame = gr.Number(label="extract_nth_frame", value=da.extract_nth_frame, interactive=True, precision=0)
        overwrite_extracted_frames = gr.Checkbox(label="overwrite_extracted_frames", value=False, interactive=True)
        use_mask_video = gr.Checkbox(label="use_mask_video", value=False, interactive=True)
    with gr.Row():
        video_mask_path = gr.Textbox(label="video_mask_path", lines=1, value = da.video_mask_path, interactive=True)
    
    i16 = gr.HTML("<p style=\"margin-bottom:0.75em\">Interpolation:</p>")
    with gr.Row():
        interpolate_key_frames = gr.Checkbox(label="interpolate_key_frames", value=da.interpolate_key_frames, interactive=True)
        interpolate_x_frames = gr.Number(label="interpolate_x_frames", value=da.interpolate_x_frames, interactive=True, precision=0)
    
    i17 = gr.HTML("<p style=\"margin-bottom:0.75em\">Resume animation:</p>")
    with gr.Row():
        resume_from_timestring = gr.Checkbox(label="resume_from_timestring", value=da.resume_from_timestring, interactive=True)
        resume_timestring = gr.Textbox(label="resume_timestring", lines=1, value = da.resume_timestring, interactive=True)
    # Animation settings END
    
    # Prompts settings START
    
    i18 = gr.HTML("<p style=\"font-weight:bold;margin-bottom:0.75em\">Prompts</p>")
    i19 = gr.HTML("<p style=\"margin-bottom:0.75em\">`animation_mode: None` batches on list of *prompts*.</p>")
    i20 = gr.HTML("<p style=\"font-weight:bold;margin-bottom:0.75em\">*Important change from vanilla Deforum!*</p>")
    i21 = gr.HTML("<p style=\"font-weight:italic;margin-bottom:0.75em\">This script uses the built-in webui weighting settings.</p>")
    i22 = gr.HTML("<p style=\"font-weight:italic;margin-bottom:0.75em\">So if you want to use math functions as prompt weights,</p>")
    i23 = gr.HTML("<p style=\"font-weight:italic;margin-bottom:0.75em\">keep the values above zero in both parts</p>")
    i24 = gr.HTML("<p style=\"font-weight:italic;margin-bottom:0.75em\">Negative prompt part can be specified with --negative</p>")
    with gr.Row():
        prompts = gr.Textbox(label="prompts", lines=8, interactive=True, value = DeforumPrompts())
    with gr.Row():
        animation_prompts = gr.Textbox(label="animation_prompts", lines=8, interactive=True, value = DeforumAnimPrompts())
    
    # Prompts settings END
    
    i25 = gr.HTML("<p style=\"font-weight:bold;margin-bottom:0.75em\">Run settings</p>")
    
    # Sampling settings START
    i26 = gr.HTML("<p style=\"margin-bottom:0.75em\">Sampling settings</p>")
    i27 = gr.HTML("<p style=\"margin-bottom:0.75em\">The following settings have already been set up in the webui</p>")
    i28 = gr.HTML("<p style=\"margin-bottom:0.75em\">Do you want to override them with the values above?</p>")
    i29 = gr.HTML("<p style=\"font-weight:bold;margin-bottom:0.75em\">FIXME! Need to make deforum <-> webui samplers map. Before that, the sampler gets used from webui anyway</p>") #TODO
    with gr.Row():
        override_these_with_webui = gr.Checkbox(label="override_these_with_webui", value=False, interactive=True)
    i30 = gr.HTML("<p style=\"font-weight:bold;margin-bottom:0.75em\">W, H, seed, sampler, steps, scale, ddim_eta, n_batch, make_grid, grid_rows</p>")
    with gr.Row():
        W = gr.Slider(label="W", minimum=512, maximum=2048, step=64, value=d.W, interactive=True)
    with gr.Row():
        H = gr.Slider(label="H", minimum=512, maximum=2048, step=64, value=d.W, interactive=True)
    
    with gr.Row():
        seed = gr.Number(label="seed", value=d.seed, interactive=True, precision=0)
        sampler = gr.Dropdown(label="sampler", choices=["klms","dpm2","dpm2_ancestral","heun","euler","euler_ancestral","plms", "ddim"], value=d.sampler, type="value", elem_id="sampler", interactive=True)
    with gr.Row():
        steps = gr.Slider(label="steps", minimum=0, maximum=200, step=1, value=d.steps, interactive=True)
        scale = gr.Slider(label="scale", minimum=1, maximum=100, step=1, value=d.scale, interactive=True)
        ddim_eta = gr.Number(label="ddim_eta", value=d.ddim_eta, interactive=True)
        n_batch = gr.Number(label="n_batch", value=d.n_batch, interactive=True, precision=0)
        make_grid = gr.Checkbox(label="make_grid", value=d.make_grid, interactive=True)
        grid_rows = gr.Number(label="grid_rows", value=d.n_batch, interactive=True, precision=0)
        
    with gr.Row():
        save_settings = gr.Checkbox(label="save_settings", value=d.save_settings, interactive=True)
    with gr.Row():
        save_samples = gr.Checkbox(label="save_samples", value=d.save_samples, interactive=True)
        display_samples = gr.Checkbox(label="display_samples", value=d.display_samples, interactive=True)
    with gr.Row():
        save_sample_per_step = gr.Checkbox(label="save_sample_per_step", value=d.save_sample_per_step, interactive=True)
        show_sample_per_step = gr.Checkbox(label="show_sample_per_step", value=d.show_sample_per_step, interactive=True)
    
    # Sampling settings END
    
    # Batch settings START
    i31 = gr.HTML("<p style=\"margin-bottom:0.75em\">Batch settings</p>")
    with gr.Row():
        batch_name = gr.Textbox(label="batch_name", lines=1, interactive=True, value = d.batch_name)
    with gr.Row():    
        filename_format = gr.Textbox(label="filename_format", lines=1, interactive=True, value = d.filename_format)
    with gr.Row():
        seed_behavior = gr.Dropdown(label="seed_behavior", choices=['iter', 'fixed', 'random'], value=d.seed_behavior, type="value", elem_id="seed_behavior", interactive=True)
    # output - made in run
    # Batch settings END
    
    # Init settings START
    i32 = gr.HTML("<p style=\"margin-bottom:0.75em\">Init settings</p>")
    with gr.Row():
        use_init = gr.Checkbox(label="use_init", value=d.use_init, interactive=True, visible=True)
        from_img2img_instead_of_link = gr.Checkbox(label="from_img2img_instead_of_link", value=is_img2img, interactive=True, visible=is_img2img)
    with gr.Row():
        strength_0_no_init = gr.Checkbox(label="strength_0_no_init", value=True, interactive=True)
        strength = gr.Slider(label="strength", minimum=0, maximum=1, step=0.02, value=0, interactive=True)
    with gr.Row():
        init_image = gr.Textbox(label="init_image", lines=1, interactive=True, value = d.init_image)
    with gr.Row():
        use_mask = gr.Checkbox(label="use_mask", value=d.use_mask, interactive=True)
        use_alpha_as_mask = gr.Checkbox(label="use_alpha_as_mask", value=d.use_alpha_as_mask, interactive=True)
        invert_mask = gr.Checkbox(label="invert_mask", value=d.invert_mask, interactive=True)
        overlay_mask = gr.Checkbox(label="overlay_mask", value=d.overlay_mask, interactive=True)
    with gr.Row():
        mask_file = gr.Textbox(label="mask_file", lines=1, interactive=True, value = d.mask_file)
    with gr.Row():
        mask_brightness_adjust = gr.Number(label="mask_brightness_adjust", value=d.mask_brightness_adjust, interactive=True)
        mask_overlay_blur = gr.Number(label="mask_overlay_blur", value=d.mask_overlay_blur, interactive=True)
    # Init settings END
    
    # Video output settings START
    
    with gr.Row():
        skip_video_for_run_all = gr.Checkbox(label="skip_video_for_run_all", value=dv.skip_video_for_run_all, interactive=True)
        fps = gr.Number(label="fps", value=dv.fps, interactive=True)
    with gr.Row():
        use_manual_settings = gr.Checkbox(label="use_manual_settings", value=dv.use_manual_settings, interactive=True)
        render_steps = gr.Checkbox(label="render_steps", value=dv.render_steps, interactive=True)
    with gr.Row():
        max_video_frames = gr.Number(label="max_video_frames", value=200, interactive=True)
        path_name_modifier = gr.Dropdown(label="path_name_modifier", choices=['x0_pred', 'x'], value=dv.path_name_modifier, type="value", elem_id="path_name_modifier", interactive=True)
        
    with gr.Row():
        image_path = gr.Textbox(label="image_path", lines=1, interactive=True, value = dv.image_path)
    with gr.Row():
        mp4_path = gr.Textbox(label="mp4_path", lines=1, interactive=True, value = dv.mp4_path)
    # Video output settings END
    
    return [override_settings_with_file, custom_settings_file, animation_mode, max_frames, border, angle, zoom, translation_x, translation_y, translation_z, rotation_3d_x, rotation_3d_y, rotation_3d_z, flip_2d_perspective, perspective_flip_theta, perspective_flip_phi, perspective_flip_gamma, perspective_flip_fv, noise_schedule, strength_schedule, contrast_schedule, color_coherence, diffusion_cadence, use_depth_warping, midas_weight, near_plane, far_plane, fov, padding_mode, sampling_mode, save_depth_maps, video_init_path, extract_nth_frame, overwrite_extracted_frames, use_mask_video, video_mask_path, interpolate_key_frames, interpolate_x_frames, resume_from_timestring, resume_timestring, prompts, animation_prompts, W, H, seed, sampler, steps, scale, ddim_eta, n_batch, make_grid, grid_rows, save_settings, save_samples, display_samples, save_sample_per_step, show_sample_per_step, override_these_with_webui, batch_name, filename_format, seed_behavior, use_init, from_img2img_instead_of_link, strength_0_no_init, strength, init_image, use_mask, use_alpha_as_mask, invert_mask, overlay_mask, mask_file, mask_brightness_adjust, mask_overlay_blur, skip_video_for_run_all, fps, use_manual_settings, render_steps, max_video_frames, path_name_modifier, image_path, mp4_path, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, i16, i17, i18, i19, i20, i21, i22, i23, i24, i25, i26, i27, i28, i29, i30, i31, i32, i33, i34]

def pack_anim_args(animation_mode, max_frames, border, angle, zoom, translation_x, translation_y, translation_z, rotation_3d_x, rotation_3d_y, rotation_3d_z, flip_2d_perspective, perspective_flip_theta, perspective_flip_phi, perspective_flip_gamma, perspective_flip_fv, noise_schedule, strength_schedule, contrast_schedule, color_coherence, diffusion_cadence, use_depth_warping, midas_weight, near_plane, far_plane, fov, padding_mode, sampling_mode, save_depth_maps, video_init_path, extract_nth_frame, overwrite_extracted_frames, use_mask_video, video_mask_path, interpolate_key_frames, interpolate_x_frames, resume_from_timestring, resume_timestring):
    return locals()

def pack_args(W, H, seed, sampler, steps, scale, ddim_eta, n_batch, make_grid, grid_rows, save_settings, save_samples, display_samples, save_sample_per_step, show_sample_per_step, batch_name, filename_format, seed_behavior, use_init, from_img2img_instead_of_link, strength_0_no_init, strength, init_image, use_mask, use_alpha_as_mask, invert_mask, overlay_mask, mask_file, mask_brightness_adjust, mask_overlay_blur):
    precision = 'autocast' 
    C = 4
    f = 8

    prompt = ""
    timestring = ""
    init_latent = None
    init_sample = None
    init_c = None
    return locals()
    
def pack_video_args(skip_video_for_run_all, fps, use_manual_settings, render_steps, max_video_frames, path_name_modifier, image_path, mp4_path):
    return locals()
    
def process_args(self, p, override_settings_with_file, custom_settings_file, animation_mode, max_frames, border, angle, zoom, translation_x, translation_y, translation_z, rotation_3d_x, rotation_3d_y, rotation_3d_z, flip_2d_perspective, perspective_flip_theta, perspective_flip_phi, perspective_flip_gamma, perspective_flip_fv, noise_schedule, strength_schedule, contrast_schedule, color_coherence, diffusion_cadence, use_depth_warping, midas_weight, near_plane, far_plane, fov, padding_mode, sampling_mode, save_depth_maps, video_init_path, extract_nth_frame, overwrite_extracted_frames, use_mask_video, video_mask_path, interpolate_key_frames, interpolate_x_frames, resume_from_timestring, resume_timestring, prompts, animation_prompts, W, H, seed, sampler, steps, scale, ddim_eta, n_batch, make_grid, grid_rows, save_settings, save_samples, display_samples, save_sample_per_step, show_sample_per_step, override_these_with_webui, batch_name, filename_format, seed_behavior, use_init, from_img2img_instead_of_link, strength_0_no_init, strength, init_image, use_mask, use_alpha_as_mask, invert_mask, overlay_mask, mask_file, mask_brightness_adjust, mask_overlay_blur, skip_video_for_run_all, fps, use_manual_settings, render_steps, max_video_frames, path_name_modifier, image_path, mp4_path, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, i16, i17, i18, i19, i20, i21, i22, i23, i24, i25, i26, i27, i28, i29, i30, i31, i32, i33, i34):

    args_dict = pack_args(W, H, seed, sampler, steps, scale, ddim_eta, n_batch, make_grid, grid_rows, save_settings, save_samples, display_samples, save_sample_per_step, show_sample_per_step, batch_name, filename_format, seed_behavior, use_init, from_img2img_instead_of_link, strength_0_no_init, strength, init_image, use_mask, use_alpha_as_mask, invert_mask, overlay_mask, mask_file, mask_brightness_adjust, mask_overlay_blur)
    anim_args_dict = pack_anim_args(animation_mode, max_frames, border, angle, zoom, translation_x, translation_y, translation_z, rotation_3d_x, rotation_3d_y, rotation_3d_z, flip_2d_perspective, perspective_flip_theta, perspective_flip_phi, perspective_flip_gamma, perspective_flip_fv, noise_schedule, strength_schedule, contrast_schedule, color_coherence, diffusion_cadence, use_depth_warping, midas_weight, near_plane, far_plane, fov, padding_mode, sampling_mode, save_depth_maps, video_init_path, extract_nth_frame, overwrite_extracted_frames, use_mask_video, video_mask_path, interpolate_key_frames, interpolate_x_frames, resume_from_timestring, resume_timestring)
    video_args_dict = pack_video_args(skip_video_for_run_all, fps, use_manual_settings, render_steps, max_video_frames, path_name_modifier, image_path, mp4_path)
    
    import json
    
    animation_prompts_json = json.loads(animation_prompts)
    
    from scripts.deforum.settings import load_args
    
    if override_settings_with_file:
        load_args(args_dict,anim_args_dict,custom_settings_file, animation_prompts_json)

    root = SimpleNamespace(**Root())
    root.p = p
    root.prompts = json.loads(prompts)
    root.animation_prompts = animation_prompts_json
    
    print(f"Additional models path: {root.models_path}")
    if not os.path.exists(root.models_path):
        os.mkdir(root.models_path)

    args = SimpleNamespace(**args_dict)
    anim_args = SimpleNamespace(**anim_args_dict)
    video_args = SimpleNamespace(**video_args_dict)

    # TODO handle webui sampler settings override
    
    if override_these_with_webui:
        args.n_batch = p.batch_size
        args.W = p.width
        args.H = p.height
        args.W, args.H = map(lambda x: x - x % 64, (args.W, args.H))
        args.steps = p.steps
        args.seed = p.seed
    
    args.sampler = str(p.sampler_index)

    args.outdir = os.path.join(p.outpath_samples, batch_name)
    root.outpath_samples = args.outdir
    args.outdir = os.path.join(os.getcwd(), args.outdir)
    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)
        
    if args.seed == -1:
        import numpy as np
        args.seed = np.random.randint(0, 2**32 - 1)
        
    args.timestring = time.strftime('%Y%m%d%H%M%S')
    args.strength = max(0.0, min(1.0, args.strength))

    if not args.use_init:
        args.init_image = None
        
    if anim_args.animation_mode == 'None':
        anim_args.max_frames = 1
    elif anim_args.animation_mode == 'Video Input':
        args.use_init = True
    
    return root, args, anim_args, video_args
