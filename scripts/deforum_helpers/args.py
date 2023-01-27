from modules.shared import cmd_opts
from modules.processing import get_fixed_seed
import modules.shared as sh
import modules.paths as ph
import os
from pkg_resources import resource_filename

def Root():
    device = sh.device
    models_path = ph.models_path + '/Deforum'
    half_precision = not cmd_opts.no_half
    p = None
    frames_cache = []
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
    max_frames = 120 #@param {type:"number"}
    border = 'replicate' #@param ['wrap', 'replicate'] {type:'string'}

    #@markdown ####**Motion Parameters:**
    angle = "0:(0)"#@param {type:"string"}
    zoom = "0:(1.02+0.02*sin(2*3.14*t/20))"#@param {type:"string"}
    translation_x = "0:(0)"#@param {type:"string"}
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
    noise_schedule = "0: (0.08)"#@param {type:"string"}
    strength_schedule = "0: (0.6)"#@param {type:"string"}
    contrast_schedule = "0: (1.0)"#@param {type:"string"}
    cfg_scale_schedule = "0: (7)"
    enable_steps_scheduling = False#@param {type:"boolean"}
    steps_schedule = "0: (25)"#@param {type:"string"}
    fov_schedule = "0: (40)"
    near_schedule = "0: (200)"
    far_schedule = "0: (10000)"
    seed_schedule = "0: (t%4294967293)"
    
    # Sampler Scheduling
    enable_sampler_scheduling = False #@param {type:"boolean"}
    sampler_schedule = '0: ("Euler a")'

    # Checkpoint Scheduling
    enable_checkpoint_scheduling = False#@param {type:"boolean"}
    checkpoint_schedule = '0: ("model1.ckpt"), 100: ("model2.ckpt")'

    # Anti-blur
    kernel_schedule = "0: (5)"
    sigma_schedule = "0: (1.0)"
    amount_schedule = "0: (0.1)"
    threshold_schedule = "0: (0.0)"
    # Hybrid video
    hybrid_comp_alpha_schedule = "0:(1)" #@param {type:"string"}
    hybrid_comp_mask_blend_alpha_schedule = "0:(0.5)" #@param {type:"string"}
    hybrid_comp_mask_contrast_schedule = "0:(1)" #@param {type:"string"}
    hybrid_comp_mask_auto_contrast_cutoff_high_schedule =  "0:(100)" #@param {type:"string"}
    hybrid_comp_mask_auto_contrast_cutoff_low_schedule =  "0:(0)" #@param {type:"string"}

    #@markdown ####**Coherence:**
    histogram_matching = True #@param {type:"boolean"}
    color_coherence = 'Match Frame 0 LAB' #@param ['None', 'Match Frame 0 HSV', 'Match Frame 0 LAB', 'Match Frame 0 RGB', 'Video Input'] {type:'string'}
    color_coherence_video_every_N_frames = 1 #@param {type:"integer"}
    diffusion_cadence = '1' #@param ['1','2','3','4','5','6','7','8'] {type:'string'}

    #@markdown ####**Noise settings:**
    noise_type = 'perlin' #@param ['uniform', 'perlin'] {type:'string'}
    # Perlin params
    perlin_w = 8 #@param {type:"number"}
    perlin_h = 8 #@param {type:"number"}
    perlin_octaves = 4 #@param {type:"number"}
    perlin_persistence = 0.5 #@param {type:"number"}

    #@markdown ####**3D Depth Warping:**
    use_depth_warping = True #@param {type:"boolean"}
    midas_weight = 0.3#@param {type:"number"}
    near_plane = 200#deprecated see schedule
    far_plane = 10000#deprecated
    fov = 40#@param {type:"number"}#deprecated
    padding_mode = 'border'#@param ['border', 'reflection', 'zeros'] {type:'string'}
    sampling_mode = 'bicubic'#@param ['bicubic', 'bilinear', 'nearest'] {type:'string'}
    save_depth_maps = False #@param {type:"boolean"}

    #@markdown ####**Video Input:**
    video_init_path ='/content/video_in.mp4'#@param {type:"string"}
    extract_nth_frame = 1#@param {type:"number"}
    extract_from_frame = 0 #@param {type:"number"}
    extract_to_frame = -1 #@param {type:"number"} minus 1 for unlimited frames
    overwrite_extracted_frames = True #@param {type:"boolean"}
    use_mask_video = False #@param {type:"boolean"}
    video_mask_path ='/content/video_in.mp4'#@param {type:"string"}

    #@markdown ####**Hybrid Video for 2D/3D Animation Mode:**
    hybrid_generate_inputframes = False #@param {type:"boolean"}
    hybrid_generate_human_masks = "None" #@param ['None','PNGs','Video', 'Both']
    hybrid_use_first_frame_as_init_image = True #@param {type:"boolean"}
    hybrid_motion = "None" #@param ['None','Optical Flow','Perspective','Affine']
    hybrid_flow_method = "Farneback" #@param ['Farneback','DenseRLOF','SF']
    hybrid_composite = False #@param {type:"boolean"}
    hybrid_comp_mask_type = "None" #@param ['None', 'Depth', 'Video Depth', 'Blend', 'Difference']
    hybrid_comp_mask_inverse = False #@param {type:"boolean"}
    hybrid_comp_mask_equalize = "None" #@param  ['None','Before','After','Both']
    hybrid_comp_mask_auto_contrast = False #@param {type:"boolean"}
    hybrid_comp_save_extra_frames = False #@param {type:"boolean"}

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
    "0": "(scenic countryside:1.0), (cherry:`where(cos(6.28*t/10)>0, 1.8*cos(6.28*t/10), 0.001)`), (strawberry:`where(cos(6.28*t/10)<0, -1.8*cos(6.28*t/10), 0.001)`), snow, detailed painting by greg rutkowski --neg (cherry:`where(cos(6.28*t/10)<0, -1.8*cos(6.28*t/10), 0.001)`), (strawberry:`where(cos(6.28*t/10)>0, 1.8*cos(6.28*t/10), 0.001)`)",
    "60": "a beautiful (((banana))), trending on Artstation",
    "80": "a beautiful coconut --neg photo, realistic",
    "100": "a beautiful durian, trending on Artstation"
}
"""

def DeforumArgs():
    #@markdown **Image Settings**
    W = 512 #@param
    H = 512 #@param
    W, H = map(lambda x: x - x % 64, (W, H))  # resize to integer multiple of 64

    #@markdonw **Webui stuff**
    restore_faces = False
    tiling = False
    enable_hr = False
    firstphase_width = 0
    firstphase_height = 0
    seed_enable_extras = False
    subseed = -1
    subseed_strength = 0
    seed_resize_from_w = 0
    seed_resize_from_h = 0
    
    #@markdown **Sampling Settings**
    seed = -1 #@param
    sampler = 'klms' #@param ["klms","dpm2","dpm2_ancestral","heun","euler","euler_ancestral","plms", "ddim"]
    steps = 21 #@param
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
    seed_behavior = "iter" #@param ["iter","fixed","random","ladder","alternate","schedule"]
    seed_iter_N = 1 #@param {type:'integer'}
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
    mask_contrast_adjust = 1.0  #@param {type:"number"}
    mask_brightness_adjust = 1.0  #@param {type:"number"}
    # Overlay the masked image at the end of the generation so it does not get degraded by encoding and decoding
    overlay_mask = True  # {type:"boolean"}
    # Blur edges of final overlay mask, if used. Minimum = 0 (no blur)
    mask_overlay_blur = 5 # {type:"number"}

    fill = 1 #MASKARGSEXPANSION Todo : Rename and convert to same formatting as used in img2img masked content
    full_res_mask = True
    full_res_mask_padding = 4
    reroll_blank_frames = 'reroll' # reroll, interrupt, or ignore

    n_samples = 1 # doesnt do anything
    precision = 'autocast' 
    C = 4
    f = 8

    prompt = ""
    timestring = ""
    init_latent = None
    init_sample = None
    init_c = None
    noise_mask = None
    seed_internal = 0

    return locals()

def ParseqArgs():
    parseq_manifest = None
    parseq_use_deltas = True
    return locals()
    
def DeforumOutputArgs():
    skip_video_for_run_all = False #@param {type: 'boolean'}
    fps = 12 #@param {type:"number"}
    #@markdown **Manual Settings**
    use_manual_settings = False #@param {type:"boolean"}
    image_path = "/content/drive/MyDrive/AI/StableDiffusion/2022-09/20220903000939_%05d.png" #@param {type:"string"}
    mp4_path = "/content/drive/MyDrive/AI/StableDiffusion/content/drive/MyDrive/AI/StableDiffusion/2022-09/kabachuha/2022-09/20220903000939.mp4" #@param {type:"string"}
    ffmpeg_location = find_ffmpeg_binary()
    ffmpeg_crf = '17'
    ffmpeg_preset = 'veryslow'
    add_soundtrack = 'None' #@param ["File","Init Video"]
    soundtrack_path = "snowfall.mp3"
    render_steps = False  #@param {type: 'boolean'}
    path_name_modifier = "x0_pred" #@param ["x0_pred","x"]
    max_video_frames = 200 #@param {type:"string"}
    store_frames_in_ram = False #@param {type: 'boolean'}
    frame_interpolation_engine = "RIFE v4.6" #@param ["RIFE v4.0","RIFE v4.3","RIFE v4.6"]
    frame_interpolation_x_amount = "Disabled" #@param ["Disabled" + all values from x2 to x10]
    frame_interpolation_slow_mo_amount = "Disabled" #@param ["Disabled","x2","x4","x8"]
    frame_interpolation_keep_imgs = False #@param {type: 'boolean'}
    return locals()
    
import gradio as gr
import os
import time
from types import SimpleNamespace

i1_store_backup = "<p style=\"font-weight:bold;margin-bottom:0.75em\">Deforum v0.5-webui-beta</p>"
i1_store = i1_store_backup

mask_fill_choices=['fill', 'original', 'latent noise', 'latent nothing']

def setup_deforum_setting_dictionary(self, is_img2img, is_extension = True):
    d = SimpleNamespace(**DeforumArgs()) #default args
    da = SimpleNamespace(**DeforumAnimArgs()) #default anim args
    dp = SimpleNamespace(**ParseqArgs()) #default parseq ars
    dv = SimpleNamespace(**DeforumOutputArgs()) #default video args
    if not is_extension:
        with gr.Row():
            btn = gr.Button("Click here after the generation to show the video")
        with gr.Row():
            i1 = gr.HTML(i1_store, elem_id='deforum_header')
    else:
        btn = i1 = gr.HTML("")
    
    with gr.Accordion("Info and links", open=False):
        i2 = gr.HTML("""<strong>Made by <a href="https://deforum.github.io">deforum.github.io</a>, port for AUTOMATIC1111's webui maintained by <a href="https://github.com/kabachuha">kabachuha</a></strong>""")
        i3 = gr.HTML("""<ul style="list-style-type:circle; margin-left:1em">
        <li>The code for this extension: <a  style="color:blue" href="https://github.com/deforum-art/deforum-for-automatic1111-webui">Fork of deforum for auto1111's webui</a>.</li>
        <li>Join the <a style="color:blue" href="https://discord.gg/deforum">official Deforum Discord</a> to share your creations and suggestions.</li>
        <li>For general usage, see the <a style="color:blue" href="https://docs.google.com/document/d/1pEobUknMFMkn8F5TMsv8qRzamXX_75BShMMXV8IFslI/edit">User guide for Deforum v0.5</a>.</li>
        <li>For advanced animations, see the <a style="color:blue" href="https://docs.google.com/document/d/1pfW1PwbDIuW0cv-dnuyYj1UzPqe23BlSLTJsqazffXM/edit?usp=sharing">Math keyframing explanation</a>.</li>
        <li>Alternatively, use <a style="color:blue" href="https://sd-parseq.web.app/deforum">sd-parseq</a> as a UI to define your animation schedules (see the Parseq section in the Keyframes tab).</li>
        <li><a style="color:blue" href="https://www.framesync.xyz/">framesync.xyz</a> is also a good option, it makes compact math formulae for Deforum keyframes by selecting various waveforms.</li>
        <li>The other site allows for making keyframes using <a style="color:blue" href="https://www.chigozie.co.uk/keyframe-string-generator/">interactive splines and Bezier curves</a> (select Disco output format).</li>
        <li>After the 2022-12-30 update, the default noise type is <a style="color:blue" href="https://en.wikipedia.org/wiki/Perlin_noise">Perlin noise</a> due to its great frame coherence improvements. If you want to use the old noise and replicate the previous settings, set the type to "uniform" in the Keyframes tab.</li>
        </ul>
        <italic>If you liked this extension, please <a style="color:blue" href="https://github.com/deforum-art/deforum-for-automatic1111-webui">give it a star on GitHub</a>!</italic> 😊""")
        i4 = gr.HTML("")
        i5 = gr.HTML("")
        i6 = gr.HTML("")
        i7 = gr.HTML("")
    
    if not is_extension:
        def show_vid():
            return {
                i1: gr.update(value=i1_store, visible=True)
            }
        
        btn.click(
            show_vid,
            [],
            [i1]
            )
    
    with gr.Tab('Run'):
        i25 = gr.HTML("<p style=\"font-weight:bold;margin-bottom:0.75em\">Run settings</p>")
        
        i8 = gr.HTML("<p style=\"font-weight:bold;margin-bottom:0.75em\">Import settings from file</p>")
        with gr.Row():
            override_settings_with_file = gr.Checkbox(label="Override settings", value=False, interactive=True)
            custom_settings_file = gr.Textbox(label="Custom settings file", lines=1, interactive=True)
        
        # Sampling settings START
        i26 = gr.HTML("<p style=\"margin-bottom:0.75em\">Sampling settings</p>")
        i27 = gr.HTML("")
        i28 = gr.HTML("")#TODO cleanup
        i29 = gr.HTML("")
        #with gr.Row():
        override_these_with_webui = gr.Checkbox(label="override_these_with_webui", value=False, interactive=True)
        i30 = gr.HTML("")
        with gr.Row():
            W = gr.Slider(label="W", minimum=64, maximum=2048, step=64, value=d.W, interactive=True)
        with gr.Row():
            H = gr.Slider(label="H", minimum=64, maximum=2048, step=64, value=d.W, interactive=True)
        with gr.Row():
            restore_faces = gr.Checkbox(label='Restore faces', value=False, visible=len(sh.face_restorers) > 1)
            tiling = gr.Checkbox(label='Tiling', value=False)
            enable_hr = gr.Checkbox(label='Highres. fix', value=False)
        with gr.Row(visible=False) as hr_options:
            firstphase_width = gr.Slider(minimum=0, maximum=1024, step=64, label="Firstpass width", value=0)
            firstphase_height = gr.Slider(minimum=0, maximum=1024, step=64, label="Firstpass height", value=0)
        with gr.Row():
            seed = gr.Number(label="seed", value=d.seed, interactive=True, precision=0)
            from modules.sd_samplers import samplers_for_img2img
            sampler = gr.Dropdown(label="sampler", choices=[x.name for x in samplers_for_img2img], value=samplers_for_img2img[0].name, type="value", elem_id="sampler", interactive=True)
        with gr.Row():
            seed_enable_extras = gr.Checkbox(label="Enable extras", value=False)
            subseed = gr.Number(label="subseed", value=d.subseed, interactive=True, precision=0)
            subseed_strength = gr.Slider(label="subseed_strength", minimum=0, maximum=1, step=0.01, value=d.subseed_strength, interactive=True)
        with gr.Row():
            seed_resize_from_w = gr.Slider(minimum=0, maximum=2048, step=64, label="Resize seed from width", value=0)
            seed_resize_from_h = gr.Slider(minimum=0, maximum=2048, step=64, label="Resize seed from height", value=0)
        with gr.Row():
            steps = gr.Slider(label="steps", minimum=0, maximum=200, step=1, value=d.steps, interactive=True)
        with gr.Row():
            ddim_eta = gr.Number(label="ddim_eta", value=d.ddim_eta, interactive=True)
            n_batch = gr.Number(label="n_batch", value=d.n_batch, interactive=True, precision=0)
            make_grid = gr.Checkbox(label="make_grid", value=d.make_grid, interactive=True)
            grid_rows = gr.Number(label="grid_rows", value=d.n_batch, interactive=True, precision=0)
            
        with gr.Row():
            save_settings = gr.Checkbox(label="save_settings", value=d.save_settings, interactive=True)
        with gr.Row():
            save_samples = gr.Checkbox(label="save_samples", value=d.save_samples, interactive=True)
            display_samples = gr.Checkbox(label="display_samples", value=False, interactive=False)
        with gr.Row():
            save_sample_per_step = gr.Checkbox(label="save_sample_per_step", value=d.save_sample_per_step, interactive=True)
            show_sample_per_step = gr.Checkbox(label="show_sample_per_step", value=False, interactive=False)
        # Sampling settings END
        
        # Batch settings START
        i31 = gr.HTML("<p style=\"margin-bottom:0.75em\">Batch settings</p>")
        with gr.Row():
            batch_name = gr.Textbox(label="batch_name", lines=1, interactive=True, value = d.batch_name)
        with gr.Row():    
            filename_format = gr.Textbox(label="filename_format", lines=1, interactive=True, value = d.filename_format)
        with gr.Row():
            seed_behavior = gr.Dropdown(label="seed_behavior", choices=['iter', 'fixed', 'random', 'ladder', 'alternate', 'schedule'], value=d.seed_behavior, type="value", elem_id="seed_behavior", interactive=True)
            seed_iter_N = gr.Number(label="seed_iter_N", value=d.seed_iter_N, interactive=True, precision=0)
        # output - made in run
        # Batch settings END
            
    with gr.Tab('Keyframes'):
        # Animation settings START
        #TODO make a some sort of the original dictionary parsing
        i9 = gr.HTML("<p style=\"font-weight:bold;margin-bottom:0.75em\">Animation settings</p>")
        with gr.Row():
            animation_mode = gr.Dropdown(label="animation_mode", choices=['2D', '3D', 'Video Input', 'Interpolation'], value=da.animation_mode, type="value", elem_id="animation_mode", interactive=True)
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
        with gr.Row():
            cfg_scale_schedule = gr.Textbox(label="cfg_scale_schedule", lines=1, value = da.cfg_scale_schedule, interactive=True)
        # Sampler Scheduling
        i34 = gr.HTML("<p style=\"margin-bottom:0.75em\">Steps scheduling:</p>")
        with gr.Row():
            enable_steps_scheduling = gr.Checkbox(label="enable steps scheduling.", value=da.enable_steps_scheduling, interactive=True)
        with gr.Row():
            steps_schedule = gr.Textbox(label="steps_schedule", lines=1, value = da.steps_schedule, interactive=True)
        i27 = gr.HTML("<p style=\"margin-bottom:0.75em\">3D Fov settings:</p>")
        with gr.Row():
            fov_schedule = gr.Textbox(label="fov_schedule", lines=1, value = da.fov_schedule, interactive=True)
        with gr.Row():
            near_schedule = gr.Textbox(label="near_schedule", lines=1, value = da.near_schedule, interactive=True)
        with gr.Row():
            far_schedule = gr.Textbox(label="far_schedule", lines=1, value = da.far_schedule, interactive=True)
        i36 = gr.HTML("<p style=\"margin-bottom:0.75em\">To enable seed schedule select seed behavior — 'schedule'</p>")
        with gr.Row():
            seed_schedule = gr.Textbox(label="seed_schedule", lines=1, value = da.seed_schedule, interactive=True)

        # Sampler Scheduling
        i38 = gr.HTML("<p style=\"margin-bottom:0.75em\">Sampler scheduling:</p>")
        with gr.Row():
            enable_sampler_scheduling = gr.Checkbox(label="enable sampler scheduling.", value=da.enable_sampler_scheduling, interactive=True)
        with gr.Row():
            sampler_schedule = gr.Textbox(label="sampler_schedule", lines=1, value = da.sampler_schedule, interactive=True)

        # Checkpoint Scheduling
        i38 = gr.HTML("<p style=\"margin-bottom:0.75em\">Checkpoint scheduling:</p>")
        with gr.Row():
            enable_checkpoint_scheduling = gr.Checkbox(label="enable_checkpoint_scheduling", value=da.enable_checkpoint_scheduling, interactive=True)
        with gr.Row():
            checkpoint_schedule = gr.Textbox(label="checkpoint_schedule", lines=1, value = da.checkpoint_schedule, interactive=True)

        # Anti-blur
        i38 = gr.HTML("<p style=\"margin-bottom:0.75em\">Anti-blur settings</p>")
        with gr.Row():
            kernel_schedule = gr.Textbox(label="kernel_schedule", lines=1, value = da.kernel_schedule, interactive=True)
        with gr.Row():
            sigma_schedule = gr.Textbox(label="sigma_schedule", lines=1, value = da.sigma_schedule, interactive=True)
        with gr.Row():
            amount_schedule = gr.Textbox(label="amount_schedule", lines=1, value = da.amount_schedule, interactive=True)
        with gr.Row():
            threshold_schedule = gr.Textbox(label="threshold_schedule", lines=1, value = da.threshold_schedule, interactive=True)
        
        i13 = gr.HTML("<p style=\"margin-bottom:0.75em\">Coherence:</p>")
        with gr.Row():
            # what to do with blank frames (they may result from glitches or the NSFW filter being turned on): reroll with +1 seed, interrupt the animation generation, or do nothing
            reroll_blank_frames = gr.Dropdown(label="reroll_blank_frames", choices=['reroll', 'interrupt', 'ignore'], value=d.reroll_blank_frames, type="value", elem_id="reroll_blank_frames", interactive=True)
        with gr.Row():
            histogram_matching = gr.Checkbox(label="Force all frames to match initial frame's colors. Overrides a1111 settings. NOT RECOMMENDED, enable only for backwards compatibility.", value=da.histogram_matching, interactive=True)
        with gr.Row():
            color_coherence = gr.Dropdown(label="color_coherence", choices=['None', 'Match Frame 0 HSV', 'Match Frame 0 LAB', 'Match Frame 0 RGB', 'Video Input'], value=da.color_coherence, type="value", elem_id="color_coherence", interactive=True)
            color_coherence_video_every_N_frames = gr.Number(label="color_coherence_video_every_N_frames", value=1, interactive=True)
            diffusion_cadence = gr.Number(label="diffusion_cadence", value=1, interactive=True)
        
        i39 = gr.HTML("<p style=\"margin-bottom:0.75em\">Noise settings:</p>")
        with gr.Row():
            noise_type = gr.Dropdown(label="noise_type", choices=['uniform', 'perlin'], value=da.noise_type, type="value", elem_id="noise_type", interactive=True)
        i39 = gr.HTML("<p style=\"margin-bottom:0.75em\">Perlin noise params, if selected.</p>")
        with gr.Row():
            perlin_w = gr.Number(label="perlin_w", value=da.perlin_w, interactive=True)
            perlin_h = gr.Number(label="perlin_h", value=da.perlin_h, interactive=True)
        with gr.Row():
            perlin_octaves = gr.Slider(label="perlin_octaves", minimum=1, maximum=7, value=da.perlin_octaves, step=1, interactive=True)
            perlin_persistence = gr.Slider(label="perlin_persistence", minimum=0, maximum=1, value=da.perlin_persistence, step=0.02, interactive=True)
            
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

        with gr.Accordion('Parseq', open=False):
            i37 = gr.HTML("""
            Use an <a style='color:blue;' target='_blank' href='https://sd-parseq.web.app/deforum'>sd-parseq manifest</a> for your animation (leave blank to ignore).</p>
            <p style="margin-top:1em">
                Note that parseq overrides:
                <ul style="list-style-type:circle; margin-left:2em; margin-bottom:1em">
                    <li>Run: seed, subseed, subseed strength.</li>
                    <li>Keyframes: generation settings (noise, strength, contrast, scale).</li>
                    <li>Keyframes: motion parameters for 2D and 3D (angle, zoom, translation, rotation, perspective flip).</li>
                </ul>
            </p>
            <p">
                Parseq does <strong><em>not</em></strong> override:
                <ul style="list-style-type:circle; margin-left:2em; margin-bottom:1em">
                    <li>Run: Sampler, W, H, Restore faces, tiling, highres fix, resize seed.</li>
                    <li>Keyframes: animation settings (animation mode, max_frames, border) </li>
                    <li>Keyframes: coherence (color coherence & diffusion cadence) </li>
                    <li>Keyframes: depth warping</li>
                    <li>Video output settings: all settings (including fps and max frames)</li>
                </ul>
            </p>
            """)
            with gr.Row():
                parseq_manifest = gr.Textbox(label="Parseq Manifest (JSON or URL)", lines=4, value = dp.parseq_manifest, interactive=True)
            with gr.Row():
                parseq_use_deltas = gr.Checkbox(label="Use delta values for movement parameters", value=dp.parseq_use_deltas, interactive=True)            
    
    # Animation settings END
    
    # Prompts settings START    
    with gr.Tab('Prompts'):
        i18 = gr.HTML("<p style=\"font-weight:bold;margin-bottom:0.75em\">Prompts</p>")
        i19 = gr.HTML("<p>`animation_mode: None` batches on list of *prompts*. (Batch mode disabled atm, only animation_prompts are working)</p>")
        i20 = gr.HTML("<p style=\"font-weight:bold\">*Important change from vanilla Deforum!*</p>")
        i21 = gr.HTML("<p style=\"font-weight:italic\">This script uses the built-in webui weighting settings.</p>")
        i22 = gr.HTML("<p style=\"font-weight:italic\">So if you want to use math functions as prompt weights,</p>")
        i23 = gr.HTML("<p style=\"font-weight:italic\">keep the values above zero in both parts</p>")
        i24 = gr.HTML("<p style=\"font-weight:italic\">Negative prompt part can be specified with --neg</p>")
        with gr.Row():
            prompts = gr.Textbox(label="batch_prompts (disabled atm)", lines=8, interactive=False, value = DeforumPrompts(), visible = True) # TODO
        with gr.Row():
            animation_prompts = gr.Textbox(label="animation_prompts", lines=8, interactive=True, value = DeforumAnimPrompts())
    
    # Prompts settings END
    
    with gr.Tab('Init'):
        # Init settings START
        i32 = gr.HTML("<p style=\"margin-bottom:0.75em\">Init settings</p>")
        with gr.Row():
            use_init = gr.Checkbox(label="use_init", value=d.use_init, interactive=True, visible=True)
            from_img2img_instead_of_link = gr.Checkbox(label="from_img2img_instead_of_link", value=False, interactive=False, visible=True)
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
            mask_contrast_adjust = gr.Number(label="mask_contrast_adjust", value=d.mask_contrast_adjust, interactive=True)
            mask_brightness_adjust = gr.Number(label="mask_brightness_adjust", value=d.mask_brightness_adjust, interactive=True)
            mask_overlay_blur = gr.Number(label="mask_overlay_blur", value=d.mask_overlay_blur, interactive=True)
        with gr.Row():
            choice = mask_fill_choices[d.fill]
            fill = gr.Radio(label='mask_fill', choices=mask_fill_choices, value=choice, type="index")
        with gr.Row():
            #fill = gr.Slider(minimum=0, maximum=3, step=1, label="mask_fill_type", value=d.fill, interactive=True)
            full_res_mask = gr.Checkbox(label="full_res_mask", value=d.full_res_mask, interactive=True)
            full_res_mask_padding = gr.Slider(minimum=0, maximum=512, step=1, label="full_res_mask_padding", value=d.full_res_mask_padding, interactive=True)
        i15 = gr.HTML("<p style=\"margin-bottom:0.75em\">Video Input:</p>")
        with gr.Row():
            video_init_path = gr.Textbox(label="video_init_path", lines=1, value = da.video_init_path, interactive=True)
        with gr.Row():
            extract_nth_frame = gr.Number(label="extract_nth_frame", value=da.extract_nth_frame, interactive=True, precision=0)
            extract_from_frame = gr.Number(label="extract_from_frame", value=da.extract_from_frame, interactive=True, precision=0)
            extract_to_frame = gr.Number(label="extract_to_frame", value=da.extract_to_frame, interactive=True, precision=0)
            overwrite_extracted_frames = gr.Checkbox(label="overwrite_extracted_frames", value=False, interactive=True)
            use_mask_video = gr.Checkbox(label="use_mask_video", value=False, interactive=True)
        with gr.Row():
            video_mask_path = gr.Textbox(label="video_mask_path", lines=1, value = da.video_mask_path, interactive=True)
        
        i16 = gr.HTML("<p style=\"margin-bottom:0.75em\">Interpolation (turned off atm)</p>")
        with gr.Row():
            interpolate_key_frames = gr.Checkbox(label="interpolate_key_frames", value=da.interpolate_key_frames, interactive=False, visible = False)
            interpolate_x_frames = gr.Number(label="interpolate_x_frames", value=da.interpolate_x_frames, interactive=False, precision=0, visible = False)#TODO
        
        i17 = gr.HTML("<p style=\"margin-bottom:0.75em\">Resume animation:</p>")
        with gr.Row():
            resume_from_timestring = gr.Checkbox(label="resume_from_timestring", value=da.resume_from_timestring, interactive=True)
            resume_timestring = gr.Textbox(label="resume_timestring", lines=1, value = da.resume_timestring, interactive=True)
        # Init settings END

    with gr.Tab('Hybrid Video'):
        hybrid_html = "<p style=\"padding-bottom:0\"><b style=\"text-shadow: blue -1px -1px;\">Hybrid Video Compositing in 2D/3D Mode</b><span style=\"color:#DDD;font-size:0.7rem;text-shadow: black -1px -1px;margin-left:10px;\">by reallybigname</span></p>"
        i40 = gr.HTML(hybrid_html)
        with gr.Accordion("More Info", open=False):
            hybrid_html = "<ul style=\"list-style-type:circle; margin-left:1em; margin-bottom:1em;\"><li>Composite video with previous frame init image in <b>2D or 3D animation_mode</b> <i>(not for Video Input mode)</i></li>"
            hybrid_html += "<li>Uses your <b>Init</b> settings for <b>video_init_path, extract_nth_frame, overwrite_extracted_frames</b></li>"
            hybrid_html += "<li>In Keyframes tab, you can also set <b>color_coherence</b> = '<b>Video Input</b>'</li>"
            hybrid_html += "<li><b>color_coherence_video_every_N_frames</b> lets you only match every N frames</li>"
            hybrid_html += "<li>Color coherence may be used with hybrid composite off, to just use video color.</li>"
            hybrid_html += "<li>Hybrid motion may be used with hybrid composite off, to just use video motion.</li></ul>"
            hybrid_html += "Hybrid Video Schedules"
            hybrid_html += "<ul style=\"list-style-type:circle; margin-left:1em; margin-bottom:1em;\"><li>The alpha schedule controls overall alpha for video mix, whether using a composite mask or not.</li>"
            hybrid_html += "<li>The <b>hybrid_comp_mask_blend_alpha_schedule</b> only affects the 'Blend' <b>hybrid_comp_mask_type</b>.</li>"
            hybrid_html += "<li>Mask contrast schedule is from 0-255. Normal is 1. Affects all masks.</li>"
            hybrid_html += "<li>Autocontrast low/high cutoff schedules 0-100. Low 0 High 100 is full range. <br>(<i><b>hybrid_comp_mask_auto_contrast</b> must be enabled</i>)</li></ul>"            
            hybrid_html += "See main Deforum document for more details."            
            i41 = gr.HTML(hybrid_html)
        with gr.Row():
            with gr.Column(variant="compact"):
                hybrid_generate_inputframes = gr.Checkbox(label="hybrid_generate_inputframes", value=False, interactive=True)
            with gr.Column(variant="compact"):
                #hybrid_generate_human_masks = gr.Checkbox(label="hybrid_generate_human_masks", value=False, interactive=True)
                hybrid_generate_human_masks = gr.Dropdown(label="hybrid_generate_human_masks", choices=['None', 'PNGs', 'Video', 'Both'], value=da.hybrid_generate_human_masks, type="value", elem_id="hybrid_generate_human_masks", interactive=True)
            with gr.Column(variant="compact"):
                hybrid_use_first_frame_as_init_image = gr.Checkbox(label="hybrid_use_first_frame_as_init_image", value=False, interactive=True)
        with gr.Row():
            with gr.Column(variant="compact"):
                hybrid_motion = gr.Dropdown(label="hybrid_motion", choices=['None', 'Optical Flow', 'Perspective', 'Affine'], value=da.hybrid_motion, type="value", elem_id="hybrid_motion", interactive=True)
            with gr.Column(variant="compact"):
                hybrid_flow_method = gr.Dropdown(label="hybrid_flow_method", choices=['Farneback', 'DenseRLOF', 'SF'], value=da.hybrid_flow_method, type="value", elem_id="hybrid_flow_method", interactive=True)
        with gr.Row():
            hybrid_composite = gr.Checkbox(label="hybrid_composite", value=False, interactive=True)
            hybrid_comp_mask_type = gr.Dropdown(label="hybrid_comp_mask_type", choices=['None', 'Depth', 'Video Depth', 'Blend', 'Difference'], value=da.hybrid_comp_mask_type, type="value", elem_id="hybrid_comp_mask_type", interactive=True)
        with gr.Row():
            with gr.Column(variant="compact"):
                hybrid_comp_mask_auto_contrast = gr.Checkbox(label="hybrid_comp_mask_auto_contrast", value=False, interactive=True)
            with gr.Column(variant="compact"):
                hybrid_comp_mask_inverse = gr.Checkbox(label="hybrid_comp_mask_inverse", value=False, interactive=True)
        with gr.Row():
            hybrid_comp_mask_equalize = gr.Dropdown(label="hybrid_comp_mask_equalize", choices=['None', 'Before', 'After', 'Both'], value=da.hybrid_comp_mask_equalize, type="value", elem_id="hybrid_comp_mask_equalize", interactive=True)
            with gr.Column(variant="compact"):
                hybrid_comp_save_extra_frames = gr.Checkbox(label="hybrid_comp_save_extra_frames", value=False, interactive=True)

        hybrid_html = "<p style=\"margin-bottom:0.75em\"><b>Hybrid Video Schedules</b></p>"
        i42 = gr.HTML(hybrid_html)
        with gr.Row():
            hybrid_comp_alpha_schedule = gr.Textbox(label="hybrid_comp_alpha_schedule", lines=1, value = da.hybrid_comp_alpha_schedule, interactive=True)
        with gr.Row():
            hybrid_comp_mask_blend_alpha_schedule = gr.Textbox(label="hybrid_comp_mask_blend_alpha_schedule", lines=1, value = da.hybrid_comp_mask_blend_alpha_schedule, interactive=True)
        with gr.Row():
            hybrid_comp_mask_contrast_schedule = gr.Textbox(label="hybrid_comp_mask_contrast_schedule", lines=1, value = da.hybrid_comp_mask_contrast_schedule, interactive=True)
        with gr.Row():
            hybrid_comp_mask_auto_contrast_cutoff_high_schedule = gr.Textbox(label="hybrid_comp_mask_auto_contrast_cutoff_high_schedule", lines=1, value = da.hybrid_comp_mask_auto_contrast_cutoff_high_schedule, interactive=True)
        with gr.Row():
            hybrid_comp_mask_auto_contrast_cutoff_low_schedule = gr.Textbox(label="hybrid_comp_mask_auto_contrast_cutoff_low_schedule", lines=1, value = da.hybrid_comp_mask_auto_contrast_cutoff_low_schedule, interactive=True)

    with gr.Tab('Video output'):
        # Video output settings START
        i35 = gr.HTML("<p style=\"margin-bottom:0.75em\">Video output settings</p>")
        
        with gr.Row():
            skip_video_for_run_all = gr.Checkbox(label="skip_video_for_run_all", value=dv.skip_video_for_run_all, interactive=True)
            fps = gr.Number(label="fps", value=dv.fps, interactive=True)
            output_format = gr.Dropdown(label="output_format", choices=['PIL gif', 'FFMPEG mp4'], value='FFMPEG mp4', type="value", elem_id="output_format", interactive=True)
        with gr.Row():
            ffmpeg_location = gr.Textbox(label="ffmpeg_location", lines=1, interactive=True, value = dv.ffmpeg_location)
            ffmpeg_crf = gr.Number(label="ffmpeg_crf", interactive=True, value = dv.ffmpeg_crf)
            ffmpeg_preset = gr.Dropdown(label="ffmpeg_preset", choices=['veryslow', 'slower', 'slow', 'medium', 'fast', 'faster', 'veryfast', 'superfast', 'ultrafast'], interactive=True, value = dv.ffmpeg_preset, type="value")
            add_soundtrack = gr.Dropdown(label="add_soundtrack", choices=['None', 'File', 'Init Video'], value=dv.add_soundtrack, interactive=True, type="value")
            soundtrack_path = gr.Textbox(label="soundtrack_path", lines=1, interactive=True, value = dv.soundtrack_path)
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
        with gr.Row():
            store_frames_in_ram = gr.Checkbox(label="store_frames_in_ram", value=dv.store_frames_in_ram, interactive=True)
        with gr.Accordion('Frame Interpolation', open=False):
            i43 = gr.HTML("""
            Use RIFE and other Video Frame Interpolation methods to smooth out, slow-mo (or both) your output videos.</p>
             <p style="margin-top:1em">
                Supported engines:
                <ul style="list-style-type:circle; margin-left:1em; margin-bottom:1em">
                    <li>RIFE v4.6, v4.3 and v4.0. Recommended for now: v4.6.</li>
                    <li>RIFE v2.3 and other interpolation engines might come in the future.</li>
                </ul>
            </p>
             <p style="margin-top:1em">
                Important notes:
                <ul style="list-style-type:circle; margin-left:1em; margin-bottom:1em">
                    <li>ffmpeg has to be installed for this feature to work properly. If you can't have it, make sure to check the "keep_imgs" tab so that your interpolated frames are saved into HD'
                    <li>Frame Interpolation will *not* run if 'store_frames_in_ram' is enabled.</li>
                    <li>Audio (if provided) will be transferred to the interpolated video even if Slow-Mo is enabled.</li>
                    <li>Frame Interpolation will always save an .mp4 video even if you used GIF for the raw video.</li>
                </ul>
            </p>
            """)
            with gr.Row():
                frame_interpolation_engine = gr.Dropdown(label="frame_interpolation_engine", choices=['RIFE v4.0','RIFE v4.3','RIFE v4.6'], value=dv.frame_interpolation_engine, type="value", elem_id="frame_interpolation_engine", interactive=True)
            with gr.Row():
                frame_interpolation_x_amount = gr.Dropdown(label="frame_interpolation_x_amount", choices=['Disabled','x2','x3','x4','x5','x6','x7','x8','x9','x10'], value=dv.frame_interpolation_x_amount, type="value", elem_id="frame_interpolation_x_amount", interactive=True)
            with gr.Row():
                frame_interpolation_slow_mo_amount = gr.Dropdown(label="frame_interpolation_slow_mo_amount", choices=['Disabled','x2','x4','x8'], value=dv.frame_interpolation_slow_mo_amount, type="value", elem_id="frame_interpolation_slow_mo_amount", interactive=True)
            with gr.Row():
                frame_interpolation_keep_imgs = gr.Checkbox(label="frame_interpolation_keep_imgs", value=dv.frame_interpolation_keep_imgs, interactive=True)            

        # Video output settings END
    return locals()
    
def setup_deforum_setting_ui(self, is_img2img, is_extension = True):
    ds = SimpleNamespace(**setup_deforum_setting_dictionary(self, is_img2img, is_extension))
    return [ds.btn, ds.override_settings_with_file, ds.custom_settings_file, ds.animation_mode, ds.max_frames, ds.border, ds.angle, ds.zoom, ds.translation_x, ds.translation_y, ds.translation_z, ds.rotation_3d_x, ds.rotation_3d_y, ds.rotation_3d_z, ds.flip_2d_perspective, ds.perspective_flip_theta, ds.perspective_flip_phi, ds.perspective_flip_gamma, ds.perspective_flip_fv, ds.noise_schedule, ds.strength_schedule, ds.contrast_schedule, ds.cfg_scale_schedule, ds.enable_steps_scheduling, ds.steps_schedule, ds.fov_schedule, ds.near_schedule, ds.far_schedule, ds.seed_schedule, ds.enable_sampler_scheduling, ds.sampler_schedule, ds.enable_checkpoint_scheduling, ds.checkpoint_schedule, ds.kernel_schedule, ds.sigma_schedule, ds.amount_schedule, ds.threshold_schedule, ds.histogram_matching, ds.color_coherence, ds.color_coherence_video_every_N_frames, ds.diffusion_cadence, ds.noise_type, ds.perlin_w, ds.perlin_h, ds.perlin_octaves, ds.perlin_persistence, ds.use_depth_warping, ds.midas_weight, ds.near_plane, ds.far_plane, ds.fov, ds.padding_mode, ds.sampling_mode, ds.save_depth_maps, ds.video_init_path, ds.extract_nth_frame, ds.extract_from_frame, ds.extract_to_frame, ds.overwrite_extracted_frames, ds.use_mask_video, ds.video_mask_path, ds.interpolate_key_frames, ds.interpolate_x_frames, ds.resume_from_timestring, ds.resume_timestring, ds.prompts, ds.animation_prompts, ds.W, ds.H, ds.restore_faces, ds.tiling, ds.enable_hr, ds.firstphase_width, ds.firstphase_height, ds.seed, ds.sampler, ds.seed_enable_extras, ds.subseed, ds.subseed_strength, ds.seed_resize_from_w, ds.seed_resize_from_h, ds.steps, ds.ddim_eta, ds.n_batch, ds.make_grid, ds.grid_rows, ds.save_settings, ds.save_samples, ds.display_samples, ds.save_sample_per_step, ds.show_sample_per_step, ds.override_these_with_webui, ds.batch_name, ds.filename_format, ds.seed_behavior, ds.seed_iter_N, ds.use_init, ds.from_img2img_instead_of_link, ds.strength_0_no_init, ds.strength, ds.init_image, ds.use_mask, ds.use_alpha_as_mask, ds.invert_mask, ds.overlay_mask, ds.mask_file, ds.mask_contrast_adjust, ds.mask_brightness_adjust, ds.mask_overlay_blur, ds.fill, ds.full_res_mask, ds.full_res_mask_padding, ds.reroll_blank_frames, ds.skip_video_for_run_all, ds.fps, ds.output_format, ds.ffmpeg_location, ds.ffmpeg_crf, ds.ffmpeg_preset, ds.add_soundtrack, ds.soundtrack_path, ds.use_manual_settings, ds.render_steps, ds.max_video_frames, ds.path_name_modifier, ds.image_path, ds.mp4_path, ds.store_frames_in_ram, ds.frame_interpolation_engine, ds.frame_interpolation_x_amount, ds.frame_interpolation_slow_mo_amount, ds.frame_interpolation_keep_imgs, ds.parseq_manifest, ds.parseq_use_deltas, ds.hybrid_generate_inputframes, ds.hybrid_generate_human_masks, ds.hybrid_use_first_frame_as_init_image, ds.hybrid_motion, ds.hybrid_flow_method, ds.hybrid_composite, ds.hybrid_comp_mask_type, ds.hybrid_comp_mask_inverse, ds.hybrid_comp_mask_equalize, ds.hybrid_comp_mask_auto_contrast, ds.hybrid_comp_save_extra_frames, ds.hybrid_comp_alpha_schedule, ds.hybrid_comp_mask_blend_alpha_schedule, ds.hybrid_comp_mask_contrast_schedule, ds.hybrid_comp_mask_auto_contrast_cutoff_high_schedule, ds.hybrid_comp_mask_auto_contrast_cutoff_low_schedule, ds.i1, ds.i2, ds.i3, ds.i4, ds.i5, ds.i6, ds.i7, ds.i8, ds.i9, ds.i10, ds.i11, ds.i12, ds.i13, ds.i14, ds.i15, ds.i16, ds.i17, ds.i18, ds.i19, ds.i20, ds.i21, ds.i22, ds.i23, ds.i24, ds.i25, ds.i26, ds.i27, ds.i28, ds.i29, ds.i30, ds.i31, ds.i32, ds.i33, ds.i34, ds.i35, ds.i36, ds.i37, ds.i38, ds.i39, ds.i40, ds.i41, ds.i42, ds.i43]

def pack_anim_args(animation_mode, max_frames, border, angle, zoom, translation_x, translation_y, translation_z, rotation_3d_x, rotation_3d_y, rotation_3d_z, flip_2d_perspective, perspective_flip_theta, perspective_flip_phi, perspective_flip_gamma, perspective_flip_fv, noise_schedule, strength_schedule, contrast_schedule, cfg_scale_schedule, enable_steps_scheduling, steps_schedule, fov_schedule, near_schedule, far_schedule, seed_schedule, enable_sampler_scheduling, sampler_schedule, enable_checkpoint_scheduling, checkpoint_schedule, kernel_schedule, sigma_schedule, amount_schedule, threshold_schedule, histogram_matching, color_coherence, color_coherence_video_every_N_frames, diffusion_cadence, noise_type, perlin_w, perlin_h, perlin_octaves, perlin_persistence, use_depth_warping, midas_weight, near_plane, far_plane, fov, padding_mode, sampling_mode, save_depth_maps, video_init_path, extract_nth_frame, extract_from_frame, extract_to_frame, overwrite_extracted_frames, use_mask_video, video_mask_path, interpolate_key_frames, interpolate_x_frames, resume_from_timestring, resume_timestring, hybrid_generate_inputframes, hybrid_generate_human_masks, hybrid_use_first_frame_as_init_image, hybrid_motion, hybrid_flow_method, hybrid_composite, hybrid_comp_mask_type, hybrid_comp_mask_inverse, hybrid_comp_mask_equalize, hybrid_comp_mask_auto_contrast, hybrid_comp_save_extra_frames, hybrid_comp_alpha_schedule, hybrid_comp_mask_blend_alpha_schedule, hybrid_comp_mask_contrast_schedule, hybrid_comp_mask_auto_contrast_cutoff_high_schedule, hybrid_comp_mask_auto_contrast_cutoff_low_schedule):
    return locals()

def pack_args(W, H, restore_faces, tiling, enable_hr, firstphase_width, firstphase_height, seed, sampler, seed_enable_extras, subseed, subseed_strength, seed_resize_from_w, seed_resize_from_h, steps, ddim_eta, n_batch, make_grid, grid_rows, save_settings, save_samples, display_samples, save_sample_per_step, show_sample_per_step, override_these_with_webui, batch_name, filename_format, seed_behavior, seed_iter_N, use_init, from_img2img_instead_of_link, strength_0_no_init, strength, init_image, use_mask, use_alpha_as_mask, invert_mask, overlay_mask,  mask_file, mask_contrast_adjust, mask_brightness_adjust, mask_overlay_blur, fill, full_res_mask, full_res_mask_padding, reroll_blank_frames):
    precision = 'autocast' 
    scale = 7
    C = 4
    f = 8
    prompt = ""
    timestring = ""
    init_latent = None
    init_sample = None
    init_c = None
    noise_mask = None
    seed_internal = 0
    return locals()
    
def pack_video_args(skip_video_for_run_all, fps, output_format, ffmpeg_location, ffmpeg_crf, ffmpeg_preset, add_soundtrack, soundtrack_path, use_manual_settings, render_steps, max_video_frames, path_name_modifier, image_path, mp4_path, store_frames_in_ram, frame_interpolation_engine, frame_interpolation_x_amount, frame_interpolation_slow_mo_amount, frame_interpolation_keep_imgs):
    return locals()

def pack_parseq_args(parseq_manifest, parseq_use_deltas):
    return locals()

def process_args(self, p, override_settings_with_file, custom_settings_file, animation_mode, max_frames, border, angle, zoom, translation_x, translation_y, translation_z, rotation_3d_x, rotation_3d_y, rotation_3d_z, flip_2d_perspective, perspective_flip_theta, perspective_flip_phi, perspective_flip_gamma, perspective_flip_fv, noise_schedule, strength_schedule, contrast_schedule, cfg_scale_schedule, enable_steps_scheduling, steps_schedule, fov_schedule, near_schedule, far_schedule, seed_schedule, enable_sampler_scheduling, sampler_schedule, enable_checkpoint_scheduling, checkpoint_schedule, kernel_schedule, sigma_schedule, amount_schedule, threshold_schedule, histogram_matching, color_coherence, color_coherence_video_every_N_frames, diffusion_cadence, noise_type, perlin_w, perlin_h, perlin_octaves, perlin_persistence, use_depth_warping, midas_weight, near_plane, far_plane, fov, padding_mode, sampling_mode, save_depth_maps, video_init_path, extract_nth_frame, extract_from_frame, extract_to_frame, overwrite_extracted_frames, use_mask_video, video_mask_path, interpolate_key_frames, interpolate_x_frames, resume_from_timestring, resume_timestring, prompts, animation_prompts, W, H, restore_faces, tiling, enable_hr, firstphase_width, firstphase_height, seed, sampler, seed_enable_extras, subseed, subseed_strength, seed_resize_from_w, seed_resize_from_h, steps, ddim_eta, n_batch, make_grid, grid_rows, save_settings, save_samples, display_samples, save_sample_per_step, show_sample_per_step, override_these_with_webui, batch_name, filename_format, seed_behavior, seed_iter_N, use_init, from_img2img_instead_of_link, strength_0_no_init, strength, init_image, use_mask, use_alpha_as_mask, invert_mask, overlay_mask, mask_file, mask_contrast_adjust, mask_brightness_adjust, mask_overlay_blur, fill, full_res_mask, full_res_mask_padding, reroll_blank_frames, skip_video_for_run_all, fps, output_format, ffmpeg_location, ffmpeg_crf, ffmpeg_preset, add_soundtrack, soundtrack_path, use_manual_settings, render_steps, max_video_frames, path_name_modifier, image_path, mp4_path, store_frames_in_ram, frame_interpolation_engine, frame_interpolation_x_amount, frame_interpolation_slow_mo_amount, frame_interpolation_keep_imgs, parseq_manifest, parseq_use_deltas, hybrid_generate_inputframes, hybrid_generate_human_masks, hybrid_use_first_frame_as_init_image, hybrid_motion, hybrid_flow_method, hybrid_composite, hybrid_comp_mask_type, hybrid_comp_mask_inverse, hybrid_comp_mask_equalize, hybrid_comp_mask_auto_contrast, hybrid_comp_save_extra_frames, hybrid_comp_alpha_schedule, hybrid_comp_mask_blend_alpha_schedule, hybrid_comp_mask_contrast_schedule, hybrid_comp_mask_auto_contrast_cutoff_high_schedule, hybrid_comp_mask_auto_contrast_cutoff_low_schedule, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, i16, i17, i18, i19, i20, i21, i22, i23, i24, i25, i26, i27, i28, i29, i30, i31, i32, i33, i34, i35, i36, i37, i38, i39, i40, i41, i42, i43):
    args_dict = pack_args(W, H, restore_faces, tiling, enable_hr, firstphase_width, firstphase_height, seed, sampler, seed_enable_extras, subseed, subseed_strength, seed_resize_from_w, seed_resize_from_h, steps, ddim_eta, n_batch, make_grid, grid_rows, save_settings, save_samples, display_samples, save_sample_per_step, show_sample_per_step, override_these_with_webui, batch_name, filename_format, seed_behavior, seed_iter_N, use_init, from_img2img_instead_of_link, strength_0_no_init, strength, init_image, use_mask, use_alpha_as_mask, invert_mask, overlay_mask, mask_file, mask_contrast_adjust, mask_brightness_adjust, mask_overlay_blur, fill, full_res_mask, full_res_mask_padding, reroll_blank_frames)
    anim_args_dict = pack_anim_args(animation_mode, max_frames, border, angle, zoom, translation_x, translation_y, translation_z, rotation_3d_x, rotation_3d_y, rotation_3d_z, flip_2d_perspective, perspective_flip_theta, perspective_flip_phi, perspective_flip_gamma, perspective_flip_fv, noise_schedule, strength_schedule, contrast_schedule, cfg_scale_schedule, enable_steps_scheduling, steps_schedule, fov_schedule, near_schedule, far_schedule, seed_schedule, enable_sampler_scheduling, sampler_schedule, enable_checkpoint_scheduling, checkpoint_schedule, kernel_schedule, sigma_schedule, amount_schedule, threshold_schedule, histogram_matching, color_coherence, color_coherence_video_every_N_frames, diffusion_cadence, noise_type, perlin_w, perlin_h, perlin_octaves, perlin_persistence, use_depth_warping, midas_weight, near_plane, far_plane, fov, padding_mode, sampling_mode, save_depth_maps, video_init_path, extract_nth_frame, extract_from_frame, extract_to_frame, overwrite_extracted_frames, use_mask_video, video_mask_path, interpolate_key_frames, interpolate_x_frames, resume_from_timestring, resume_timestring, hybrid_generate_inputframes, hybrid_generate_human_masks, hybrid_use_first_frame_as_init_image, hybrid_motion, hybrid_flow_method, hybrid_composite, hybrid_comp_mask_type, hybrid_comp_mask_inverse, hybrid_comp_mask_equalize, hybrid_comp_mask_auto_contrast, hybrid_comp_save_extra_frames, hybrid_comp_alpha_schedule, hybrid_comp_mask_blend_alpha_schedule, hybrid_comp_mask_contrast_schedule, hybrid_comp_mask_auto_contrast_cutoff_high_schedule, hybrid_comp_mask_auto_contrast_cutoff_low_schedule)
    video_args_dict = pack_video_args(skip_video_for_run_all, fps, output_format, ffmpeg_location, ffmpeg_crf, ffmpeg_preset, add_soundtrack, soundtrack_path, use_manual_settings, render_steps, max_video_frames, path_name_modifier, image_path, mp4_path, store_frames_in_ram, frame_interpolation_engine, frame_interpolation_x_amount, frame_interpolation_slow_mo_amount, frame_interpolation_keep_imgs)

    parseq_args_dict = pack_parseq_args(parseq_manifest, parseq_use_deltas)

    import json
    
    root = SimpleNamespace(**Root())
    root.p = p
    #root.prompts = json.loads(prompts)#TODO make proper animation_mode=None handling
    root.animation_prompts = json.loads(animation_prompts)
    
    from deforum_helpers.settings import load_args
    
    if override_settings_with_file:
        load_args(args_dict, anim_args_dict, parseq_args_dict, custom_settings_file, root)
    
    print(f"Additional models path: {root.models_path}")
    if not os.path.exists(root.models_path):
        os.mkdir(root.models_path)

    args = SimpleNamespace(**args_dict)
    anim_args = SimpleNamespace(**anim_args_dict)
    video_args = SimpleNamespace(**video_args_dict)
    parseq_args = SimpleNamespace(**parseq_args_dict)

    # TODO handle webui sampler settings override
    
    if override_these_with_webui:
        args.n_batch = p.batch_size
        args.W = p.width
        args.H = p.height
        args.restore_faces = p.restore_faces
        args.tiling = p.tiling
        args.enable_hr = p.enable_hr
        args.firstphase_width = p.firstphase_width
        args.firstphase_height = p.firstphase_height
        args.seed = p.seed
        args.seed_enable_extras = p.seed_enable_extras
        args.subseed = p.subseed
        args.subseed_strength = p.subseed_strength
        args.seed_resize_from_w = p.seed_resize_from_w
        args.seed_resize_from_h = p.seed_resize_from_h
        args.steps = p.steps        
        args.ddim_eta = p.ddim_eta
        args.W, args.H = map(lambda x: x - x % 64, (args.W, args.H))
        args.steps = p.steps
        args.seed = p.seed
        args.sampler = p.sampler_name
    else:
        p.width, p.height = map(lambda x: x - x % 64, (args.W, args.H))
        p.steps = args.steps
        p.seed = args.seed
        p.sampler_name = args.sampler
        p.batch_size = args.n_batch
        p.restore_faces = args.restore_faces
        p.tiling = args.tiling
        p.enable_hr = args.enable_hr
        p.firstphase_width = args.firstphase_width
        p.firstphase_height = args.firstphase_height
        p.seed_enable_extras = args.seed_enable_extras
        p.subseed = args.subseed
        p.subseed_strength = args.subseed_strength
        p.seed_resize_from_w = args.seed_resize_from_w
        p.seed_resize_from_h = args.seed_resize_from_h
        p.fill = args.fill
        p.ddim_eta = args.ddim_eta


    # TODO: Handle batch name dynamically?
    current_arg_list = [args, anim_args, video_args, parseq_args]
    #batch_name = replace_args(batch_name, current_arg_list)
    #print_args(args)
    args.outdir = os.path.join(p.outpath_samples, batch_name)
    root.outpath_samples = args.outdir
    args.outdir = os.path.join(os.getcwd(), args.outdir)
    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)
    
    args.seed = get_fixed_seed(args.seed)
        
    args.timestring = time.strftime('%Y%m%d%H%M%S')
    args.strength = max(0.0, min(1.0, args.strength))

    if not args.use_init:
        args.init_image = None
        
    if anim_args.animation_mode == 'None':
        anim_args.max_frames = 1
    elif anim_args.animation_mode == 'Video Input':
        args.use_init = True
    
    return root, args, anim_args, video_args, parseq_args


#def replace_args(text, args_dict):
#  return text.format(**args_dict.__dict__)
 

def print_args(args):
    print("ARGS: /n")
    for key, value in args.__dict__.items():
        print(f"{key}: {value}")
        
def find_ffmpeg_binary():
    package_path = None
    for package in ['imageio_ffmpeg', 'imageio-ffmpeg']:
        try:
            package_path = resource_filename(package, '')
            break
        except:
            pass

    if package_path:
        binaries_path = os.path.join(package_path, 'binaries')
        if os.path.exists(binaries_path):
            files = [os.path.join(binaries_path, f) for f in os.listdir(binaries_path) if f.startswith("ffmpeg-")]
            files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
            return files[0] if files else 'ffmpeg'
    return 'ffmpeg'


# def replace_args(text, args_list):
    # for args_dict in args_list:
        # #print(f"Arg list: {args_dict}")
        # args_dict = vars(args_dict)
        # for key, value in args_dict.items():
            # print(f"{key}: {value}")
        # text = text.format_map(args_dict)
    # return text


