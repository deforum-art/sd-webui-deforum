from modules.shared import cmd_opts
from modules.processing import get_fixed_seed
import modules.shared as sh
import modules.paths as ph
import os
from .frame_interpolation import set_interp_out_fps, gradio_f_interp_get_fps_and_fcount, process_rife_vid_upload_logic
from .video_audio_utilities import find_ffmpeg_binary, ffmpeg_stitch_video, direct_stitch_vid_from_frames
from .gradio_funcs import *

def Root():
    device = sh.device
    models_path = ph.models_path + '/Deforum'
    half_precision = not cmd_opts.no_half
    mask_preset_names = ['everywhere','init_mask','video_mask']
    p = None
    frames_cache = []
    initial_seed = None
    initial_info = None
    first_frame = None
    outpath_samples = ""
    animation_prompts = None
    color_corrections = None 
    initial_clipskip = None
    return locals()

def DeforumAnimArgs():

    #@markdown ####**Animation:**
    animation_mode = '2D' #@param ['None', '2D', '3D', 'Video Input', 'Interpolation'] {type:'string'}
    max_frames = 120 #@param {type:"number"}
    border = 'replicate' #@param ['wrap', 'replicate'] {type:'string'}
    #@markdown ####**Motion Parameters:**
    angle = "0:(0)"#@param {type:"string"}
    zoom = "0:(1.0025+0.002*sin(1.25*3.14*t/30))"#@param {type:"string"}
    translation_x = "0:(0)"#@param {type:"string"}
    translation_y = "0:(0)"#@param {type:"string"}
    translation_z = "0:(1.75)"#@param {type:"string"}
    rotation_3d_x = "0:(0)"#@param {type:"string"}
    rotation_3d_y = "0:(0)"#@param {type:"string"}
    rotation_3d_z = "0:(0)"#@param {type:"string"}
    enable_perspective_flip = False #@param {type:"boolean"}
    perspective_flip_theta = "0:(0)"#@param {type:"string"}
    perspective_flip_phi = "0:(0)"#@param {type:"string"}
    perspective_flip_gamma = "0:(0)"#@param {type:"string"}
    perspective_flip_fv = "0:(53)"#@param {type:"string"}
    noise_schedule = "0: (0.065)"#@param {type:"string"}
    strength_schedule = "0: (0.65)"#@param {type:"string"}
    contrast_schedule = "0: (1.0)"#@param {type:"string"}
    cfg_scale_schedule = "0: (7)"
    enable_steps_scheduling = False#@param {type:"boolean"}
    steps_schedule = "0: (25)"#@param {type:"string"}
    fov_schedule = "0: (70)"
    near_schedule = "0: (200)"
    far_schedule = "0: (10000)"
    seed_schedule = "0:(5), 1:(-1), 219:(-1), 220:(5)"
    
    # Sampler Scheduling
    enable_sampler_scheduling = False #@param {type:"boolean"}
    sampler_schedule = '0: ("Euler a")'

    # Composable mask scheduling
    use_noise_mask = False
    mask_schedule = '0: ("!({everywhere}^({init_mask}|{video_mask}) ) ")'
    noise_mask_schedule = '0: ("!({everywhere}^({init_mask}|{video_mask}) ) ")'
    # Checkpoint Scheduling
    enable_checkpoint_scheduling = False#@param {type:"boolean"}
    checkpoint_schedule = '0: ("model1.ckpt"), 100: ("model2.ckpt")'
    
    # CLIP skip Scheduling
    enable_clipskip_scheduling = False #@param {type:"boolean"}
    clipskip_schedule = '0: (2)'

    # Anti-blur
    kernel_schedule = "0: (5)"
    sigma_schedule = "0: (1.0)"
    amount_schedule = "0: (0.35)"
    threshold_schedule = "0: (0.0)"
    # Hybrid video
    hybrid_comp_alpha_schedule = "0:(1)" #@param {type:"string"}
    hybrid_comp_mask_blend_alpha_schedule = "0:(0.5)" #@param {type:"string"}
    hybrid_comp_mask_contrast_schedule = "0:(1)" #@param {type:"string"}
    hybrid_comp_mask_auto_contrast_cutoff_high_schedule =  "0:(100)" #@param {type:"string"}
    hybrid_comp_mask_auto_contrast_cutoff_low_schedule =  "0:(0)" #@param {type:"string"}

    #@markdown ####**Coherence:**
    color_coherence = 'Match Frame 0 LAB' #@param ['None', 'Match Frame 0 HSV', 'Match Frame 0 LAB', 'Match Frame 0 RGB', 'Video Input'] {type:'string'}
    color_coherence_video_every_N_frames = 1 #@param {type:"integer"}
    color_force_grayscale = False #@param {type:"boolean"}
    diffusion_cadence = '2' #@param ['1','2','3','4','5','6','7','8'] {type:'string'}

    #@markdown ####**Noise settings:**
    noise_type = 'perlin' #@param ['uniform', 'perlin'] {type:'string'}
    # Perlin params
    perlin_w = 8 #@param {type:"number"}
    perlin_h = 8 #@param {type:"number"}
    perlin_octaves = 4 #@param {type:"number"}
    perlin_persistence = 0.5 #@param {type:"number"}

    #@markdown ####**3D Depth Warping:**
    use_depth_warping = True #@param {type:"boolean"}
    midas_weight = 0.2 #@param {type:"number"}

    padding_mode = 'border'#@param ['border', 'reflection', 'zeros'] {type:'string'}
    sampling_mode = 'bicubic'#@param ['bicubic', 'bilinear', 'nearest'] {type:'string'}
    save_depth_maps = False #@param {type:"boolean"}

    #@markdown ####**Video Input:**
    video_init_path ='https://github.com/hithereai/d/releases/download/m/vid.mp4' #@param {type:"string"}
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
    hybrid_motion_use_prev_img = False #@param {type:"boolean"}
    hybrid_flow_method = "Farneback" #@param ['DIS Medium','Farneback']
    hybrid_composite = False #@param {type:"boolean"}
    hybrid_comp_mask_type = "None" #@param ['None', 'Depth', 'Video Depth', 'Blend', 'Difference']
    hybrid_comp_mask_inverse = False #@param {type:"boolean"}
    hybrid_comp_mask_equalize = "None" #@param  ['None','Before','After','Both']
    hybrid_comp_mask_auto_contrast = False #@param {type:"boolean"}
    hybrid_comp_save_extra_frames = False #@param {type:"boolean"}

    #@markdown ####**Resume Animation:**
    resume_from_timestring = False #@param {type:"boolean"}
    resume_timestring = "20220829210106" #@param {type:"string"}

    return locals()

# def DeforumPrompts():
    # return
    
def DeforumAnimPrompts():
    return r"""{
    "0": "tiny cute swamp bunny, highly detailed, intricate, ultra hd, sharp photo, crepuscular rays, in focus, by tomasz alen kopera",
    "30": "anthropomorphic clean cat, surrounded by fractals, epic angle and pose, symmetrical, 3d, depth of field, ruan jia and fenghua zhong",
    "60": "a beautiful coconut --neg photo, realistic",
    "90": "a beautiful durian, trending on Artstation"
}
    """

def DeforumArgs():
    #@markdown **Image Settings**
    W = 512 #@param
    H = 512 #@param
    W, H = map(lambda x: x - x % 64, (W, H))  # resize to integer multiple of 64

    #@markdonw **Webui stuff**
    tiling = False
    restore_faces = False
    # firstphase_width = 0
    # firstphase_height = 0
    seed_enable_extras = False
    subseed = -1
    subseed_strength = 0
    seed_resize_from_w = 0
    seed_resize_from_h = 0
    
    #@markdown **Sampling Settings**
    seed = -1 #@param
    sampler = 'euler_ancestral' #@param ["klms","dpm2","dpm2_ancestral","heun","euler","euler_ancestral","plms", "ddim"]
    steps = 25 #@param
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
    # make_grid = False #@param {type:"boolean"}
    # grid_rows = 2 #@param 
    outdir = ""#get_output_folder(output_path, batch_name)

    #@markdown **Init Settings**
    use_init = False #@param {type:"boolean"}
    strength = 0.0 #@param {type:"number"}
    strength_0_no_init = True # Set the strength to 0 automatically when no init image is used
    init_image = "https://github.com/hithereai/d/releases/download/m/kaba.png" #@param {type:"string"}
    # Whiter areas of the mask are areas that change more
    use_mask = False #@param {type:"boolean"}
    use_alpha_as_mask = False # use the alpha channel of the init image as the mask
    mask_file = "https://github.com/hithereai/d/releases/download/m/mask.jpg" #@param {type:"string"}
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
    mask_image = None
    noise_mask = None
    seed_internal = 0

    return locals()

def keyframeExamples():
    return '''{
    "0": "https://user-images.githubusercontent.com/121192995/215279228-1673df8a-f919-4380-b04c-19379b2041ff.png",
    "50": "https://user-images.githubusercontent.com/121192995/215279281-7989fd6f-4b9b-4d90-9887-b7960edd59f8.png",
    "100": "https://user-images.githubusercontent.com/121192995/215279284-afc14543-d220-4142-bbf4-503776ca2b8b.png",
    "150": "https://user-images.githubusercontent.com/121192995/215279286-23378635-85b3-4457-b248-23e62c048049.jpg",
    "200": "https://user-images.githubusercontent.com/121192995/215279228-1673df8a-f919-4380-b04c-19379b2041ff.png"
}'''

def LoopArgs():
    use_looper = False
    init_images = keyframeExamples()
    image_strength_schedule = "0:(0.6)"
    blendFactorMax = "0:(0.6)"
    blendFactorSlope = "0:(0.25)"
    tweening_frames_schedule = "0:(0.25)"
    color_correction_factor = "0:(0.075)"
    return locals()

def ParseqArgs():
    parseq_manifest = None
    parseq_use_deltas = True
    return locals()
    
def DeforumOutputArgs():
    skip_video_for_run_all = False #@param {type: 'boolean'}
    fps = 15 #@param {type:"number"}
    #@markdown **Manual Settings**
    image_path = "C:/SD/20230124234916_%05d.png" #@param {type:"string"}
    mp4_path = "testvidmanualsettings.mp4" #@param {type:"string"}
    ffmpeg_location = find_ffmpeg_binary()
    ffmpeg_crf = '17'
    ffmpeg_preset = 'slow'
    add_soundtrack = 'None' #@param ["File","Init Video"]
    soundtrack_path = "https://freetestdata.com/wp-content/uploads/2021/09/Free_Test_Data_1MB_MP3.mp3"
    render_steps = False  #@param {type: 'boolean'}
    path_name_modifier = "x0_pred" #@param ["x0_pred","x"]
    # max_video_frames = 200 #@param {type:"string"}
    store_frames_in_ram = False #@param {type: 'boolean'}
    frame_interpolation_engine = "RIFE v4.6" #@param ["RIFE v4.0","RIFE v4.3","RIFE v4.6"]
    frame_interpolation_x_amount = "Disabled" #"Disabled" #@param ["Disabled" + all values from x2 to x10]
    frame_interpolation_slow_mo_amount = "Disabled" #@param ["Disabled","x2","x4","x8"]
    frame_interpolation_keep_imgs = False #@param {type: 'boolean'}
    return locals()
    
import gradio as gr
import os
import time
from types import SimpleNamespace

i1_store_backup = "<p style=\"text-align:center;font-weight:bold;margin-bottom:0em\">Deforum extension for auto1111 — version 2.0b</p>"
i1_store = i1_store_backup

mask_fill_choices=['fill', 'original', 'latent noise', 'latent nothing']

def setup_deforum_setting_dictionary(self, is_img2img, is_extension = True):
    d = SimpleNamespace(**DeforumArgs()) #default args
    da = SimpleNamespace(**DeforumAnimArgs()) #default anim args
    dp = SimpleNamespace(**ParseqArgs()) #default parseq ars
    dv = SimpleNamespace(**DeforumOutputArgs()) #default video args
    dloopArgs = SimpleNamespace(**LoopArgs())
    if not is_extension:
        with gr.Row():
            btn = gr.Button("Click here after the generation to show the video")
        with gr.Row():
            i1 = gr.HTML(i1_store, elem_id='deforum_header')
    else:
        btn = i1 = gr.HTML("")
    with gr.Accordion("Info, Links and Help", open=False):
            gr.HTML("""<strong>Made by <a href="https://deforum.github.io">deforum.github.io</a>, port for AUTOMATIC1111's webui maintained by <a href="https://github.com/kabachuha">kabachuha</a></strong>""")
            gr.HTML("""<a  style="color:SteelBlue" href="https://github.com/deforum-art/deforum-for-automatic1111-webui/wiki/FAQ-&-Troubleshooting">FOR HELP CLICK HERE</a""", elem_id="for_help_click_here")
            gr.HTML("""<ul style="list-style-type:circle; margin-left:1em">
            <li>The code for this extension: <a  style="color:SteelBlue" href="https://github.com/deforum-art/deforum-for-automatic1111-webui">here</a>.</li>
            <li>Join the <a style="color:SteelBlue" href="https://discord.gg/deforum">official Deforum Discord</a> to share your creations and suggestions.</li>
            <li>Official Deforum Wiki: <a style="color:SteelBlue" href="https://github.com/deforum-art/deforum-for-automatic1111-webui/wiki">here</a>.</li>
            <li>Anime-inclined great guide (by FizzleDorf) with lots of examples: <a style="color:SteelBlue" href="https://rentry.org/AnimAnon-Deforum">here</a>.</li>
            <li>For advanced keyframing with Math functions, see <a style="color:SteelBlue" href="https://github.com/deforum-art/deforum-for-automatic1111-webui/wiki/Maths-in-Deforum">here</a>.</li>
            <li>Alternatively, use <a style="color:SteelBlue" href="https://sd-parseq.web.app/deforum">sd-parseq</a> as a UI to define your animation schedules (see the Parseq section in the Keyframes tab).</li>
            <li><a style="color:SteelBlue" href="https://www.framesync.xyz/">framesync.xyz</a> is also a good option, it makes compact math formulae for Deforum keyframes by selecting various waveforms.</li>
            <li>The other site allows for making keyframes using <a style="color:SteelBlue" href="https://www.chigozie.co.uk/keyframe-string-generator/">interactive splines and Bezier curves</a> (select Disco output format).</li>
            <li>If you want to use Width/Height which are not multiples of 64, please change noise_type to 'Uniform', in Keyframes --> Noise.</li>
            </ul>
            <italic>If you liked this extension, please <a style="color:SteelBlue" href="https://github.com/deforum-art/deforum-for-automatic1111-webui">give it a star on GitHub</a>!</italic> 😊""")
  
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
    with gr.Blocks():
        with gr.Tab('Run'):
            # Sampling settings START
            with gr.Accordion('General Image Sampling Settings', open=True):
                with gr.Row().style(equal_height=False):
                    # with gr.Column(variant='compact'):
                    with gr.Column():
                        from modules.sd_samplers import samplers_for_img2img
                        with gr.Row(variant='compact'):
                            sampler = gr.Dropdown(label="Sampler", choices=[x.name for x in samplers_for_img2img], value=samplers_for_img2img[0].name, type="value", elem_id="sampler", interactive=True)
                            steps = gr.Slider(label="Steps", minimum=0, maximum=200, step=1, value=d.steps, interactive=True)
                        with gr.Row(variant='compact'):
                            with gr.Column(scale=4):
                                W = gr.Slider(label="Width", minimum=64, maximum=2048, step=64, value=d.W, interactive=True)
                                H = gr.Slider(label="Height", minimum=64, maximum=2048, step=64, value=d.H, interactive=True)
                            with gr.Column(scale=4):
                                seed = gr.Number(label="Seed", value=d.seed, interactive=True, precision=0)
                                batch_name = gr.Textbox(label="Batch name", lines=1, interactive=True, value = d.batch_name)
                                with gr.Row(visible=False):
                                    filename_format = gr.Textbox(label="Filename format", lines=1, interactive=True, value = d.filename_format, visible=False)
                        with gr.Accordion('Subseed controls & More', open=False):
                            # Not visible until fixed, 06-02-23
                            with gr.Row(visible=False):
                                restore_faces = gr.Checkbox(label='Restore Faces', value=d.restore_faces)
                            with gr.Row(variant='compact'):
                                seed_enable_extras = gr.Checkbox(label="Enable subseed controls", value=False)
                                subseed = gr.Number(label="Subseed", value=d.subseed, interactive=True, precision=0)
                                subseed_strength = gr.Slider(label="Subseed strength", minimum=0, maximum=1, step=0.01, value=d.subseed_strength, interactive=True)
                            with gr.Row(variant='compact'):
                                seed_resize_from_w = gr.Slider(minimum=0, maximum=2048, step=64, label="Resize seed from width", value=0)
                                seed_resize_from_h = gr.Slider(minimum=0, maximum=2048, step=64, label="Resize seed from height", value=0)
                            with gr.Row(variant='compact'):
                                ddim_eta = gr.Number(label="DDIM ETA", value=d.ddim_eta, interactive=True)
                                tiling = gr.Checkbox(label='Tiling', value=False)
                                n_batch = gr.Number(label="N Batch", value=d.n_batch, interactive=True, precision=0, visible=False)
                        # NOT VISIBLE IN THE UI!
                        with gr.Row(visible=False):
                            save_settings = gr.Checkbox(label="save_settings", value=d.save_settings, interactive=True)
                        # NOT VISIBLE IN THE UI!
                        with gr.Row(visible=False):
                            save_samples = gr.Checkbox(label="save_samples", value=d.save_samples, interactive=True)
                            display_samples = gr.Checkbox(label="display_samples", value=False, interactive=False)
                        with gr.Row(visible=False):
                            save_sample_per_step = gr.Checkbox(label="Save sample per step", value=d.save_sample_per_step, interactive=True)
                            show_sample_per_step = gr.Checkbox(label="Show sample per step", value=d.show_sample_per_step, interactive=True)
            with gr.Accordion('Run from Settings file', open=False):
                with gr.Row(variant='compact'):
                    override_settings_with_file = gr.Checkbox(label="Override settings", value=False, interactive=True)
                    custom_settings_file = gr.Textbox(label="Custom settings file", lines=1, interactive=True)
            with gr.Accordion('Resume Animation', open=False):
                with gr.Row(variant='compact'):
                    resume_from_timestring = gr.Checkbox(label="Resume from timestring", value=da.resume_from_timestring, interactive=True)
                    resume_timestring = gr.Textbox(label="Resume timestring", lines=1, value = da.resume_timestring, interactive=True)
        # Animation settings 'Key' tab
        with gr.Tab('Keyframes'):
            #TODO make a some sort of the original dictionary parsing
            # Main top animation settings
            # with gr.Tab('Main Settings', elem_id='main_settings_tab') as main_settings_tab:
            # with gr.Accordion('Main Settings', open=True) as a1:
            with gr.Row(variant='compact'):
                with gr.Column(scale=5):
                    with gr.Row(variant='compact'):
                        with gr.Column(scale=2):
                            animation_mode = gr.Radio(['2D', '3D', 'Interpolation', 'Video Input'], label="Animation mode", value=da.animation_mode, elem_id="animation_mode")
                        with gr.Column(scale=1, min_width=180):
                            border = gr.Radio(['replicate', 'wrap'], label="Border", value=da.border, elem_id="border")
                # with gr.Column(scale=1, min_width=115) as max_frames_column:
                    
            with gr.Row(variant='compact') as diffusion_cadence_row:
                with gr.Column(scale=5):
                    with gr.Row(variant='compact'):
                        with gr.Column(scale=2):
                            diffusion_cadence = gr.Slider(label="Cadence", minimum=1, maximum=50, step=1, value=da.diffusion_cadence, interactive=True)
                        with gr.Column(scale=1, min_width=180):
                            # max_frames = gr.Number(label="Max frames", value=da.max_frames, interactive=True, precision=0, visible=True)
                            max_frames = gr.Slider(label="Max frames", minimum=1, maximum=9999, step=1, value=da.max_frames, interactive=True)
                            
                # diffusion_cadence = gr.Slider(label="Cadence", minimum=1, maximum=50, step=1, value=da.diffusion_cadence, interactive=True)
                # max_frames = gr.Number(label="Max frames", value=da.max_frames, interactive=True, precision=0, visible=True)
            # loopArgs
            # testme = gr.HTML(value='HIHTHERE', visible=True, elem_id='testme')
            with gr.Accordion('Guided Images', open=False, elem_id='guided_images_accord') as guided_images_accord:
                with gr.Accordion('*READ ME before you use this mode!*', open=False):
                    gr.HTML("""You can use this as a guided image tool or as a looper depending on your settings in the keyframe images field. 
                               Set the keyframes and the images that you want to show up. 
                               Note: the number of frames between each keyframe should be greater than the tweening frames.""")
                    #    In later versions this should be also in the strength schedule, but for now you need to set it.
                    gr.HTML("""Prerequisites and Important Info: 
                               <ul style="list-style-type:circle; margin-left:2em; margin-bottom:0em">
                                   <li>This mode works ONLY with 2D/3D animation modes. Interpolation and Video Input modes aren't supported.</ li>
                                   <li>Set Init tab's strength slider greater than 0. Recommended value (.65 - .80).</ li>
                                   <li>Set 'seed_behavior' to 'schedule' under the Seed Scheduling section below.</li>
                                </ul>
                            """)
                    gr.HTML("""Looping recommendations: 
                                <ul style="list-style-type:circle; margin-left:2em; margin-bottom:0em">
                                    <li>seed_schedule should start and end on the same seed. <br />
                                        Example: seed_schedule could use 0:(5), 1:(-1), 219:(-1), 220:(5)</li>
                                    <li>The 1st and last keyframe images should match.</li>
                                    <li>Set your total number of keyframes to be 21 more than the last inserted keyframe image. <br />
                                        Example: Default args should use 221 as total keyframes.</li>
                                    <li>Prompts are stored in JSON format. If you've got an error, check it in validator, <a style="color:SteelBlue" href="https://odu.github.io/slingjsonlint/">like here</a></li>
                                </ul>
                            """)
                with gr.Row():
                    use_looper = gr.Checkbox(label="Use guided images for the next run", value=False, interactive=True)
                with gr.Row():
                    init_images = gr.Textbox(label="Images to use for keyframe guidance", lines=9, value = keyframeExamples(), interactive=True)
                gr.HTML("""strength schedule might be better if this is higher, around .75 during the keyfames you want to switch on""")
                with gr.Row():
                    image_strength_schedule = gr.Textbox(label="Image strength schedule", lines=1, value = "0:(.75)", interactive=True)
                gr.HTML("""blendFactor = blendFactorMax - blendFactorSlope * cos((frame % tweening_frames_schedule) / (tweening_frames_schedule / 2))""")
                with gr.Row():
                    blendFactorMax = gr.Textbox(label="BlendFactorMax", lines=1, value = "0:(.35)", interactive=True)
                with gr.Row():
                    blendFactorSlope = gr.Textbox(label="BlendFactorSlope", lines=1, value = "0:(.25)", interactive=True)
                with gr.Row():
                    gr.HTML("""number of frames this will calculated over. After each insersion frame.""")
                with gr.Row():
                    tweening_frames_schedule = gr.Textbox(label="Tweening frames schedule", lines=1, value = "0:(20)", interactive=True)
                with gr.Row():
                    gr.HTML("""the amount each frame during a tweening step to use the new images colors""")
                with gr.Row():
                    color_correction_factor = gr.Textbox(label="Color correction factor", lines=1, value = "0:(.075)", interactive=True)
            # 2D + 3D Motion
            # with gr.Accordion('2D Motion', open=True) as motion_accord:
            # schedule_msg = gr.HTML('*Schedules*:')
            with gr.Tabs(elem_id='TTTTTTTT'):
                with gr.TabItem('Strength'):
                    strength_schedule = gr.Textbox(label="Strength schedule", lines=1, value = da.strength_schedule, interactive=True)
                with gr.TabItem('CFG'):
                    cfg_scale_schedule = gr.Textbox(label="CFG scale schedule", lines=1, value = da.cfg_scale_schedule, interactive=True)
                with gr.TabItem('Seed') as a3:
                    with gr.Row():
                        seed_behavior = gr.Radio(['iter', 'fixed', 'random', 'ladder', 'alternate', 'schedule'], label="Seed behavior", value=d.seed_behavior, elem_id="seed_behavior")
                    with gr.Row() as seed_iter_N_row:
                        seed_iter_N = gr.Number(label="Seed iter N", value=d.seed_iter_N, interactive=True, precision=0)
                    with gr.Row(visible=False) as seed_schedule_row:
                        seed_schedule = gr.Textbox(label="Seed schedule", lines=1, value = da.seed_schedule, interactive=True)
                # Steps Scheduling
                # with gr.Tab('Special Schedules'
                # with gr.Accordion('Special Schedules', open=False) as a13:
                with gr.TabItem('Steps') as a13:
                    with gr.Row():
                        enable_steps_scheduling = gr.Checkbox(label="Enable steps scheduling", value=da.enable_steps_scheduling, interactive=True)
                    with gr.Row():
                        steps_schedule = gr.Textbox(label="Steps schedule", lines=1, value = da.steps_schedule, interactive=True)
                # Sampler Scheduling
                with gr.TabItem('Sampler') as a14:
                    with gr.Row():
                        enable_sampler_scheduling = gr.Checkbox(label="Enable sampler scheduling", value=da.enable_sampler_scheduling, interactive=True)
                    with gr.Row():
                        sampler_schedule = gr.Textbox(label="Sampler schedule", lines=1, value = da.sampler_schedule, interactive=True)
                # Checkpoint Scheduling
                with gr.TabItem('Checkpoint') as a15:
                    with gr.Row():
                        enable_checkpoint_scheduling = gr.Checkbox(label="Enable checkpoint scheduling", value=da.enable_checkpoint_scheduling, interactive=True)
                    with gr.Row():
                        checkpoint_schedule = gr.Textbox(label="Checkpoint schedule", lines=1, value = da.checkpoint_schedule, interactive=True)
                with gr.TabItem('CLIP Skip', open=False) as a16:
                    with gr.Row():
                        enable_clipskip_scheduling = gr.Checkbox(label="Enable CLIP skip scheduling", value=da.enable_clipskip_scheduling, interactive=True)
                    with gr.Row():
                        clipskip_schedule = gr.Textbox(label="CLIP skip schedule", lines=1, value = da.clipskip_schedule, interactive=True)
                    
                    
            with gr.Tab('Motion') as motion_accord:
                with gr.Column(visible=True) as only_2d_motion_column:
                    with gr.Row(variant='compact'):
                        angle = gr.Textbox(label="Angle", lines=1, value = da.angle, interactive=True)
                    with gr.Row(variant='compact'):
                        zoom = gr.Textbox(label="Zoom", lines=1, value = da.zoom, interactive=True)
                with gr.Column(visible=True) as both_anim_mode_motion_params_column:
                    with gr.Row(variant='compact'):
                        translation_x = gr.Textbox(label="Translation X", lines=1, value = da.translation_x, interactive=True)
                    with gr.Row(variant='compact'):
                        translation_y = gr.Textbox(label="Translation Y", lines=1, value = da.translation_y, interactive=True)
                # 3D-only Motion
                with gr.Column(visible=False) as only_3d_motion_column:
                    with gr.Row(variant='compact'):
                        translation_z = gr.Textbox(label="Translation Z", lines=1, value = da.translation_z, interactive=True)
                    with gr.Row(variant='compact'):
                        rotation_3d_x = gr.Textbox(label="Rotation 3D X", lines=1, value = da.rotation_3d_x, interactive=True)
                    with gr.Row(variant='compact'):
                        rotation_3d_y = gr.Textbox(label="Rotation 3D Y", lines=1, value = da.rotation_3d_y, interactive=True)
                    with gr.Row(variant='compact'):
                        rotation_3d_z = gr.Textbox(label="Rotation 3D Z", lines=1, value = da.rotation_3d_z, interactive=True)
                # 3D Depth Warping
                with gr.Accordion('Depth Warping', visible=False, open=False) as depth_3d_warping_accord:
                    with gr.Row(variant='compact'):
                        use_depth_warping = gr.Checkbox(label="Use depth warping", value=da.use_depth_warping, interactive=True)
                        midas_weight = gr.Number(label="MiDaS weight", value=da.midas_weight, interactive=True)
                    with gr.Row(variant='compact'):
                        padding_mode = gr.Radio(['border', 'reflection', 'zeros'], label="Padding mode", value=da.padding_mode, elem_id="padding_mode")
                        sampling_mode = gr.Radio(['bicubic', 'bilinear', 'nearest'], label="Sampling mode", value=da.sampling_mode, elem_id="sampling_mode")
                with gr.Accordion('Field Of View', visible=False, open=False) as fov_accord:
                    with gr.Row(variant='compact'):
                        fov_schedule = gr.Textbox(label="FOV schedule", lines=1, value = da.fov_schedule, interactive=True)
                    with gr.Row():
                        near_schedule = gr.Textbox(label="Near schedule", lines=1, value = da.near_schedule, interactive=True)
                    with gr.Row():
                        far_schedule = gr.Textbox(label="Far schedule", lines=1, value = da.far_schedule, interactive=True)
                # Perspective Flip
                with gr.Accordion('Perspective Flip', open=False) as perspective_flip_accord:
                    with gr.Row():
                        enable_perspective_flip = gr.Checkbox(label="Enable perspective flip", value=da.enable_perspective_flip, interactive=True)
                    with gr.Row():
                        perspective_flip_theta = gr.Textbox(label="Perspective flip theta", lines=1, value = da.perspective_flip_theta, interactive=True)
                    with gr.Row():
                        perspective_flip_phi = gr.Textbox(label="Perspective flip phi", lines=1, value = da.perspective_flip_phi, interactive=True)
                    with gr.Row():
                        perspective_flip_gamma = gr.Textbox(label="Perspective flip gamma", lines=1, value = da.perspective_flip_gamma, interactive=True)
                    with gr.Row():
                        perspective_flip_fv = gr.Textbox(label="Perspective flip fv", lines=1, value = da.perspective_flip_fv, interactive=True)
            # Noise
        
            def change_perlin_visibility(choice):
                return gr.update(visible=choice=="perlin")
            with gr.Tab('Noise', open=True) as a8:
                with gr.Row():
                    noise_type = gr.Radio(['uniform', 'perlin'], label="Noise type", value=da.noise_type, elem_id="noise_type")
                with gr.Row():
                    noise_schedule = gr.Textbox(label="Noise schedule", lines=1, value = da.noise_schedule, interactive=True)
                with gr.Row() as perlin_row:
                    with gr.Column(min_width=220):
                        perlin_octaves = gr.Slider(label="Perlin octaves", minimum=1, maximum=7, value=da.perlin_octaves, step=1, interactive=True)
                    with gr.Column(min_width=220):
                        perlin_persistence = gr.Slider(label="Perlin persistence", minimum=0, maximum=1, value=da.perlin_persistence, step=0.02, interactive=True)
            # Coherence
            with gr.Tab('Coherence', open=False) as coherence_accord:
                with gr.Row(equal_height=True):
                    # Future TODO: remove 'match frame 0' prefix (after we manage the deprecated-names settings import), then convert from Dropdown to Radio!
                    color_coherence = gr.Dropdown(label="Color coherence", choices=['None', 'Match Frame 0 HSV', 'Match Frame 0 LAB', 'Match Frame 0 RGB', 'Video Input'], value=da.color_coherence, type="value", elem_id="color_coherence", interactive=True)
                    with gr.Column() as force_grayscale_column:
                        color_force_grayscale = gr.Checkbox(label="Color force Grayscale", value=da.color_force_grayscale, interactive=True)
                with gr.Row(visible=False) as color_coherence_video_every_N_frames_row:
                    color_coherence_video_every_N_frames = gr.Number(label="Color coherence video every N frames", value=1, interactive=True)
                with gr.Row():
                    contrast_schedule = gr.Textbox(label="Contrast schedule", lines=1, value = da.contrast_schedule, interactive=True)
                with gr.Row():
                    # what to do with blank frames (they may result from glitches or the NSFW filter being turned on): reroll with +1 seed, interrupt the animation generation, or do nothing
                    reroll_blank_frames = gr.Radio(['reroll', 'interrupt', 'ignore'], label="Reroll blank frames", value=d.reroll_blank_frames, elem_id="reroll_blank_frames")
            # Anti-blur
            with gr.Tab('Anti Blur', open=False) as anti_blur_accord:
                with gr.Row(variant='compact'):
                    kernel_schedule = gr.Textbox(label="Kernel schedule", lines=1, value = da.kernel_schedule, interactive=True)
                with gr.Row(variant='compact'):
                    sigma_schedule = gr.Textbox(label="Sigma schedule", lines=1, value = da.sigma_schedule, interactive=True)
                with gr.Row(variant='compact'):
                    amount_schedule = gr.Textbox(label="Amount schedule", lines=1, value = da.amount_schedule, interactive=True)
                with gr.Row(variant='compact'):
                    threshold_schedule = gr.Textbox(label="Threshold schedule", lines=1, value = da.threshold_schedule, interactive=True)
            # 3D FOV
            # Seed Scheduling
            
        # Animation settings END
        # Prompts tab START    
        with gr.Tab('Prompts'):
            with gr.Accordion(label='*Important* notes on Prompts', elem_id='prompts_info_accord', open=False, visible=True) as prompts_info_accord:
                gr.HTML("""
                    <ul style="list-style-type:circle; margin-left:2em; margin-bottom:0.2em">
                    <li>Please always keep values in math functions above 0.</li>
                    <li>There is *no* Batch mode like in vanilla deforum. Please Use the txt2img tab for that.</li>
                    <li>For negative prompts, please write your positive prompt, then --neg ugly, text, assymetric, or any other negative tokens of your choice. OR:</li>
                    <li>Use the negative_prompts field to automatically append all words as a negative prompt. *Don't* add --neg in the negative_prompts field!</li>
                    <li>Prompts are stored in JSON format. If you've got an error, check it in validator, <a style="color:SteelBlue" href="https://odu.github.io/slingjsonlint/">like here</a></li>
                    </ul>
                    """)
            with gr.Row():
                animation_prompts = gr.Textbox(label="Prompts", lines=8, interactive=True, value = DeforumAnimPrompts())
            with gr.Row():
                animation_prompts_positive = gr.Textbox(label="Prompts positive", lines=1, interactive=True, value = "")
            with gr.Row():
                animation_prompts_negative = gr.Textbox(label="Prompts negative", lines=1, interactive=True, value = "")
            # Composable Mask scheduling
            with gr.Accordion('Composable Mask scheduling', open=False):
                gr.HTML("To enable, check use_mask in the Init tab.<br>Supports boolean operations (! - negation, & - and, | - or, ^ - xor, \ - difference, () - nested operations); <br>default variables in \{\}, like \{init_mask\}, \{video_mask\}, \{everywhere\}; <br>masks from files in [], like [mask1.png]; <br>description-based <i>word masks</i> in &lt;&gt;, like &lt;apple&gt;, &lt;hair&gt;")
                with gr.Row():
                    mask_schedule = gr.Textbox(label="Mask schedule", lines=1, value = da.mask_schedule, interactive=True)
                with gr.Row():
                    use_noise_mask = gr.Checkbox(label="Use noise mask", value=da.use_noise_mask, interactive=True)
                with gr.Row():
                    noise_mask_schedule = gr.Textbox(label="Noise mask schedule", lines=1, value = da.noise_mask_schedule, interactive=True)
        # Prompts settings END
        with gr.Tab('Init'):
            # Image Init
            with gr.Accordion('Image Init', open=True):
                with gr.Row():
                    with gr.Column(min_width=150):
                        use_init = gr.Checkbox(label="Use init", value=d.use_init, interactive=True, visible=True)
                    with gr.Column(min_width=150):
                        strength_0_no_init = gr.Checkbox(label="Strength 0 no init", value=True, interactive=True)
                    with gr.Column(min_width=170):
                        strength = gr.Slider(label="Strength", minimum=0, maximum=1, step=0.01, value=0, interactive=True)
                with gr.Row():
                    init_image = gr.Textbox(label="Init image", lines=1, interactive=True, value = d.init_image)
            with gr.Accordion('Mask Init', open=True):
                with gr.Row():
                    use_mask = gr.Checkbox(label="Use mask", value=d.use_mask, interactive=True)
                    use_alpha_as_mask = gr.Checkbox(label="Use alpha as mask", value=d.use_alpha_as_mask, interactive=True)
                    invert_mask = gr.Checkbox(label="Invert mask", value=d.invert_mask, interactive=True)
                    overlay_mask = gr.Checkbox(label="Overlay mask", value=d.overlay_mask, interactive=True)
                with gr.Row():
                    mask_file = gr.Textbox(label="Mask file", lines=1, interactive=True, value = d.mask_file)
                with gr.Row():
                    mask_contrast_adjust = gr.Number(label="Mask contrast adjust", value=d.mask_contrast_adjust, interactive=True)
                    mask_brightness_adjust = gr.Number(label="Mask brightness adjust", value=d.mask_brightness_adjust, interactive=True)
                    mask_overlay_blur = gr.Number(label="Mask overlay blur", value=d.mask_overlay_blur, interactive=True)
                with gr.Row():
                    choice = mask_fill_choices[d.fill]
                    fill = gr.Radio(label='Mask fill', choices=mask_fill_choices, value=choice, type="index")
                with gr.Row():
                    full_res_mask = gr.Checkbox(label="Full res mask", value=d.full_res_mask, interactive=True)
                    full_res_mask_padding = gr.Slider(minimum=0, maximum=512, step=1, label="Full res mask padding", value=d.full_res_mask_padding, interactive=True)
            # Video Init
            with gr.Accordion('Video Init', open=True):
                with gr.Row():
                    video_init_path = gr.Textbox(label="Video init path", lines=1, value = da.video_init_path, interactive=True)
                with gr.Row():
                    extract_from_frame = gr.Number(label="Extract from frame", value=da.extract_from_frame, interactive=True, precision=0)
                    extract_to_frame = gr.Number(label="Extract to frame", value=da.extract_to_frame, interactive=True, precision=0)
                    extract_nth_frame = gr.Number(label="Extract nth frame", value=da.extract_nth_frame, interactive=True, precision=0)
                    overwrite_extracted_frames = gr.Checkbox(label="Overwrite extracted frames", value=False, interactive=True)
                    use_mask_video = gr.Checkbox(label="Use mask video", value=False, interactive=True)
                with gr.Row():
                    video_mask_path = gr.Textbox(label="Video mask path", lines=1, value = da.video_mask_path, interactive=True)
            # Parseq
            with gr.Accordion('Parseq', open=False):
                gr.HTML("""
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
        def show_hybrid_html_msg(choice):
            if choice not in ['2D','3D']:
                return gr.update(visible=True) 
            else:
                return gr.update(visible=False)
        def change_hybrid_tab_status(choice):
            if choice in ['2D','3D']:
                return gr.update(visible=True) 
            else:
                return gr.update(visible=False)
        # HYBRID VIDEO tab
        with gr.Tab('Hybrid Video'):
            # this html only shows when not in 2d/3d mode
            hybrid_msg_html = gr.HTML(value='Please, change animation mode to 2D or 3D to enable Hybrid Mode',visible=False, elem_id='hybrid_msg_html')
            with gr.Accordion("Info & Help", open=False):
                hybrid_html = "<p style=\"padding-bottom:0\"><b style=\"text-shadow: blue -1px -1px;\">Hybrid Video Compositing in 2D/3D Mode</b><span style=\"color:#DDD;font-size:0.7rem;text-shadow: black -1px -1px;margin-left:10px;\">by <a href=\"https://github.com/reallybigname\">reallybigname</a></span></p>"
                hybrid_html += "<ul style=\"list-style-type:circle; margin-left:1em; margin-bottom:1em;\"><li>Composite video with previous frame init image in <b>2D or 3D animation_mode</b> <i>(not for Video Input mode)</i></li>"
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
                hybrid_html += "<a style='color:SteelBlue;' target='_blank' href='https://github.com/deforum-art/deforum-for-automatic1111-webui/wiki/Animation-Settings#hybrid-video-mode-for-2d3d-animations'>Click Here</a> for more info/ a Guide."      
                gr.HTML(hybrid_html)
            with gr.Accordion("Hybrid Settings", open=True) as hybrid_settings_accord:
                with gr.Row(variant='compact'):
                    with gr.Column(min_width=340):
                        with gr.Row(variant='compact'):
                            hybrid_generate_inputframes = gr.Checkbox(label="Generate inputframes", value=False, interactive=True)
                            hybrid_composite = gr.Checkbox(label="Hybrid composite", value=False, interactive=True)
                    with gr.Column(min_width=340):
                        with gr.Row(variant='compact'):
                            hybrid_use_first_frame_as_init_image = gr.Checkbox(label="First frame as init image", value=False, interactive=True)
                            hybrid_motion_use_prev_img = gr.Checkbox(label="Motion use prev img", value=False, interactive=True)
                with gr.Row():
                    with gr.Column(variant='compact'):
                        with gr.Row(variant='compact'):
                            hybrid_motion = gr.Radio(['None', 'Optical Flow', 'Perspective', 'Affine'], label="Hybrid motion", value=da.hybrid_motion, elem_id="hybrid_motion")
                    with gr.Column(variant='compact'):
                        with gr.Row(variant='compact'):
                            with gr.Column(scale=1):
                                hybrid_flow_method = gr.Radio(['DIS Medium', 'Farneback'], label="Flow method", value=da.hybrid_flow_method, elem_id="hybrid_flow_method")
                                hybrid_comp_mask_type = gr.Radio(['None', 'Depth', 'Video Depth', 'Blend', 'Difference'], label="Comp mask type", value=da.hybrid_comp_mask_type, elem_id="hybrid_comp_mask_type")
                with gr.Row(visible=False) as hybrid_comp_mask_row:
                    hybrid_comp_mask_equalize = gr.Radio(['None', 'Before', 'After', 'Both'], label="Comp mask equalize", value=da.hybrid_comp_mask_equalize, elem_id="hybrid_comp_mask_equalize")
                    with gr.Column(variant='compact'):
                        hybrid_comp_mask_auto_contrast = gr.Checkbox(label="Comp mask auto contrast", value=False, interactive=True)
                        hybrid_comp_mask_inverse = gr.Checkbox(label="Comp mask inverse", value=False, interactive=True)
                with gr.Row(variant='compact'):
                        hybrid_comp_save_extra_frames = gr.Checkbox(label="Comp save extra frames", value=False, interactive=True)
            with gr.Accordion("Hybrid Schedules", open=False) as hybrid_sch_accord:
                with gr.Row(variant='compact'):
                    hybrid_comp_alpha_schedule = gr.Textbox(label="Comp alpha schedule", lines=1, value = da.hybrid_comp_alpha_schedule, interactive=True)
                with gr.Row(variant='compact'):
                    hybrid_comp_mask_blend_alpha_schedule = gr.Textbox(label="Comp mask blend alpha schedule", lines=1, value = da.hybrid_comp_mask_blend_alpha_schedule, interactive=True, elem_id="hybridelemtest")
                with gr.Row(variant='compact'):
                    hybrid_comp_mask_contrast_schedule = gr.Textbox(label="Comp mask contrast schedule", lines=1, value = da.hybrid_comp_mask_contrast_schedule, interactive=True)
                with gr.Row(variant='compact'):
                    hybrid_comp_mask_auto_contrast_cutoff_high_schedule = gr.Textbox(label="Comp mask auto contrast cutoff high schedule", lines=1, value = da.hybrid_comp_mask_auto_contrast_cutoff_high_schedule, interactive=True)
                with gr.Row(variant='compact'):
                    hybrid_comp_mask_auto_contrast_cutoff_low_schedule = gr.Textbox(label="Comp mask auto contrast cutoff low schedule", lines=1, value = da.hybrid_comp_mask_auto_contrast_cutoff_low_schedule, interactive=True)
            with gr.Accordion("Humans Masking", open=False) as humans_masking_accord:
                with gr.Row(variant='compact'):
                    hybrid_generate_human_masks = gr.Radio(['None', 'PNGs', 'Video', 'Both'], label="Generate human masks", value=da.hybrid_generate_human_masks, elem_id="hybrid_generate_human_masks")
        # VIDEO OUTPUT TAB
        with gr.Tab('Output'):
            with gr.Accordion('Video Output Settings', open=True):
                with gr.Row(variant='compact') as fps_out_format_row:
                    fps = gr.Slider(label="FPS", value=dv.fps, minimum=1, maximum=240, step=1)
                    output_format = gr.Dropdown(label="Output format", choices=['PIL gif', 'FFMPEG mp4'], value='FFMPEG mp4', type="value", elem_id="output_format", interactive=True)
                # with gr.Column(variant='compact'):
                with gr.Row(equal_height=True, variant='compact', visible=True) as ffmpeg_set_row:
                    # with gr.Row(variant='compact', equal_height=True) as ffmpeg_set_row:
                    ffmpeg_crf = gr.Slider(minimum=0, maximum=51, step=1, label="CRF", value=dv.ffmpeg_crf, interactive=True)
                    ffmpeg_preset = gr.Dropdown(label="Preset", choices=['veryslow', 'slower', 'slow', 'medium', 'fast', 'faster', 'veryfast', 'superfast', 'ultrafast'], interactive=True, value = dv.ffmpeg_preset, type="value")
                    ffmpeg_location = gr.Textbox(label="Location", lines=1, interactive=True, value = dv.ffmpeg_location)
                with gr.Column(variant='compact'):
                    with gr.Row(variant='compact') as soundtrack_row:
                        add_soundtrack = gr.Radio(['None', 'File', 'Init Video'], label="Add soundtrack", value=dv.add_soundtrack)
                        soundtrack_path = gr.Textbox(label="Soundtrack path", lines=1, interactive=True, value = dv.soundtrack_path)
                    with gr.Row(variant='compact'):
                        skip_video_for_run_all = gr.Checkbox(label="Skip video for run all", value=dv.skip_video_for_run_all, interactive=True)
                        store_frames_in_ram = gr.Checkbox(label="Store frames in ram", value=dv.store_frames_in_ram, interactive=True)
                        save_depth_maps = gr.Checkbox(label="Save depth maps", value=da.save_depth_maps, interactive=True)
                
                with gr.Accordion('Stitch Frames to Video', open=False, visible=True) as stitch_imgs_to_vid_row:
                    with gr.Row(visible=False):
                        # max_video_frames = gr.Number(label="max_video_frames", value=200, interactive=True)
                        path_name_modifier = gr.Dropdown(label="Path name modifier", choices=['x0_pred', 'x'], value=dv.path_name_modifier, type="value", elem_id="path_name_modifier", interactive=True, visible=False) 
                    gr.HTML("""
                     <p style="margin-top:0em">
                        Important Notes:
                        <ul style="list-style-type:circle; margin-left:1em; margin-bottom:0.25em">
                            <li>Enter relative to webui folder or Full-Absolute path, and make sure it ends with something like this: '20230124234916_%05d.png', just replace 20230124234916 with your batch ID</li>
                            <li>Working FFMPEG under 'ffmpeg_location' is required to stitch a video in this mode!</li>
                        </ul>
                        """)
                    with gr.Row(variant='compact'):
                          image_path = gr.Textbox(label="Image path", lines=1, interactive=True, value = dv.image_path)
                    with gr.Row(visible=False):
                        mp4_path = gr.Textbox(label="MP4 path", lines=1, interactive=True, value = dv.mp4_path)
                    # not visible as of 06-02-23 since render_steps is disabled as well and they work together. Need to fix both.
                    with gr.Row(visible=False):
                        # rend_step Never worked - set to visible false 28-1-23 # MOVE OUT FROM HERE!
                        render_steps = gr.Checkbox(label="Render steps", value=dv.render_steps, interactive=True, visible=False)
                    ffmpeg_stitch_imgs_but = gr.Button(value="*Stitch frames to video*")
                    ffmpeg_stitch_imgs_but.click(direct_stitch_vid_from_frames,inputs=[image_path, fps, ffmpeg_location, ffmpeg_crf, ffmpeg_preset, add_soundtrack, soundtrack_path])
            with gr.Accordion('Frame Interpolation (RIFE)', open=True) as rife_accord:
                with gr.Accordion('Important notes and Help', open=False):
                    gr.HTML("""
                    Use <a href="https://github.com/megvii-research/ECCV2022-RIFE">RIFE</a> Frame Interpolation to smooth out, slow-mo (or both) any video.</p>
                     <p style="margin-top:1em">
                        Supported engines:
                        <ul style="list-style-type:circle; margin-left:1em; margin-bottom:1em">
                            <li>RIFE v4.6, v4.3 and v4.0. Recommended for now: v4.6.</li>
                        </ul>
                    </p>
                     <p style="margin-top:1em">
                        Important notes:
                        <ul style="list-style-type:circle; margin-left:1em; margin-bottom:1em">
                            <li>Working FFMPEG is required to get an output interpolated video. No ffmepg will leave you with just the interpolated imgs.</li>
                            <li>Frame Interpolation will *not* run if 'store_frames_in_ram' is enabled.</li>
                            <li>Audio (if provided) will *not* be transferred to the interpolated video if Slow-Mo is enabled.</li>
                            <li>'add_soundtrack' and 'soundtrack_path' aren't being honoured in "Interpolate an existing video" mode. Original vid audio will be used instead with the same slow-mo rules above.</li>
                            <li>Frame Interpolation will always save an .mp4 video even if you used GIF for the raw video.</li>
                        </ul>
                    </p>
                    """)
                with gr.Column():
                    with gr.Row(variant='compact'):
                        # Interpolation Engine
                        frame_interpolation_engine = gr.Dropdown(label="Engine", choices=['RIFE v4.0','RIFE v4.3','RIFE v4.6'], value=dv.frame_interpolation_engine, type="value", elem_id="frame_interpolation_engine", interactive=True)
                        # How many times to interpolate (interp x)
                        frame_interpolation_x_amount = gr.Dropdown(label="Interp x", choices=['Disabled','x2','x3','x4','x5','x6','x7','x8','x9','x10'], value=dv.frame_interpolation_x_amount, type="value", elem_id="frame_interpolation_x_amount", interactive=True)
                        # Interp Slow-Mo (setting final output fps, not really doing anything direclty with RIFE)
                        frame_interpolation_slow_mo_amount = gr.Dropdown(label="Slow-Mo x", choices=['Disabled','x2','x4','x8'], value=dv.frame_interpolation_slow_mo_amount, type="value", elem_id="frame_interpolation_slow_mo_amount", interactive=True)
                        # If this is set to True, we keep all of the interpolated frames in a folder. Default is False - means we delete them at the end of the run
                        frame_interpolation_keep_imgs = gr.Checkbox(label="Keep Imgs", elem_id="frame_interpolation_keep_imgs", value=dv.frame_interpolation_keep_imgs, interactive=True)
                    with gr.Row():
                        # Intrpolate any existing video from the connected PC
                        with gr.Accordion('Interpolate an existing video', open=False):
                            # A drag-n-drop UI box to which the user uploads a *single* (at this stage) video
                            vid_to_rife_chosen_file = gr.File(label="Video to interpolate", interactive=True, file_count="single", file_types=["video"], elem_id="vid_to_rife_chosen_file")
                            with gr.Row(variant='compact'):
                                # Non interactive textbox showing uploaded input vid total Frame Count
                                in_vid_frame_count_window = gr.Textbox(label="In Frame Count", lines=1, interactive=False, value='---')
                                # Non interactive textbox showing uploaded input vid FPS
                                in_vid_fps_ui_window = gr.Textbox(label="In FPS", lines=1, interactive=False, value='---')
                                # Non interactive textbox showing expected output interpolated video FPS
                                out_interp_vid_estimated_fps = gr.Textbox(label="Interpolated Vid FPS", value='---')
                            # This is the actual button that's pressed to initiate the interpolation:
                            rife_btn = gr.Button(value="*Interpolate uploaded video*")
                            # Show a text about CLI outputs:
                            gr.HTML("* check your CLI for outputs")
                            # make the functin call when the RIFE button is clicked
                            rife_btn.click(upload_vid_to_rife,inputs=[vid_to_rife_chosen_file, frame_interpolation_engine, frame_interpolation_x_amount, frame_interpolation_slow_mo_amount, frame_interpolation_keep_imgs, ffmpeg_location, ffmpeg_crf, ffmpeg_preset, in_vid_fps_ui_window])
                            # update output fps field upon changing of interp_x and/ or slow_mo_x
                            frame_interpolation_x_amount.change(set_interp_out_fps, inputs=[frame_interpolation_x_amount, frame_interpolation_slow_mo_amount, in_vid_fps_ui_window], outputs=out_interp_vid_estimated_fps)
                            frame_interpolation_slow_mo_amount.change(set_interp_out_fps, inputs=[frame_interpolation_x_amount, frame_interpolation_slow_mo_amount, in_vid_fps_ui_window], outputs=out_interp_vid_estimated_fps)
                            # Populate the above FPS and FCount values as soon as a video is uploaded to the FileUploadBox (vid_to_rife_chosen_file)
                            vid_to_rife_chosen_file.change(gradio_f_interp_get_fps_and_fcount,inputs=[vid_to_rife_chosen_file, frame_interpolation_x_amount, frame_interpolation_slow_mo_amount],outputs=[in_vid_fps_ui_window,in_vid_frame_count_window, out_interp_vid_estimated_fps])
            output_format.change(fn=hide_by_gif, inputs=output_format, outputs=ffmpeg_set_row)
            output_format.change(fn=hide_by_gif, inputs=output_format, outputs=soundtrack_row)
            output_format.change(fn=hide_by_gif, inputs=output_format, outputs=stitch_imgs_to_vid_row)
            output_format.change(fn=hide_by_gif, inputs=output_format, outputs=rife_accord)
            # Old/ Non actives accordion
            with gr.Accordion(visible=False, label='INVISIBLE') as not_in_use_accordion:
                from_img2img_instead_of_link = gr.Checkbox(label="from_img2img_instead_of_link", value=False, interactive=False, visible=False)
                # INVISIBLE AS OF 08-02 (with static value of 8 for both W and H). Was in Perlin section before Perlin Octaves/Persistence
                with gr.Column(min_width=200, visible=False):
                    perlin_w = gr.Slider(label="Perlin W", minimum=0.1, maximum=16, step=0.1, value=da.perlin_w, interactive=True)
                    perlin_h = gr.Slider(label="Perlin H", minimum=0.1, maximum=16, step=0.1, value=da.perlin_h, interactive=True)
            
    # Gradio's Change functions - hiding and renaming elements based on other elements
    animation_mode.change(fn=change_max_frames_visibility, inputs=animation_mode, outputs=max_frames)
    animation_mode.change(fn=change_diffusion_cadence_visibility, inputs=animation_mode, outputs=diffusion_cadence_row)
    animation_mode.change(fn=disable_when_not_in_2d_or_3d_modes, inputs=animation_mode, outputs=anti_blur_accord)
    animation_mode.change(fn=disble_3d_related_stuff, inputs=animation_mode, outputs=depth_3d_warping_accord)
    animation_mode.change(fn=disble_3d_related_stuff, inputs=animation_mode, outputs=fov_accord)
    animation_mode.change(fn=disble_3d_related_stuff, inputs=animation_mode, outputs=only_3d_motion_column)
    animation_mode.change(fn=enable_2d_related_stuff, inputs=animation_mode, outputs=only_2d_motion_column) 
    animation_mode.change(fn=update_motion_accord_name, inputs=animation_mode, outputs=motion_accord) 
    animation_mode.change(fn=disable_by_interpolation, inputs=animation_mode, outputs=force_grayscale_column)
    animation_mode.change(fn=disable_motion_accord, inputs=animation_mode, outputs=motion_accord) 
    animation_mode.change(fn=disable_motion_accord, inputs=animation_mode, outputs=perspective_flip_accord)    
    #Hybrid related:
    animation_mode.change(fn=show_hybrid_html_msg, inputs=animation_mode, outputs=hybrid_msg_html)
    animation_mode.change(fn=change_hybrid_tab_status, inputs=animation_mode, outputs=hybrid_sch_accord)
    animation_mode.change(fn=change_hybrid_tab_status, inputs=animation_mode, outputs=hybrid_settings_accord)
    animation_mode.change(fn=change_hybrid_tab_status, inputs=animation_mode, outputs=humans_masking_accord)
    # End of hybrid related
    seed_behavior.change(fn=change_seed_iter_visibility, inputs=seed_behavior, outputs=seed_iter_N_row) 
    seed_behavior.change(fn=change_seed_schedule_visibility, inputs=seed_behavior, outputs=seed_schedule_row)
    color_coherence.change(fn=change_color_coherence_video_every_N_frames_visibility, inputs=color_coherence, outputs=color_coherence_video_every_N_frames_row)
    noise_type.change(fn=change_perlin_visibility, inputs=noise_type, outputs=perlin_row)
    hybrid_comp_mask_type.change(fn=change_comp_mask_x_visibility, inputs=hybrid_comp_mask_type, outputs=hybrid_comp_mask_row)
    outputs = [fps_out_format_row, soundtrack_row, ffmpeg_set_row, store_frames_in_ram]
    
    
    for output in outputs:
        skip_video_for_run_all.change(fn=change_visibility_from_skip_video, inputs=skip_video_for_run_all, outputs=output)  
        
    # END OF UI TABS
    return locals()

### SETTINGS STORAGE UPDATE! 2023-01-27
### To Reduce The Number Of Settings Overrides,
### They Are Being Passed As Dictionaries
### It Would Have Been Also Nice To Retrieve Them
### From Functions Like Deforumoutputargs(),
### But Over Time There Was Some Cross-Polination,
### So They Are Now Hardcoded As 'List'-Strings Below
### If you're adding a new setting, add it to one of the lists
### besides writing it in the setup functions above

anim_args_names =   str(r'''animation_mode, max_frames, border,
                        angle, zoom, translation_x, translation_y, translation_z,
                        rotation_3d_x, rotation_3d_y, rotation_3d_z,
                        enable_perspective_flip,
                        perspective_flip_theta, perspective_flip_phi, perspective_flip_gamma, perspective_flip_fv,
                        noise_schedule, strength_schedule, contrast_schedule, cfg_scale_schedule,
                        enable_steps_scheduling, steps_schedule,
                        fov_schedule, near_schedule, far_schedule,
                        seed_schedule,
                        enable_sampler_scheduling, sampler_schedule,
                        mask_schedule, use_noise_mask, noise_mask_schedule,
                        enable_checkpoint_scheduling, checkpoint_schedule,
                        enable_clipskip_scheduling, clipskip_schedule,
                        kernel_schedule, sigma_schedule, amount_schedule, threshold_schedule,
                        color_coherence, color_coherence_video_every_N_frames, color_force_grayscale,
                        diffusion_cadence,
                        noise_type, perlin_w, perlin_h, perlin_octaves, perlin_persistence,
                        use_depth_warping, midas_weight,
                        padding_mode, sampling_mode, save_depth_maps,
                        video_init_path, extract_nth_frame, extract_from_frame, extract_to_frame, overwrite_extracted_frames,
                        use_mask_video, video_mask_path,
                        resume_from_timestring, resume_timestring'''
                    ).replace("\n", "").replace("\r", "").replace(" ", "").split(',')
hybrid_args_names =   str(r'''hybrid_generate_inputframes, hybrid_generate_human_masks, hybrid_use_first_frame_as_init_image,
                        hybrid_motion, hybrid_motion_use_prev_img, hybrid_flow_method, hybrid_composite, hybrid_comp_mask_type, hybrid_comp_mask_inverse,
                        hybrid_comp_mask_equalize, hybrid_comp_mask_auto_contrast, hybrid_comp_save_extra_frames,
                        hybrid_comp_alpha_schedule, hybrid_comp_mask_blend_alpha_schedule, hybrid_comp_mask_contrast_schedule,
                        hybrid_comp_mask_auto_contrast_cutoff_high_schedule, hybrid_comp_mask_auto_contrast_cutoff_low_schedule'''
                    ).replace("\n", "").replace("\r", "").replace(" ", "").split(',')
args_names =    str(r'''W, H, tiling, restore_faces,
                        seed, sampler,
                        seed_enable_extras, subseed, subseed_strength, seed_resize_from_w, seed_resize_from_h,
                        steps, ddim_eta,
                        n_batch,
                        save_settings, save_samples, display_samples,
                        save_sample_per_step, show_sample_per_step, 
                        batch_name, filename_format,
                        seed_behavior, seed_iter_N,
                        use_init, from_img2img_instead_of_link, strength_0_no_init, strength, init_image,
                        use_mask, use_alpha_as_mask, invert_mask, overlay_mask,
                        mask_file, mask_contrast_adjust, mask_brightness_adjust, mask_overlay_blur,
                        fill, full_res_mask, full_res_mask_padding,
                        reroll_blank_frames'''
                    ).replace("\n", "").replace("\r", "").replace(" ", "").split(',')
video_args_names =  str(r'''skip_video_for_run_all,
                            fps, output_format, ffmpeg_location, ffmpeg_crf, ffmpeg_preset,
                            add_soundtrack, soundtrack_path,
                            render_steps,
                            path_name_modifier, image_path, mp4_path, store_frames_in_ram,
                            frame_interpolation_engine, frame_interpolation_x_amount, frame_interpolation_slow_mo_amount,
                            frame_interpolation_keep_imgs'''
                    ).replace("\n", "").replace("\r", "").replace(" ", "").split(',')
parseq_args_names = str(r'''parseq_manifest, parseq_use_deltas'''
                    ).replace("\n", "").replace("\r", "").replace(" ", "").split(',')
loop_args_names = str(r'''use_looper, init_images, image_strength_schedule, blendFactorMax, blendFactorSlope, 
                          tweening_frames_schedule, color_correction_factor'''
                    ).replace("\n", "").replace("\r", "").replace(" ", "").split(',')

component_names =   ['override_settings_with_file', 'custom_settings_file'] + anim_args_names +['animation_prompts', 'animation_prompts_positive', 'animation_prompts_negative'] + args_names + video_args_names + parseq_args_names + hybrid_args_names + loop_args_names
settings_component_names = [name for name in component_names if name not in video_args_names]

def setup_deforum_setting_ui(self, is_img2img, is_extension = True):
    ds = setup_deforum_setting_dictionary(self, is_img2img, is_extension)
    return [ds[name] for name in (['btn'] + component_names)]

def pack_anim_args(args_dict):
    return {name: args_dict[name] for name in (anim_args_names + hybrid_args_names)}

def pack_args(args_dict):
    args_dict = {name: args_dict[name] for name in args_names}
    args_dict['precision'] = 'autocast' 
    args_dict['scale'] = 7
    args_dict['C'] = 4
    args_dict['f'] = 8
    args_dict['timestring'] = ""
    args_dict['init_latent'] = None
    args_dict['init_sample'] = None
    args_dict['init_c'] = None
    args_dict['noise_mask'] = None
    args_dict['seed_internal'] = 0
    return args_dict
    
def pack_video_args(args_dict):
    return {name: args_dict[name] for name in video_args_names}

def pack_parseq_args(args_dict):
    return {name: args_dict[name] for name in parseq_args_names}
    
def pack_loop_args(args_dict):
    return {name: args_dict[name] for name in loop_args_names}

def process_args(args_dict_main):
    override_settings_with_file = args_dict_main['override_settings_with_file']
    custom_settings_file = args_dict_main['custom_settings_file']
    args_dict = pack_args(args_dict_main)
    anim_args_dict = pack_anim_args(args_dict_main)
    video_args_dict = pack_video_args(args_dict_main)
    parseq_args_dict = pack_parseq_args(args_dict_main)
    loop_args_dict = pack_loop_args(args_dict_main)

    import json
    
    root = SimpleNamespace(**Root())
    root.p = args_dict_main['p']
    p = root.p
    root.animation_prompts = json.loads(args_dict_main['animation_prompts'])
    positive_prompts = args_dict_main['animation_prompts_positive']
    negative_prompts = args_dict_main['animation_prompts_negative']
    # remove --neg from negative_prompts if recieved by mistake
    negative_prompts = negative_prompts.replace('--neg', '')
    for key in root.animation_prompts:
        animationPromptCurr = root.animation_prompts[key]
        root.animation_prompts[key] = f"{positive_prompts} {animationPromptCurr} {'' if '--neg' in animationPromptCurr else '--neg'} {negative_prompts}"
    from deforum_helpers.settings import load_args
    
    if override_settings_with_file:
        load_args(args_dict, anim_args_dict, parseq_args_dict, loop_args_dict, custom_settings_file, root)
    
    print(f"Additional models path: {root.models_path}")
    if not os.path.exists(root.models_path):
        os.mkdir(root.models_path)

    args = SimpleNamespace(**args_dict)
    anim_args = SimpleNamespace(**anim_args_dict)
    video_args = SimpleNamespace(**video_args_dict)
    parseq_args = SimpleNamespace(**parseq_args_dict)
    loop_args = SimpleNamespace(**loop_args_dict)

    p.width, p.height = map(lambda x: x - x % 64, (args.W, args.H))
    p.steps = args.steps
    p.seed = args.seed
    p.sampler_name = args.sampler
    p.batch_size = args.n_batch
    p.tiling = args.tiling
    p.restore_faces = args.restore_faces
    p.seed_enable_extras = args.seed_enable_extras
    p.subseed = args.subseed
    p.subseed_strength = args.subseed_strength
    p.seed_resize_from_w = args.seed_resize_from_w
    p.seed_resize_from_h = args.seed_resize_from_h
    p.fill = args.fill
    p.ddim_eta = args.ddim_eta

    # TODO: Handle batch name dynamically?
    current_arg_list = [args, anim_args, video_args, parseq_args]
    args.outdir = os.path.join(p.outpath_samples, args.batch_name)
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
    
    return root, args, anim_args, video_args, parseq_args, loop_args

def print_args(args):
    print("ARGS: /n")
    for key, value in args.__dict__.items():
        print(f"{key}: {value}")
 
# Local gradio-to-rife function. *Needs* to stay here since we do Root() and use gradio elements directly, to be changed in the future
def upload_vid_to_rife(file, engine, x_am, sl_am, keep_imgs, f_location, f_crf, f_preset, in_vid_fps):
    # print msg and do nothing if vid not uploaded or interp_x not provided
    if not file or x_am == 'Disabled':
        return print("Please upload a video and set a proper value for 'Interp x'. Can't interpolate x0 times :)")

    root_params = Root()
    f_models_path = root_params['models_path']

    process_rife_vid_upload_logic(file, engine, x_am, sl_am, keep_imgs, f_location, f_crf, f_preset, in_vid_fps, f_models_path, file.orig_name)
