from types import SimpleNamespace
from .defaults import *  # TODO: change this to not *
from .deforum_controlnet import setup_controlnet_ui
from .gradio_funcs import *
from .frame_interpolation import gradio_f_interp_get_fps_and_fcount
from .args import DeforumArgs, DeforumAnimArgs, ParseqArgs, DeforumOutputArgs, RootArgs, LoopArgs
from .ui_elements import get_tab_prompts, get_tab_hybrid, get_tab_output

def set_arg_lists():
    d = SimpleNamespace(**DeforumArgs())  # default args
    da = SimpleNamespace(**DeforumAnimArgs())  # default anim args
    dp = SimpleNamespace(**ParseqArgs())  # default parseq ars
    dv = SimpleNamespace(**DeforumOutputArgs())  # default video args
    dr = SimpleNamespace(**RootArgs())  # ROOT args
    dloopArgs = SimpleNamespace(**LoopArgs())  # Guided imgs args
    return d, da, dp, dv, dr, dloopArgs


def get_tab_run(d, da):
    with gr.TabItem('Run'):  # RUN TAB
        from modules.sd_samplers import samplers_for_img2img
        with gr.Row(variant='compact'):
            sampler = gr.Dropdown(label="Sampler", choices=[x.name for x in samplers_for_img2img], value=samplers_for_img2img[0].name, type="value", elem_id="sampler", interactive=True)
            steps = gr.Slider(label="Steps", minimum=0, maximum=200, step=1, value=d.steps, interactive=True)
        with gr.Row(variant='compact'):
            W = gr.Slider(label="Width", minimum=64, maximum=2048, step=64, value=d.W, interactive=True)
            H = gr.Slider(label="Height", minimum=64, maximum=2048, step=64, value=d.H, interactive=True)
        with gr.Row(variant='compact'):
            seed = gr.Number(label="Seed", value=d.seed, interactive=True, precision=0, info="Starting seed for the animation. -1 for random")
            batch_name = gr.Textbox(label="Batch name", lines=1, interactive=True, value=d.batch_name,
                                    info="output images will be placed in a folder with this name ({timestring} token will be replaced) inside the img2img output folder. Supports params placeholders. e.g {seed}, {w}, {h}, {prompts}")
        with gr.Row(variant='compact'):
            # Seed enable extras is INVISIBLE in the ui!
            seed_enable_extras = gr.Checkbox(label="Enable subseed controls", value=False, visible=False)
            restore_faces = gr.Checkbox(label='Restore Faces', value=d.restore_faces)
            tiling = gr.Checkbox(label='Tiling', value=d.tiling)
            enable_ddim_eta_scheduling = gr.Checkbox(label='Enable DDIM ETA scheduling', value=da.enable_ddim_eta_scheduling, visible=False)
            enable_ancestral_eta_scheduling = gr.Checkbox(label='Enable Ancestral ETA scheduling', value=da.enable_ancestral_eta_scheduling)
        with gr.Row(variant='compact') as eta_sch_row:
            ddim_eta_schedule = gr.Textbox(label="DDIM ETA Schedule", lines=1, value=da.ddim_eta_schedule, interactive=True, visible=False)
            ancestral_eta_schedule = gr.Textbox(label="Ancestral ETA Schedule", lines=1, value=da.ancestral_eta_schedule, interactive=True, visible=False)
        # RUN FROM SETTING FILE ACCORD
        with gr.Accordion('Batch Mode, Resume and more', open=False):
            with gr.Tab('Batch Mode/ run from setting files'):
                with gr.Row(variant='compact'):
                    override_settings_with_file = gr.Checkbox(label="Enable batch mode", value=False, interactive=True, elem_id='override_settings',
                                                              info="run from a list of setting .txt files. Upload them to the box on the right (visible when enabled)")
                    custom_settings_file = gr.File(label="Setting files", interactive=True, file_count="multiple", file_types=[".txt"], elem_id="custom_setting_file", visible=False)
            # RESUME ANIMATION ACCORD
            with gr.Tab('Resume Animation'):
                with gr.Row(variant='compact'):
                    resume_from_timestring = gr.Checkbox(label="Resume from timestring", value=da.resume_from_timestring, interactive=True)
                    resume_timestring = gr.Textbox(label="Resume timestring", lines=1, value=da.resume_timestring, interactive=True)
            with gr.Row(variant='compact') as pix2pix_img_cfg_scale_row:
                pix2pix_img_cfg_scale_schedule = gr.Textbox(label="Pix2Pix img CFG schedule", value=da.pix2pix_img_cfg_scale_schedule, interactive=True,
                                                            info="ONLY in use when working with a P2P ckpt!")
    return {k: v for k, v in {**locals(), **vars()}.items()}

def get_tab_keyframes(d, da, dloopArgs):
    with gr.TabItem('Keyframes'):  # TODO make a some sort of the original dictionary parsing
        with gr.Row(variant='compact'):
            with gr.Column(scale=2):
                animation_mode = gr.Radio(['2D', '3D', 'Interpolation', 'Video Input'], label="Animation mode", value=da.animation_mode, elem_id="animation_mode",
                                          info="control animation mode, will hide non relevant params upon change")
            with gr.Column(scale=1, min_width=180):
                border = gr.Radio(['replicate', 'wrap'], label="Border", value=da.border, elem_id="border",
                                  info="controls pixel generation method for images smaller than the frame. hover on the options to see more info")
        with gr.Row(variant='compact'):
            diffusion_cadence = gr.Slider(label="Cadence", minimum=1, maximum=50, step=1, value=da.diffusion_cadence, interactive=True,
                                          info="# of in-between frames that will not be directly diffused")
            max_frames = gr.Number(label="Max frames", lines=1, value=da.max_frames, interactive=True, precision=0, info="end the animation at this frame number")
        # GUIDED IMAGES ACCORD
        with gr.Accordion('Guided Images', open=False, elem_id='guided_images_accord') as guided_images_accord:
            # GUIDED IMAGES INFO ACCORD
            with gr.Accordion('*READ ME before you use this mode!*', open=False):
                gr.HTML(value=get_gradio_html('guided_imgs'))
            with gr.Row(variant='compact'):
                use_looper = gr.Checkbox(label="Enable guided images mode", value=dloopArgs.use_looper, interactive=True)
            with gr.Row(variant='compact'):
                init_images = gr.Textbox(label="Images to use for keyframe guidance", lines=9, value=get_guided_imgs_default_json(), interactive=True)
            # GUIDED IMAGES SCHEDULES ACCORD
            with gr.Accordion('Guided images schedules', open=False):
                with gr.Row(variant='compact'):
                    image_strength_schedule = gr.Textbox(label="Image strength schedule", lines=1, value=dloopArgs.image_strength_schedule, interactive=True)
                with gr.Row(variant='compact'):
                    blendFactorMax = gr.Textbox(label="Blend factor max", lines=1, value=dloopArgs.blendFactorMax, interactive=True)
                with gr.Row(variant='compact'):
                    blendFactorSlope = gr.Textbox(label="Blend factor slope", lines=1, value=dloopArgs.blendFactorSlope, interactive=True)
                with gr.Row(variant='compact'):
                    tweening_frames_schedule = gr.Textbox(label="Tweening frames schedule", lines=1, value=dloopArgs.tweening_frames_schedule, interactive=True)
                with gr.Row(variant='compact'):
                    color_correction_factor = gr.Textbox(label="Color correction factor", lines=1, value=dloopArgs.color_correction_factor, interactive=True)
        # EXTRA SCHEDULES TABS
        with gr.Tabs(elem_id='extra_schedules'):
            with gr.TabItem('Strength'):
                with gr.Row(variant='compact'):
                    strength_schedule = gr.Textbox(label="Strength schedule", lines=1, value=da.strength_schedule, interactive=True,
                                                   info="amount of presence of previous frame to influence next frame, also controls steps in the following formula [steps - (strength_schedule * steps)]")
            with gr.TabItem('CFG'):
                with gr.Row(variant='compact'):
                    cfg_scale_schedule = gr.Textbox(label="CFG scale schedule", lines=1, value=da.cfg_scale_schedule, interactive=True,
                                                    info="how closely the image should conform to the prompt. Lower values produce more creative results. (recommended range 5-15)")
                with gr.Row(variant='compact'):
                    enable_clipskip_scheduling = gr.Checkbox(label="Enable CLIP skip scheduling", value=da.enable_clipskip_scheduling, interactive=True)
                with gr.Row(variant='compact'):
                    clipskip_schedule = gr.Textbox(label="CLIP skip schedule", lines=1, value=da.clipskip_schedule, interactive=True)
            with gr.TabItem('Seed') as a3:
                with gr.Row(variant='compact'):
                    seed_behavior = gr.Radio(['iter', 'fixed', 'random', 'ladder', 'alternate', 'schedule'], label="Seed behavior", value=d.seed_behavior, elem_id="seed_behavior",
                                             info="controls the seed behavior that is used for animation. hover on the options to see more info")
                with gr.Row(variant='compact') as seed_iter_N_row:
                    seed_iter_N = gr.Number(label="Seed iter N", value=d.seed_iter_N, interactive=True, precision=0,
                                            info="for how many frames the same seed should stick before iterating to the next one")
                with gr.Row(visible=False) as seed_schedule_row:
                    seed_schedule = gr.Textbox(label="Seed schedule", lines=1, value=da.seed_schedule, interactive=True)
            with gr.TabItem('SubSeed', open=False) as subseed_sch_tab:
                with gr.Row(variant='compact'):
                    enable_subseed_scheduling = gr.Checkbox(label="Enable Subseed scheduling", value=da.enable_subseed_scheduling, interactive=True)
                    subseed_schedule = gr.Textbox(label="Subseed schedule", lines=1, value=da.subseed_schedule, interactive=True)
                    subseed_strength_schedule = gr.Textbox(label="Subseed strength schedule", lines=1, value=da.subseed_strength_schedule, interactive=True)
                with gr.Row(variant='compact'):
                    seed_resize_from_w = gr.Slider(minimum=0, maximum=2048, step=64, label="Resize seed from width", value=0)
                    seed_resize_from_h = gr.Slider(minimum=0, maximum=2048, step=64, label="Resize seed from height", value=0)
            # Steps Scheduling
            with gr.TabItem('Step') as a13:
                with gr.Row(variant='compact'):
                    enable_steps_scheduling = gr.Checkbox(label="Enable steps scheduling", value=da.enable_steps_scheduling, interactive=True)
                with gr.Row(variant='compact'):
                    steps_schedule = gr.Textbox(label="Steps schedule", lines=1, value=da.steps_schedule, interactive=True,
                                                info="mainly allows using more than 200 steps. otherwise, it's a mirror-like param of 'strength schedule'")
            # Sampler Scheduling
            with gr.TabItem('Sampler') as a14:
                with gr.Row(variant='compact'):
                    enable_sampler_scheduling = gr.Checkbox(label="Enable sampler scheduling", value=da.enable_sampler_scheduling, interactive=True)
                with gr.Row(variant='compact'):
                    sampler_schedule = gr.Textbox(label="Sampler schedule", lines=1, value=da.sampler_schedule, interactive=True,
                                                  info="allows keyframing different samplers. Use names as they appear in ui dropdown in 'run' tab")
            # Checkpoint Scheduling
            with gr.TabItem('Checkpoint') as a15:
                with gr.Row(variant='compact'):
                    enable_checkpoint_scheduling = gr.Checkbox(label="Enable checkpoint scheduling", value=da.enable_checkpoint_scheduling, interactive=True)
                with gr.Row(variant='compact'):
                    checkpoint_schedule = gr.Textbox(label="Checkpoint schedule", lines=1, value=da.checkpoint_schedule, interactive=True,
                                                     info="allows keyframing different sd models. use *full* name as appears in ui dropdown")
        # MOTION INNER TAB
        with gr.Tabs(elem_id='motion_noise_etc'):
            with gr.TabItem('Motion') as motion_tab:
                with gr.Column(visible=True) as only_2d_motion_column:
                    with gr.Row(variant='compact'):
                        zoom = gr.Textbox(label="Zoom", lines=1, value=da.zoom, interactive=True, info="scale the canvas size, multiplicatively. [static = 1.0]")
                    with gr.Row(variant='compact'):
                        angle = gr.Textbox(label="Angle", lines=1, value=da.angle, interactive=True, info="rotate canvas clockwise/anticlockwise in degrees per frame")
                    with gr.Row(variant='compact'):
                        transform_center_x = gr.Textbox(label="Transform Center X", lines=1, value=da.transform_center_x, interactive=True, info="x center axis for 2D angle/zoom")
                    with gr.Row(variant='compact'):
                        transform_center_y = gr.Textbox(label="Transform Center Y", lines=1, value=da.transform_center_y, interactive=True, info="y center axis for 2D angle/zoom")
                with gr.Column(visible=True) as both_anim_mode_motion_params_column:
                    with gr.Row(variant='compact'):
                        translation_x = gr.Textbox(label="Translation X", lines=1, value=da.translation_x, interactive=True, info="move canvas left/right in pixels per frame")
                    with gr.Row(variant='compact'):
                        translation_y = gr.Textbox(label="Translation Y", lines=1, value=da.translation_y, interactive=True, info="move canvas up/down in pixels per frame")
                with gr.Column(visible=False) as only_3d_motion_column:
                    with gr.Row(variant='compact'):
                        translation_z = gr.Textbox(label="Translation Z", lines=1, value=da.translation_z, interactive=True, info="move canvas towards/away from view [speed set by FOV]")
                    with gr.Row(variant='compact'):
                        rotation_3d_x = gr.Textbox(label="Rotation 3D X", lines=1, value=da.rotation_3d_x, interactive=True, info="tilt canvas up/down in degrees per frame")
                    with gr.Row(variant='compact'):
                        rotation_3d_y = gr.Textbox(label="Rotation 3D Y", lines=1, value=da.rotation_3d_y, interactive=True, info="pan canvas left/right in degrees per frame")
                    with gr.Row(variant='compact'):
                        rotation_3d_z = gr.Textbox(label="Rotation 3D Z", lines=1, value=da.rotation_3d_z, interactive=True, info="roll canvas clockwise/anticlockwise")
                # PERSPECTIVE FLIP - params are hidden if not enabled
                with gr.Row(variant='compact') as enable_per_f_row:
                    enable_perspective_flip = gr.Checkbox(label="Enable perspective flip", value=da.enable_perspective_flip, interactive=True)
                with gr.Row(variant='compact', visible=False) as per_f_th_row:
                    perspective_flip_theta = gr.Textbox(label="Perspective flip theta", lines=1, value=da.perspective_flip_theta, interactive=True)
                with gr.Row(variant='compact', visible=False) as per_f_ph_row:
                    perspective_flip_phi = gr.Textbox(label="Perspective flip phi", lines=1, value=da.perspective_flip_phi, interactive=True)
                with gr.Row(variant='compact', visible=False) as per_f_ga_row:
                    perspective_flip_gamma = gr.Textbox(label="Perspective flip gamma", lines=1, value=da.perspective_flip_gamma, interactive=True)
                with gr.Row(variant='compact', visible=False) as per_f_f_row:
                    perspective_flip_fv = gr.Textbox(label="Perspective flip fv", lines=1, value=da.perspective_flip_fv, interactive=True,
                                                     info="the 2D vanishing point of perspective (rec. range 30-160)")
            # NOISE INNER TAB
            with gr.TabItem('Noise'):
                with gr.Column() as noise_tab_column:
                    with gr.Row(variant='compact'):
                        noise_type = gr.Radio(['uniform', 'perlin'], label="Noise type", value=da.noise_type, elem_id="noise_type")
                    with gr.Row(variant='compact'):
                        noise_schedule = gr.Textbox(label="Noise schedule", lines=1, value=da.noise_schedule, interactive=True)
                    with gr.Row(variant='compact') as perlin_row:
                        with gr.Column(min_width=220):
                            perlin_octaves = gr.Slider(label="Perlin octaves", minimum=1, maximum=7, value=da.perlin_octaves, step=1, interactive=True)
                        with gr.Column(min_width=220):
                            perlin_persistence = gr.Slider(label="Perlin persistence", minimum=0, maximum=1, value=da.perlin_persistence, step=0.02, interactive=True)
                            # following two params are INVISIBLE IN UI
                            perlin_w = gr.Slider(label="Perlin W", minimum=0.1, maximum=16, step=0.1, value=da.perlin_w, interactive=True, visible=False)
                            perlin_h = gr.Slider(label="Perlin H", minimum=0.1, maximum=16, step=0.1, value=da.perlin_h, interactive=True, visible=False)
                    with gr.Row(variant='compact'):
                        enable_noise_multiplier_scheduling = gr.Checkbox(label="Enable noise multiplier scheduling", value=da.enable_noise_multiplier_scheduling, interactive=True)
                    with gr.Row(variant='compact'):
                        noise_multiplier_schedule = gr.Textbox(label="Noise multiplier schedule", lines=1, value=da.noise_multiplier_schedule, interactive=True)
            # COHERENCE INNER TAB
            with gr.TabItem('Coherence', open=False) as coherence_accord:
                with gr.Row(variant='compact'):
                    color_coherence = gr.Dropdown(label="Color coherence", choices=['None', 'HSV', 'LAB', 'RGB', 'Video Input', 'Image'], value=da.color_coherence, type="value",
                                                  elem_id="color_coherence", interactive=True, info="choose an algorithm/ method for keeping color coherence across the animation")
                    color_force_grayscale = gr.Checkbox(label="Color force Grayscale", value=da.color_force_grayscale, interactive=True, info="force all frames to be in grayscale")
                with gr.Row(variant='compact'):
                    legacy_colormatch = gr.Checkbox(label="Legacy colormatch", value=da.legacy_colormatch, interactive=True)
                with gr.Row(visible=False) as color_coherence_image_path_row:
                    color_coherence_image_path = gr.Textbox(label="Color coherence image path", lines=1, value=da.color_coherence_image_path, interactive=True)
                with gr.Row(visible=False) as color_coherence_video_every_N_frames_row:
                    color_coherence_video_every_N_frames = gr.Number(label="Color coherence video every N frames", value=1, interactive=True)
                with gr.Row(variant='compact') as optical_flow_cadence_row:
                    with gr.Column(min_width=220) as optical_flow_cadence_column:
                        optical_flow_cadence = gr.Dropdown(choices=['None', 'RAFT', 'DIS Medium', 'DIS Fine', 'Farneback'], label="Optical flow cadence", value=da.optical_flow_cadence,
                                                           elem_id="optical_flow_cadence", interactive=True, info="use optical flow estimation for your in-between (cadence) frames")
                    with gr.Column(min_width=220, visible=False) as cadence_flow_factor_schedule_column:
                        cadence_flow_factor_schedule = gr.Textbox(label="Cadence flow factor schedule", lines=1, value=da.cadence_flow_factor_schedule, interactive=True)
                with gr.Row(variant='compact'):
                    with gr.Column(min_width=220):
                        optical_flow_redo_generation = gr.Dropdown(choices=['None', 'RAFT', 'DIS Medium', 'DIS Fine', 'Farneback'], label="Optical flow generation",
                                                                   value=da.optical_flow_redo_generation, elem_id="optical_flow_redo_generation", visible=True, interactive=True,
                                                                   info="this option takes twice as long because it generates twice in order to capture the optical flow from the previous image to the first generation, then warps the previous image and redoes the generation")
                    with gr.Column(min_width=220, visible=False) as redo_flow_factor_schedule_column:
                        redo_flow_factor_schedule = gr.Textbox(label="Generation flow factor schedule", lines=1, value=da.redo_flow_factor_schedule, interactive=True)
                with gr.Row(variant='compact'):
                    contrast_schedule = gr.Textbox(label="Contrast schedule", lines=1, value=da.contrast_schedule, interactive=True,
                                                   info="adjusts the overall contrast per frame [neutral at 1.0, recommended to *not* play with this param]")
                    diffusion_redo = gr.Slider(label="Redo generation", minimum=0, maximum=50, step=1, value=da.diffusion_redo, interactive=True,
                                               info="this option renders N times before the final render. it is suggested to lower your steps if you up your redo. seed is randomized during redo generations and restored afterwards")
                with gr.Row(variant='compact'):
                    # what to do with blank frames (they may result from glitches or the NSFW filter being turned on): reroll with +1 seed, interrupt the animation generation, or do nothing
                    reroll_blank_frames = gr.Radio(['reroll', 'interrupt', 'ignore'], label="Reroll blank frames", value=d.reroll_blank_frames, elem_id="reroll_blank_frames")
                    reroll_patience = gr.Number(value=d.reroll_patience, label="Reroll patience", interactive=True)
            # ANTI BLUR INNER TAB
            with gr.TabItem('Anti Blur', elem_id='anti_blur_accord') as anti_blur_tab:
                with gr.Row(variant='compact'):
                    amount_schedule = gr.Textbox(label="Amount schedule", lines=1, value=da.amount_schedule, interactive=True)
                with gr.Row(variant='compact'):
                    kernel_schedule = gr.Textbox(label="Kernel schedule", lines=1, value=da.kernel_schedule, interactive=True)
                with gr.Row(variant='compact'):
                    sigma_schedule = gr.Textbox(label="Sigma schedule", lines=1, value=da.sigma_schedule, interactive=True)
                with gr.Row(variant='compact'):
                    threshold_schedule = gr.Textbox(label="Threshold schedule", lines=1, value=da.threshold_schedule, interactive=True)
            with gr.TabItem('Depth Warping & FOV', elem_id='depth_warp_fov_tab') as depth_warp_fov_tab:
                # this html only shows when not in 2d/3d mode
                depth_warp_msg_html = gr.HTML(value='Please switch to 3D animation mode to view this section.', elem_id='depth_warp_msg_html')
                with gr.Row(variant='compact', visible=False) as depth_warp_row_1:
                    use_depth_warping = gr.Checkbox(label="Use depth warping", value=da.use_depth_warping, interactive=True)
                    # this following html only shows when using LeReS depth
                    leres_license_msg = gr.HTML(
                        value='Note that LeReS has a Non-Commercial <a href="https://github.com/aim-uofa/AdelaiDepth/blob/main/LeReS/LICENSE" target="_blank">license</a>. Use it only for fun/personal use.',
                        visible=False, elem_id='leres_license_msg')
                    depth_algorithm = gr.Dropdown(label="Depth Algorithm", choices=['Midas+AdaBins (old)', 'Zoe+AdaBins (old)', 'Midas-3-Hybrid', 'AdaBins', 'Zoe', 'Leres'],
                                                  value=da.depth_algorithm, type="value", elem_id="df_depth_algorithm",
                                                  interactive=True)  # 'Midas-3.1-BeitLarge' is temporarily removed until fixed 04-05-23
                    midas_weight = gr.Number(label="MiDaS/Zoe weight", value=da.midas_weight, interactive=True, visible=False,
                                             info="sets a midpoint at which a depthmap is to be drawn: range [-1 to +1]")
                with gr.Row(variant='compact', visible=False) as depth_warp_row_2:
                    padding_mode = gr.Radio(['border', 'reflection', 'zeros'], label="Padding mode", value=da.padding_mode, elem_id="padding_mode",
                                            info="controls the handling of pixels outside the field of view as they come into the scene. hover on the options for more info")
                    sampling_mode = gr.Radio(['bicubic', 'bilinear', 'nearest'], label="Sampling mode", value=da.sampling_mode, elem_id="sampling_mode")
                with gr.Row(variant='compact', visible=False) as depth_warp_row_3:
                    aspect_ratio_use_old_formula = gr.Checkbox(label="Use old aspect ratio formula", value=da.aspect_ratio_use_old_formula, interactive=True,
                                                               info="for backward compatibility. uses the formula width/height")
                with gr.Row(variant='compact', visible=False) as depth_warp_row_4:
                    aspect_ratio_schedule = gr.Textbox(label="Aspect Ratio schedule", lines=1, value=da.aspect_ratio_schedule, interactive=True,
                                                       info="adjusts the aspect ratio for the depth calculation")
                with gr.Row(variant='compact', visible=False) as depth_warp_row_5:
                    fov_schedule = gr.Textbox(label="FOV schedule", lines=1, value=da.fov_schedule, interactive=True,
                                              info="adjusts the scale at which the canvas is moved in 3D by the translation_z value. [maximum range -180 to +180, with 0 being undefined. Values closer to 180 will make the image have less depth, while values closer to 0 will allow more depth]")
                with gr.Row(variant='compact', visible=False) as depth_warp_row_6:
                    near_schedule = gr.Textbox(label="Near schedule", lines=1, value=da.near_schedule, interactive=True)
                with gr.Row(variant='compact', visible=False) as depth_warp_row_7:
                    far_schedule = gr.Textbox(label="Far schedule", lines=1, value=da.far_schedule, interactive=True)
    return {k: v for k, v in {**locals(), **vars()}.items()}

def get_tab_init(d, da, dp):
    with gr.TabItem('Init'):
        # IMAGE INIT INNER-TAB
        with gr.Tab('Image Init'):
            with gr.Row(variant='compact'):
                with gr.Column(min_width=150):
                    use_init = gr.Checkbox(label="Use init", value=d.use_init, interactive=True, visible=True)
                with gr.Column(min_width=150):
                    strength_0_no_init = gr.Checkbox(label="Strength 0 no init", value=d.strength_0_no_init, interactive=True)
                with gr.Column(min_width=170):
                    strength = gr.Slider(label="Strength", minimum=0, maximum=1, step=0.01, value=d.strength, interactive=True)
            with gr.Row(variant='compact'):
                init_image = gr.Textbox(label="Init image", lines=1, interactive=True, value=d.init_image)
        # VIDEO INIT INNER-TAB
        with gr.Tab('Video Init'):
            with gr.Row(variant='compact'):
                video_init_path = gr.Textbox(label="Video init path", lines=1, value=da.video_init_path, interactive=True)
            with gr.Row(variant='compact'):
                extract_from_frame = gr.Number(label="Extract from frame", value=da.extract_from_frame, interactive=True, precision=0)
                extract_to_frame = gr.Number(label="Extract to frame", value=da.extract_to_frame, interactive=True, precision=0)
                extract_nth_frame = gr.Number(label="Extract nth frame", value=da.extract_nth_frame, interactive=True, precision=0)
                overwrite_extracted_frames = gr.Checkbox(label="Overwrite extracted frames", value=False, interactive=True)
                use_mask_video = gr.Checkbox(label="Use mask video", value=False, interactive=True)
            with gr.Row(variant='compact'):
                video_mask_path = gr.Textbox(label="Video mask path", lines=1, value=da.video_mask_path, interactive=True)
        # MASK INIT INNER-TAB
        with gr.Tab('Mask Init'):
            with gr.Row(variant='compact'):
                use_mask = gr.Checkbox(label="Use mask", value=d.use_mask, interactive=True)
                use_alpha_as_mask = gr.Checkbox(label="Use alpha as mask", value=d.use_alpha_as_mask, interactive=True)
                invert_mask = gr.Checkbox(label="Invert mask", value=d.invert_mask, interactive=True)
                overlay_mask = gr.Checkbox(label="Overlay mask", value=d.overlay_mask, interactive=True)
            with gr.Row(variant='compact'):
                mask_file = gr.Textbox(label="Mask file", lines=1, interactive=True, value=d.mask_file)
            with gr.Row(variant='compact'):
                mask_overlay_blur = gr.Slider(label="Mask overlay blur", minimum=0, maximum=64, step=1, value=d.mask_overlay_blur, interactive=True)
            with gr.Row(variant='compact'):
                choice = mask_fill_choices[d.fill]
                fill = gr.Radio(label='Mask fill', choices=mask_fill_choices, value=choice, type="index")
            with gr.Row(variant='compact'):
                full_res_mask = gr.Checkbox(label="Full res mask", value=d.full_res_mask, interactive=True)
                full_res_mask_padding = gr.Slider(minimum=0, maximum=512, step=1, label="Full res mask padding", value=d.full_res_mask_padding, interactive=True)
            with gr.Row(variant='compact'):
                with gr.Column(min_width=240):
                    mask_contrast_adjust = gr.Number(label="Mask contrast adjust", value=d.mask_contrast_adjust, interactive=True)
                with gr.Column(min_width=250):
                    mask_brightness_adjust = gr.Number(label="Mask brightness adjust", value=d.mask_brightness_adjust, interactive=True)
        # PARSEQ ACCORD
        with gr.Accordion('Parseq', open=False):
            gr.HTML(value=get_gradio_html('parseq'))
            with gr.Row(variant='compact'):
                parseq_manifest = gr.Textbox(label="Parseq Manifest (JSON or URL)", lines=4, value=dp.parseq_manifest, interactive=True)
            with gr.Row(variant='compact'):
                parseq_use_deltas = gr.Checkbox(label="Use delta values for movement parameters", value=dp.parseq_use_deltas, interactive=True)
    return {k: v for k, v in {**locals(), **vars()}.items()}

def setup_deforum_left_side_ui():
    d, da, dp, dv, dr, dloopArgs = set_arg_lists()
    # MAIN (TOP) EXTENSION INFO ACCORD
    with gr.Accordion("Info, Links and Help", open=False, elem_id='main_top_info_accord'):
        gr.HTML(value=get_gradio_html('main'))
    with gr.Row(variant='compact'):
        show_info_on_ui = gr.Checkbox(label="Show more info", value=d.show_info_on_ui, interactive=True)
    with gr.Blocks():
        with gr.Tabs():
            # Get main tab contents:
            tab_run_params = get_tab_run(d, da)  # Run tab
            tab_keyframes_params = get_tab_keyframes(d, da, dloopArgs)  # Keyframes tab
            tab_prompts_params = get_tab_prompts(da)  # Prompts tab
            tab_init_params = get_tab_init(d, da, dp)  # Init tab
            controlnet_dict = setup_controlnet_ui()  # ControlNet tab
            tab_hybrid_params = get_tab_hybrid(da)  # Hybrid tab
            tab_output_params = get_tab_output(da, dv)  # Output tab
            # add returned gradio elements from main tabs to locals()
            for key, value in {**tab_run_params, **tab_keyframes_params, **tab_prompts_params, **tab_init_params, **controlnet_dict, **tab_hybrid_params, **tab_output_params}.items():
                locals()[key] = value

    # Gradio's Change functions - hiding and renaming elements based on other elements
    show_info_on_ui.change(fn=change_css, inputs=show_info_on_ui, outputs=gr.outputs.HTML())
    locals()['override_settings_with_file'].change(fn=hide_if_false, inputs=locals()['override_settings_with_file'], outputs=locals()['custom_settings_file'])
    locals()['sampler'].change(fn=show_when_ddim, inputs=locals()['sampler'], outputs=locals()['enable_ddim_eta_scheduling'])
    locals()['sampler'].change(fn=show_when_ancestral_samplers, inputs=locals()['sampler'], outputs=locals()['enable_ancestral_eta_scheduling'])
    locals()['enable_ancestral_eta_scheduling'].change(fn=hide_if_false, inputs=locals()['enable_ancestral_eta_scheduling'], outputs=locals()['ancestral_eta_schedule'])
    locals()['enable_ddim_eta_scheduling'].change(fn=hide_if_false, inputs=locals()['enable_ddim_eta_scheduling'], outputs=locals()['ddim_eta_schedule'])
    locals()['animation_mode'].change(fn=change_max_frames_visibility, inputs=locals()['animation_mode'], outputs=locals()['max_frames'])
    diffusion_cadence_outputs = [locals()['diffusion_cadence'], locals()['guided_images_accord'], locals()['optical_flow_cadence_row'],locals()['cadence_flow_factor_schedule'],
                                 locals()['optical_flow_redo_generation'], locals()['redo_flow_factor_schedule'], locals()['diffusion_redo']]
    for output in diffusion_cadence_outputs:
        locals()['animation_mode'].change(fn=change_diffusion_cadence_visibility, inputs=locals()['animation_mode'], outputs=output)
    three_d_related_outputs = [locals()['only_3d_motion_column'], locals()['depth_warp_row_1'], locals()['depth_warp_row_2'], locals()['depth_warp_row_3'], locals()['depth_warp_row_4'], locals()['depth_warp_row_5'], locals()['depth_warp_row_6'],
                               locals()['depth_warp_row_7']]
    for output in three_d_related_outputs:
        locals()['animation_mode'].change(fn=disble_3d_related_stuff, inputs=locals()['animation_mode'], outputs=output)
    pers_flip_outputs = [locals()['per_f_th_row'], locals()['per_f_ph_row'], locals()['per_f_ga_row'], locals()['per_f_f_row']]
    for output in pers_flip_outputs:
        locals()['enable_perspective_flip'].change(fn=hide_if_false, inputs=locals()['enable_perspective_flip'], outputs=output)
        locals()['animation_mode'].change(fn=per_flip_handle, inputs=[locals()['animation_mode'], locals()['enable_perspective_flip']], outputs=output)
    locals()['animation_mode'].change(fn=only_show_in_non_3d_mode, inputs=locals()['animation_mode'], outputs=locals()['depth_warp_msg_html'])
    locals()['animation_mode'].change(fn=enable_2d_related_stuff, inputs=locals()['animation_mode'], outputs=locals()['only_2d_motion_column'])
    locals()['animation_mode'].change(fn=disable_by_interpolation, inputs=locals()['animation_mode'], outputs=locals()['color_force_grayscale'])
    locals()['animation_mode'].change(fn=disable_by_interpolation, inputs=locals()['animation_mode'], outputs=locals()['noise_tab_column'])
    locals()['animation_mode'].change(fn=disable_pers_flip_accord, inputs=locals()['animation_mode'], outputs=locals()['enable_per_f_row'])
    locals()['animation_mode'].change(fn=disable_pers_flip_accord, inputs=locals()['animation_mode'], outputs=locals()['both_anim_mode_motion_params_column'])
    locals()['aspect_ratio_use_old_formula'].change(fn=hide_if_true, inputs=locals()['aspect_ratio_use_old_formula'], outputs=locals()['aspect_ratio_schedule'])
    locals()['animation_mode'].change(fn=show_hybrid_html_msg, inputs=locals()['animation_mode'], outputs=locals()['hybrid_msg_html'])
    locals()['animation_mode'].change(fn=change_hybrid_tab_status, inputs=locals()['animation_mode'], outputs=locals()['hybrid_sch_accord'])
    locals()['animation_mode'].change(fn=change_hybrid_tab_status, inputs=locals()['animation_mode'], outputs=locals()['hybrid_settings_accord'])
    locals()['animation_mode'].change(fn=change_hybrid_tab_status, inputs=locals()['animation_mode'], outputs=locals()['humans_masking_accord'])
    locals()['optical_flow_redo_generation'].change(fn=hide_if_none, inputs=locals()['optical_flow_redo_generation'], outputs=locals()['redo_flow_factor_schedule_column'])
    locals()['optical_flow_cadence'].change(fn=hide_if_none, inputs=locals()['optical_flow_cadence'], outputs=locals()['cadence_flow_factor_schedule_column'])
    locals()['seed_behavior'].change(fn=change_seed_iter_visibility, inputs=locals()['seed_behavior'], outputs=locals()['seed_iter_N_row'])
    locals()['seed_behavior'].change(fn=change_seed_schedule_visibility, inputs=locals()['seed_behavior'], outputs=locals()['seed_schedule_row'])
    locals()['color_coherence'].change(fn=change_color_coherence_video_every_N_frames_visibility, inputs=locals()['color_coherence'], outputs=locals()['color_coherence_video_every_N_frames_row'])
    locals()['color_coherence'].change(fn=change_color_coherence_image_path_visibility, inputs=locals()['color_coherence'], outputs=locals()['color_coherence_image_path_row'])
    locals()['noise_type'].change(fn=change_perlin_visibility, inputs=locals()['noise_type'], outputs=locals()['perlin_row'])
    locals()['diffusion_cadence'].change(fn=hide_optical_flow_cadence, inputs=locals()['diffusion_cadence'], outputs=locals()['optical_flow_cadence_row'])
    locals()['depth_algorithm'].change(fn=legacy_3d_mode, inputs=locals()['depth_algorithm'], outputs=locals()['midas_weight'])
    locals()['depth_algorithm'].change(fn=show_leres_html_msg, inputs=locals()['depth_algorithm'], outputs=locals()['leres_license_msg'])
    locals()['fps'].change(fn=change_gif_button_visibility, inputs=locals()['fps'], outputs=locals()['make_gif'])
    locals()['r_upscale_model'].change(fn=update_r_upscale_factor, inputs=locals()['r_upscale_model'], outputs=locals()['r_upscale_factor'])
    locals()['ncnn_upscale_model'].change(fn=update_r_upscale_factor, inputs=locals()['ncnn_upscale_model'], outputs=locals()['ncnn_upscale_factor'])
    locals()['ncnn_upscale_model'].change(update_upscale_out_res_by_model_name, inputs=[locals()['ncnn_upscale_in_vid_res'], locals()['ncnn_upscale_model']], outputs=locals()['ncnn_upscale_out_vid_res'])
    locals()['ncnn_upscale_factor'].change(update_upscale_out_res, inputs=[locals()['ncnn_upscale_in_vid_res'], locals()['ncnn_upscale_factor']], outputs=locals()['ncnn_upscale_out_vid_res'])
    locals()['vid_to_upscale_chosen_file'].change(vid_upscale_gradio_update_stats, inputs=[locals()['vid_to_upscale_chosen_file'], locals()['ncnn_upscale_factor']],
                                        outputs=[locals()['ncnn_upscale_in_vid_fps_ui_window'], locals()['ncnn_upscale_in_vid_frame_count_window'], locals()['ncnn_upscale_in_vid_res'], locals()['ncnn_upscale_out_vid_res']])
    locals()['hybrid_comp_mask_type'].change(fn=hide_if_none, inputs=locals()['hybrid_comp_mask_type'], outputs=locals()['hybrid_comp_mask_row'])
    hybrid_motion_outputs = [locals()['hybrid_flow_method'], locals()['hybrid_flow_factor_schedule'], locals()['hybrid_flow_consistency'], locals()['hybrid_consistency_blur'], locals()['hybrid_motion_use_prev_img']]
    for output in hybrid_motion_outputs:
        locals()['hybrid_motion'].change(fn=disable_by_non_optical_flow, inputs=locals()['hybrid_motion'], outputs=output)
    locals()['hybrid_flow_consistency'].change(fn=hide_if_false, inputs=locals()['hybrid_flow_consistency'], outputs=locals()['hybrid_consistency_blur'])
    locals()['hybrid_composite'].change(fn=disable_by_hybrid_composite_dynamic, inputs=[locals()['hybrid_composite'], locals()['hybrid_comp_mask_type']], outputs=locals()['hybrid_comp_mask_row'])
    hybrid_composite_outputs = [locals()['humans_masking_accord'], locals()['hybrid_sch_accord'], locals()['hybrid_comp_mask_type'], locals()['hybrid_use_first_frame_as_init_image'], locals()['hybrid_use_init_image']]
    for output in hybrid_composite_outputs:
        locals()['hybrid_composite'].change(fn=hide_if_false, inputs=locals()['hybrid_composite'], outputs=output)
    hybrid_comp_mask_type_outputs = [locals()['hybrid_comp_mask_blend_alpha_schedule_row'], locals()['hybrid_comp_mask_contrast_schedule_row'], locals()['hybrid_comp_mask_auto_contrast_cutoff_high_schedule_row'],
                                     locals()['hybrid_comp_mask_auto_contrast_cutoff_low_schedule_row']]
    for output in hybrid_comp_mask_type_outputs:
        locals()['hybrid_comp_mask_type'].change(fn=hide_if_none, inputs=locals()['hybrid_comp_mask_type'], outputs=output)
    # End of hybrid related
    skip_video_creation_outputs = [locals()['fps_out_format_row'], locals()['soundtrack_row'], locals()['store_frames_in_ram'], locals()['make_gif'], locals()['r_upscale_row'], locals()['delete_imgs']]
    for output in skip_video_creation_outputs:
        locals()['skip_video_creation'].change(fn=change_visibility_from_skip_video, inputs=locals()['skip_video_creation'], outputs=output)
    locals()['frame_interpolation_slow_mo_enabled'].change(fn=hide_if_false, inputs=locals()['frame_interpolation_slow_mo_enabled'], outputs=locals()['frame_interp_slow_mo_amount_column'])
    locals()['frame_interpolation_engine'].change(fn=change_interp_x_max_limit, inputs=[locals()['frame_interpolation_engine'], locals()['frame_interpolation_x_amount']], outputs=locals()['frame_interpolation_x_amount'])
    # Populate the FPS and FCount values as soon as a video is uploaded to the FileUploadBox (vid_to_interpolate_chosen_file)
    locals()['vid_to_interpolate_chosen_file'].change(gradio_f_interp_get_fps_and_fcount,
                                          inputs=[locals()['vid_to_interpolate_chosen_file'], locals()['frame_interpolation_x_amount'], locals()['frame_interpolation_slow_mo_enabled'], locals()['frame_interpolation_slow_mo_amount']],
                                          outputs=[locals()['in_vid_fps_ui_window'], locals()['in_vid_frame_count_window'], locals()['out_interp_vid_estimated_fps']])
    locals()['vid_to_interpolate_chosen_file'].change(fn=hide_interp_stats, inputs=[locals()['vid_to_interpolate_chosen_file']], outputs=[locals()['interp_live_stats_row']])
    interp_hide_list = [locals()['frame_interpolation_slow_mo_enabled'], locals()['frame_interpolation_keep_imgs'], locals()['frame_interp_amounts_row'], locals()['interp_existing_video_row']]
    for output in interp_hide_list:
        locals()['frame_interpolation_engine'].change(fn=hide_interp_by_interp_status, inputs=locals()['frame_interpolation_engine'], outputs=output)
    # END OF UI TABS

    return locals()
