from types import SimpleNamespace
from modules.ui_components import FormRow
import modules.shared as sh
from .defaults import *  # TODO: change this to not *
from .deforum_controlnet import setup_controlnet_ui, controlnet_infotext
from .gradio_funcs import *
from .video_audio_utilities import direct_stitch_vid_from_frames
from .frame_interpolation import set_interp_out_fps, gradio_f_interp_get_fps_and_fcount
from .args import DeforumArgs, DeforumAnimArgs, ParseqArgs, DeforumOutputArgs, RootArgs, LoopArgs

def set_arg_lists():
    d = SimpleNamespace(**DeforumArgs())  # default args
    da = SimpleNamespace(**DeforumAnimArgs())  # default anim args
    dp = SimpleNamespace(**ParseqArgs())  # default parseq ars
    dv = SimpleNamespace(**DeforumOutputArgs())  # default video args
    dr = SimpleNamespace(**RootArgs())  # ROOT args
    dloopArgs = SimpleNamespace(**LoopArgs())  # Guided imgs args
    return d, da, dp, dv, dr, dloopArgs

def setup_deforum_left_side_ui():
    d, da, dp, dv, dr, dloopArgs = set_arg_lists()
    # MAIN (TOP) EXTENSION INFO ACCORD
    with gr.Accordion("Info, Links and Help", open=False, elem_id='main_top_info_accord'):
        gr.HTML(value=get_gradio_html('main'))
    with gr.Row(variant='compact'):
        show_info_on_ui = gr.Checkbox(label="Show more info", value=d.show_info_on_ui, interactive=True)
    with gr.Blocks():
        with gr.Tabs():
            # RUN TAB
            with gr.TabItem('Run'):
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
                        # KEYFRAMES TAB
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
                # EXTA SCHEDULES TABS
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
            # PROMPTS TAB    
            with gr.TabItem('Prompts'):
                # PROMPTS INFO ACCORD  
                with gr.Accordion(label='*Important* notes on Prompts', elem_id='prompts_info_accord', open=False, visible=True) as prompts_info_accord:
                    gr.HTML(value=get_gradio_html('prompts'))
                with gr.Row(variant='compact'):
                    animation_prompts = gr.Textbox(label="Prompts", lines=8, interactive=True, value=DeforumAnimPrompts(),
                                                   info="full prompts list in a JSON format.  value on left side is the frame number")
                with gr.Row(variant='compact'):
                    animation_prompts_positive = gr.Textbox(label="Prompts positive", lines=1, interactive=True, placeholder="words in here will be added to the start of all positive prompts")
                with gr.Row(variant='compact'):
                    animation_prompts_negative = gr.Textbox(label="Prompts negative", value="nsfw, nude", lines=1, interactive=True,
                                                            placeholder="words in here will be added to the end of all negative prompts")
                # COMPOSABLE MASK SCHEDULING ACCORD
                with gr.Accordion('Composable Mask scheduling', open=False):
                    gr.HTML(value=get_gradio_html('composable_masks'))
                    with gr.Row(variant='compact'):
                        mask_schedule = gr.Textbox(label="Mask schedule", lines=1, value=da.mask_schedule, interactive=True)
                    with gr.Row(variant='compact'):
                        use_noise_mask = gr.Checkbox(label="Use noise mask", value=da.use_noise_mask, interactive=True)
                    with gr.Row(variant='compact'):
                        noise_mask_schedule = gr.Textbox(label="Noise mask schedule", lines=1, value=da.noise_mask_schedule, interactive=True)
            # INIT MAIN TAB
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
                        # CONTROLNET TAB
            with gr.TabItem('ControlNet'):
                gr.HTML(controlnet_infotext())
                try:
                    controlnet_dict = setup_controlnet_ui()
                except Exception as e:
                    raise Exception(e)
            # HYBRID VIDEO TAB
            with gr.TabItem('Hybrid Video'):
                # this html only shows when not in 2d/3d mode
                hybrid_msg_html = gr.HTML(value='Please, change animation mode to 2D or 3D to enable Hybrid Mode', visible=False, elem_id='hybrid_msg_html')
                # HYBRID INFO ACCORD
                with gr.Accordion("Info & Help", open=False):
                    gr.HTML(value=get_gradio_html('hybrid_video'))
                # HYBRID SETTINGS ACCORD
                with gr.Accordion("Hybrid Settings", open=True) as hybrid_settings_accord:
                    with gr.Row(variant='compact'):
                        hybrid_composite = gr.Radio(['None', 'Normal', 'Before Motion', 'After Generation'], label="Hybrid composite", value=da.hybrid_composite, elem_id="hybrid_composite")
                    with gr.Row(variant='compact'):
                        with gr.Column(min_width=340):
                            with gr.Row(variant='compact'):
                                hybrid_generate_inputframes = gr.Checkbox(label="Generate inputframes", value=da.hybrid_generate_inputframes, interactive=True)
                                hybrid_use_first_frame_as_init_image = gr.Checkbox(label="First frame as init image", value=da.hybrid_use_first_frame_as_init_image, interactive=True, visible=False)
                                hybrid_use_init_image = gr.Checkbox(label="Use init image as video", value=da.hybrid_use_init_image, interactive=True, visible=True)
                    with gr.Row(variant='compact'):
                        with gr.Column(variant='compact'):
                            with gr.Row(variant='compact'):
                                hybrid_motion = gr.Radio(['None', 'Optical Flow', 'Perspective', 'Affine'], label="Hybrid motion", value=da.hybrid_motion, elem_id="hybrid_motion")
                        with gr.Column(variant='compact'):
                            with gr.Row(variant='compact'):
                                with gr.Column(scale=1):
                                    hybrid_flow_method = gr.Radio(['RAFT', 'DIS Medium', 'DIS Fine', 'Farneback'], label="Flow method", value=da.hybrid_flow_method, elem_id="hybrid_flow_method",
                                                                  visible=False)
                            with gr.Row(variant='compact'):
                                with gr.Column(variant='compact'):
                                    hybrid_flow_consistency = gr.Checkbox(label="Flow consistency mask", value=da.hybrid_flow_consistency, interactive=True, visible=False)
                                    hybrid_consistency_blur = gr.Slider(label="Consistency mask blur", minimum=0, maximum=16, step=1, value=da.hybrid_consistency_blur, interactive=True, visible=False)
                                with gr.Column(variant='compact'):
                                    hybrid_motion_use_prev_img = gr.Checkbox(label="Motion use prev img", value=da.hybrid_motion_use_prev_img, interactive=True, visible=False)
                    with gr.Row(variant='compact'):
                        hybrid_comp_mask_type = gr.Radio(['None', 'Depth', 'Video Depth', 'Blend', 'Difference'], label="Comp mask type", value=da.hybrid_comp_mask_type,
                                                         elem_id="hybrid_comp_mask_type", visible=False)
                    with gr.Row(visible=False, variant='compact') as hybrid_comp_mask_row:
                        hybrid_comp_mask_equalize = gr.Radio(['None', 'Before', 'After', 'Both'], label="Comp mask equalize", value=da.hybrid_comp_mask_equalize, elem_id="hybrid_comp_mask_equalize")
                        with gr.Column(variant='compact'):
                            hybrid_comp_mask_auto_contrast = gr.Checkbox(label="Comp mask auto contrast", value=False, interactive=True)
                            hybrid_comp_mask_inverse = gr.Checkbox(label="Comp mask inverse", value=da.hybrid_comp_mask_inverse, interactive=True)
                    with gr.Row(variant='compact'):
                        hybrid_comp_save_extra_frames = gr.Checkbox(label="Comp save extra frames", value=False, interactive=True)
                # HYBRID SCHEDULES ACCORD
                with gr.Accordion("Hybrid Schedules", open=False, visible=False) as hybrid_sch_accord:
                    with gr.Row(variant='compact') as hybrid_comp_alpha_schedule_row:
                        hybrid_comp_alpha_schedule = gr.Textbox(label="Comp alpha schedule", lines=1, value=da.hybrid_comp_alpha_schedule, interactive=True)
                    with gr.Row(variant='compact') as hybrid_flow_factor_schedule_row:
                        hybrid_flow_factor_schedule = gr.Textbox(label="Flow factor schedule", visible=False, lines=1, value=da.hybrid_flow_factor_schedule, interactive=True)
                    with gr.Row(variant='compact', visible=False) as hybrid_comp_mask_blend_alpha_schedule_row:
                        hybrid_comp_mask_blend_alpha_schedule = gr.Textbox(label="Comp mask blend alpha schedule", lines=1, value=da.hybrid_comp_mask_blend_alpha_schedule, interactive=True,
                                                                           elem_id="hybridelemtest")
                    with gr.Row(variant='compact', visible=False) as hybrid_comp_mask_contrast_schedule_row:
                        hybrid_comp_mask_contrast_schedule = gr.Textbox(label="Comp mask contrast schedule", lines=1, value=da.hybrid_comp_mask_contrast_schedule, interactive=True)
                    with gr.Row(variant='compact', visible=False) as hybrid_comp_mask_auto_contrast_cutoff_high_schedule_row:
                        hybrid_comp_mask_auto_contrast_cutoff_high_schedule = gr.Textbox(label="Comp mask auto contrast cutoff high schedule", lines=1,
                                                                                         value=da.hybrid_comp_mask_auto_contrast_cutoff_high_schedule, interactive=True)
                    with gr.Row(variant='compact', visible=False) as hybrid_comp_mask_auto_contrast_cutoff_low_schedule_row:
                        hybrid_comp_mask_auto_contrast_cutoff_low_schedule = gr.Textbox(label="Comp mask auto contrast cutoff low schedule", lines=1,
                                                                                        value=da.hybrid_comp_mask_auto_contrast_cutoff_low_schedule, interactive=True)
                # HUMANS MASKING ACCORD
                with gr.Accordion("Humans Masking", open=False, visible=False) as humans_masking_accord:
                    with gr.Row(variant='compact'):
                        hybrid_generate_human_masks = gr.Radio(['None', 'PNGs', 'Video', 'Both'], label="Generate human masks", value=da.hybrid_generate_human_masks,
                                                               elem_id="hybrid_generate_human_masks")
            # OUTPUT TAB
            with gr.TabItem('Output', elem_id='output_tab'):
                # VID OUTPUT ACCORD
                with gr.Accordion('Video Output Settings', open=True):
                    with gr.Row(variant='compact') as fps_out_format_row:
                        fps = gr.Slider(label="FPS", value=dv.fps, minimum=1, maximum=240, step=1)
                    with gr.Column(variant='compact'):
                        with gr.Row(variant='compact') as soundtrack_row:
                            add_soundtrack = gr.Radio(['None', 'File', 'Init Video'], label="Add soundtrack", value=dv.add_soundtrack, info="add audio to video from file/url or init video",
                                                      elem_id="add_soundtrack")
                            soundtrack_path = gr.Textbox(label="Soundtrack path", lines=1, interactive=True, value=dv.soundtrack_path, info="abs. path or url to audio file")
                            # TODO: auto-hide if video input is selected?!
                        with gr.Row(variant='compact'):
                            skip_video_creation = gr.Checkbox(label="Skip video creation", value=dv.skip_video_creation, interactive=True, info="If enabled, only images will be saved")
                            delete_imgs = gr.Checkbox(label="Delete Imgs", value=dv.delete_imgs, interactive=True, info="auto-delete imgs when video is ready")
                            store_frames_in_ram = gr.Checkbox(label="Store frames in ram", value=dv.store_frames_in_ram, interactive=True, visible=False)
                            save_depth_maps = gr.Checkbox(label="Save depth maps", value=da.save_depth_maps, interactive=True, info="save animation's depth maps as extra files")
                            # the following param only shows for windows and linux users!
                            make_gif = gr.Checkbox(label="Make GIF", value=dv.make_gif, interactive=True, info="make gif in addition to the video/s")
                    with gr.Row(equal_height=True, variant='compact', visible=True) as r_upscale_row:
                        r_upscale_video = gr.Checkbox(label="Upscale", value=dv.r_upscale_video, interactive=True, info="upscale output imgs when run is finished")
                        r_upscale_model = gr.Dropdown(label="Upscale model", choices=['realesr-animevideov3', 'realesrgan-x4plus', 'realesrgan-x4plus-anime'], interactive=True,
                                                      value=dv.r_upscale_model, type="value")
                        r_upscale_factor = gr.Dropdown(choices=['x2', 'x3', 'x4'], label="Upscale factor", interactive=True, value=dv.r_upscale_factor, type="value")
                        r_upscale_keep_imgs = gr.Checkbox(label="Keep Imgs", value=dv.r_upscale_keep_imgs, interactive=True, info="don't delete upscaled imgs")
                # FRAME INTERPOLATION TAB
                with gr.Tab('Frame Interpolation') as frame_interp_tab:
                    with gr.Accordion('Important notes and Help', open=False, elem_id="f_interp_accord"):
                        gr.HTML(value=get_gradio_html('frame_interpolation'))
                    with gr.Column(variant='compact'):
                        with gr.Row(variant='compact'):
                            # Interpolation Engine
                            with gr.Column(min_width=110, scale=3):
                                frame_interpolation_engine = gr.Radio(['None', 'RIFE v4.6', 'FILM'], label="Engine", value=dv.frame_interpolation_engine,
                                                                      info="select the frame interpolation engine. hover on the options for more info")
                            with gr.Column(min_width=30, scale=1):
                                frame_interpolation_slow_mo_enabled = gr.Checkbox(label="Slow Mo", elem_id="frame_interpolation_slow_mo_enabled", value=dv.frame_interpolation_slow_mo_enabled,
                                                                                  interactive=True, visible=False)
                            with gr.Column(min_width=30, scale=1):
                                # If this is set to True, we keep all of the interpolated frames in a folder. Default is False - means we delete them at the end of the run
                                frame_interpolation_keep_imgs = gr.Checkbox(label="Keep Imgs", elem_id="frame_interpolation_keep_imgs", value=dv.frame_interpolation_keep_imgs, interactive=True,
                                                                            visible=False)
                        with gr.Row(variant='compact', visible=False) as frame_interp_amounts_row:
                            with gr.Column(min_width=180) as frame_interp_x_amount_column:
                                # How many times to interpolate (interp X)
                                frame_interpolation_x_amount = gr.Slider(minimum=2, maximum=10, step=1, label="Interp X", value=dv.frame_interpolation_x_amount, interactive=True)
                            with gr.Column(min_width=180, visible=False) as frame_interp_slow_mo_amount_column:
                                # Interp Slow-Mo (setting final output fps, not really doing anything direclty with RIFE/FILM)
                                frame_interpolation_slow_mo_amount = gr.Slider(minimum=2, maximum=10, step=1, label="Slow-Mo X", value=dv.frame_interpolation_x_amount, interactive=True)
                        with gr.Row(visible=False) as interp_existing_video_row:
                            # Interpolate any existing video from the connected PC
                            with gr.Accordion('Interpolate existing Video/ Images', open=False) as interp_existing_video_accord:
                                with gr.Row(variant='compact') as interpolate_upload_files_row:
                                    # A drag-n-drop UI box to which the user uploads a *single* (at this stage) video
                                    vid_to_interpolate_chosen_file = gr.File(label="Video to Interpolate", interactive=True, file_count="single", file_types=["video"],
                                                                             elem_id="vid_to_interpolate_chosen_file")
                                    # A drag-n-drop UI box to which the user uploads a pictures to interpolate
                                    pics_to_interpolate_chosen_file = gr.File(label="Pics to Interpolate", interactive=True, file_count="multiple", file_types=["image"],
                                                                              elem_id="pics_to_interpolate_chosen_file")
                                with gr.Row(variant='compact', visible=False) as interp_live_stats_row:
                                    # Non interactive textbox showing uploaded input vid total Frame Count
                                    in_vid_frame_count_window = gr.Textbox(label="In Frame Count", lines=1, interactive=False, value='---')
                                    # Non interactive textbox showing uploaded input vid FPS
                                    in_vid_fps_ui_window = gr.Textbox(label="In FPS", lines=1, interactive=False, value='---')
                                    # Non interactive textbox showing expected output interpolated video FPS
                                    out_interp_vid_estimated_fps = gr.Textbox(label="Interpolated Vid FPS", value='---')
                                with gr.Row(variant='compact') as interp_buttons_row:
                                    # This is the actual button that's pressed to initiate the interpolation:
                                    interpolate_button = gr.Button(value="*Interpolate Video*")
                                    interpolate_pics_button = gr.Button(value="*Interpolate Pics*")
                                # Show a text about CLI outputs:
                                gr.HTML("* check your CLI for outputs *", elem_id="below_interpolate_butts_msg")  # TODO: CSS THIS TO CENTER OF ROW!
                                # make the functin call when the interpolation button is clicked
                                interpolate_button.click(upload_vid_to_interpolate,
                                                         inputs=[vid_to_interpolate_chosen_file, frame_interpolation_engine, frame_interpolation_x_amount, frame_interpolation_slow_mo_enabled,
                                                                 frame_interpolation_slow_mo_amount, frame_interpolation_keep_imgs, in_vid_fps_ui_window])
                                interpolate_pics_button.click(upload_pics_to_interpolate,
                                                              inputs=[pics_to_interpolate_chosen_file, frame_interpolation_engine, frame_interpolation_x_amount, frame_interpolation_slow_mo_enabled,
                                                                      frame_interpolation_slow_mo_amount, frame_interpolation_keep_imgs, fps, add_soundtrack, soundtrack_path])
                # VIDEO UPSCALE TAB
                with gr.TabItem('Video Upscaling'):
                    vid_to_upscale_chosen_file = gr.File(label="Video to Upscale", interactive=True, file_count="single", file_types=["video"], elem_id="vid_to_upscale_chosen_file")
                    with gr.Column():
                        # NCNN UPSCALE TAB
                        with gr.Row(variant='compact') as ncnn_upload_vid_stats_row:
                            ncnn_upscale_in_vid_frame_count_window = gr.Textbox(label="In Frame Count", lines=1, interactive=False,
                                                                                value='---')  # Non interactive textbox showing uploaded input vid Frame Count
                            ncnn_upscale_in_vid_fps_ui_window = gr.Textbox(label="In FPS", lines=1, interactive=False, value='---')  # Non interactive textbox showing uploaded input vid FPS
                            ncnn_upscale_in_vid_res = gr.Textbox(label="In Res", lines=1, interactive=False, value='---')  # Non interactive textbox showing uploaded input resolution
                            ncnn_upscale_out_vid_res = gr.Textbox(label="Out Res", value='---')  # Non interactive textbox showing expected output resolution
                        with gr.Column():
                            with gr.Row(variant='compact', visible=True) as ncnn_actual_upscale_row:
                                ncnn_upscale_model = gr.Dropdown(label="Upscale model", choices=['realesr-animevideov3', 'realesrgan-x4plus', 'realesrgan-x4plus-anime'], interactive=True,
                                                                 value="realesr-animevideov3", type="value")
                                ncnn_upscale_factor = gr.Dropdown(choices=['x2', 'x3', 'x4'], label="Upscale factor", interactive=True, value="x2", type="value")
                                ncnn_upscale_keep_imgs = gr.Checkbox(label="Keep Imgs", value=True, interactive=True)  # fix value
                        ncnn_upscale_btn = gr.Button(value="*Upscale uploaded video*")
                        ncnn_upscale_btn.click(ncnn_upload_vid_to_upscale,
                                               inputs=[vid_to_upscale_chosen_file, ncnn_upscale_in_vid_fps_ui_window, ncnn_upscale_in_vid_res, ncnn_upscale_out_vid_res, ncnn_upscale_model,
                                                       ncnn_upscale_factor, ncnn_upscale_keep_imgs])
                        with gr.Column(visible=False):  # Upscale V1. Disabled 06-03-23
                            selected_tab = gr.State(value=0)
                            with gr.Tabs(elem_id="extras_resize_mode"):
                                with gr.TabItem('Scale by', elem_id="extras_scale_by_tab") as tab_scale_by:
                                    upscaling_resize = gr.Slider(minimum=1.0, maximum=8.0, step=0.05, label="Resize", value=2, elem_id="extras_upscaling_resize")
                                with gr.TabItem('Scale to', elem_id="extras_scale_to_tab") as tab_scale_to:
                                    with FormRow():
                                        upscaling_resize_w = gr.Slider(label="Width", minimum=1, maximum=7680, step=1, value=512, elem_id="extras_upscaling_resize_w")
                                        upscaling_resize_h = gr.Slider(label="Height", minimum=1, maximum=7680, step=1, value=512, elem_id="extras_upscaling_resize_h")
                                        upscaling_crop = gr.Checkbox(label='Crop to fit', value=True, elem_id="extras_upscaling_crop")
                            with FormRow():
                                extras_upscaler_1 = gr.Dropdown(label='Upscaler 1', elem_id="extras_upscaler_1", choices=[x.name for x in sh.sd_upscalers], value=sh.sd_upscalers[3].name)
                                extras_upscaler_2 = gr.Dropdown(label='Upscaler 2', elem_id="extras_upscaler_2", choices=[x.name for x in sh.sd_upscalers], value=sh.sd_upscalers[0].name)
                            with FormRow():
                                with gr.Column(scale=3):
                                    extras_upscaler_2_visibility = gr.Slider(minimum=0.0, maximum=1.0, step=0.001, label="Upscaler 2 visibility", value=0.0, elem_id="extras_upscaler_2_visibility")
                                with gr.Column(scale=1, min_width=80):
                                    upscale_keep_imgs = gr.Checkbox(label="Keep Imgs", elem_id="upscale_keep_imgs", value=True, interactive=True)
                            tab_scale_by.select(fn=lambda: 0, inputs=[], outputs=[selected_tab])
                            tab_scale_to.select(fn=lambda: 1, inputs=[], outputs=[selected_tab])
                            # This is the actual button that's pressed to initiate the Upscaling:
                            upscale_btn = gr.Button(value="*Upscale uploaded video*")
                            # Show a text about CLI outputs:
                            gr.HTML("* check your CLI for outputs")
                # Vid2Depth TAB
                with gr.TabItem('Vid2depth'):
                    vid_to_depth_chosen_file = gr.File(label="Video to get Depth from", interactive=True, file_count="single", file_types=["video"], elem_id="vid_to_depth_chosen_file")
                    with gr.Row(variant='compact'):
                        mode = gr.Dropdown(label='Mode', elem_id="mode", choices=['Depth (Midas/Adabins)', 'Anime Remove Background', 'Mixed', 'None (just grayscale)'], value='Depth (Midas/Adabins)')
                        threshold_value = gr.Slider(label="Threshold Value Lower", value=127, minimum=0, maximum=255, step=1)
                        threshold_value_max = gr.Slider(label="Threshold Value Upper", value=255, minimum=0, maximum=255, step=1)
                    with gr.Row(variant='compact'):
                        thresholding = gr.Radio(['None', 'Simple', 'Simple (Auto-value)', 'Adaptive (Mean)', 'Adaptive (Gaussian)'], label="Thresholding Mode", value='None')
                    with gr.Row(variant='compact'):
                        adapt_block_size = gr.Number(label="Block size", value=11)
                        adapt_c = gr.Number(label="C", value=2)
                        invert = gr.Checkbox(label='Closer is brighter', value=True, elem_id="invert")
                    with gr.Row(variant='compact'):
                        end_blur = gr.Slider(label="End blur width", value=0, minimum=0, maximum=255, step=1)
                        midas_weight_vid2depth = gr.Slider(label="MiDaS weight (vid2depth)", value=da.midas_weight, minimum=0, maximum=1, step=0.05, interactive=True,
                                                           info="sets a midpoint at which a depthmap is to be drawn: range [-1 to +1]")
                        depth_keep_imgs = gr.Checkbox(label='Keep Imgs', value=True, elem_id="depth_keep_imgs")
                    with gr.Row(variant='compact'):
                        # This is the actual button that's pressed to initiate the Upscaling:
                        depth_btn = gr.Button(value="*Get depth from uploaded video*")
                    with gr.Row(variant='compact'):
                        # Show a text about CLI outputs:
                        gr.HTML("* check your CLI for outputs")
                        # make the function call when the UPSCALE button is clicked
                    depth_btn.click(upload_vid_to_depth,
                                    inputs=[vid_to_depth_chosen_file, mode, thresholding, threshold_value, threshold_value_max, adapt_block_size, adapt_c, invert, end_blur, midas_weight_vid2depth,
                                            depth_keep_imgs])
                # STITCH FRAMES TO VID TAB
                with gr.TabItem('Frames to Video') as stitch_imgs_to_vid_row:
                    gr.HTML(value=get_gradio_html('frames_to_video'))
                    with gr.Row(variant='compact'):
                        image_path = gr.Textbox(label="Image path", lines=1, interactive=True, value=dv.image_path)
                    ffmpeg_stitch_imgs_but = gr.Button(value="*Stitch frames to video*")
                    ffmpeg_stitch_imgs_but.click(direct_stitch_vid_from_frames, inputs=[image_path, fps, add_soundtrack, soundtrack_path])
                # **OLD + NON ACTIVES AREA**
                with gr.Accordion(visible=False, label='INVISIBLE') as not_in_use_accordion:
                    perlin_w = gr.Slider(label="Perlin W", minimum=0.1, maximum=16, step=0.1, value=da.perlin_w, interactive=True)
                    perlin_h = gr.Slider(label="Perlin H", minimum=0.1, maximum=16, step=0.1, value=da.perlin_h, interactive=True)
                    seed_enable_extras = gr.Checkbox(label="Enable subseed controls", value=False)
    # Gradio's Change functions - hiding and renaming elements based on other elements
    show_info_on_ui.change(fn=change_css, inputs=show_info_on_ui, outputs=gr.outputs.HTML())
    fps.change(fn=change_gif_button_visibility, inputs=fps, outputs=make_gif)
    r_upscale_model.change(fn=update_r_upscale_factor, inputs=r_upscale_model, outputs=r_upscale_factor)
    ncnn_upscale_model.change(fn=update_r_upscale_factor, inputs=ncnn_upscale_model, outputs=ncnn_upscale_factor)
    ncnn_upscale_model.change(update_upscale_out_res_by_model_name, inputs=[ncnn_upscale_in_vid_res, ncnn_upscale_model], outputs=ncnn_upscale_out_vid_res)
    ncnn_upscale_factor.change(update_upscale_out_res, inputs=[ncnn_upscale_in_vid_res, ncnn_upscale_factor], outputs=ncnn_upscale_out_vid_res)
    vid_to_upscale_chosen_file.change(vid_upscale_gradio_update_stats, inputs=[vid_to_upscale_chosen_file, ncnn_upscale_factor],
                                      outputs=[ncnn_upscale_in_vid_fps_ui_window, ncnn_upscale_in_vid_frame_count_window, ncnn_upscale_in_vid_res, ncnn_upscale_out_vid_res])
    animation_mode.change(fn=change_max_frames_visibility, inputs=animation_mode, outputs=max_frames)
    diffusion_cadence_outputs = [diffusion_cadence, guided_images_accord, optical_flow_cadence_row, cadence_flow_factor_schedule,
                                 optical_flow_redo_generation, redo_flow_factor_schedule, diffusion_redo]
    for output in diffusion_cadence_outputs:
        animation_mode.change(fn=change_diffusion_cadence_visibility, inputs=animation_mode, outputs=output)
    three_d_related_outputs = [only_3d_motion_column, depth_warp_row_1, depth_warp_row_2, depth_warp_row_3, depth_warp_row_4, depth_warp_row_5, depth_warp_row_6,
                               depth_warp_row_7]  # depth_3d_warping_accord, fov_accord
    for output in three_d_related_outputs:
        animation_mode.change(fn=disble_3d_related_stuff, inputs=animation_mode, outputs=output)
    pers_flip_outputs = [per_f_th_row, per_f_ph_row, per_f_ga_row, per_f_f_row]
    for output in pers_flip_outputs:
        enable_perspective_flip.change(fn=hide_if_false, inputs=enable_perspective_flip, outputs=output)
    animation_mode.change(fn=only_show_in_non_3d_mode, inputs=animation_mode, outputs=depth_warp_msg_html)
    animation_mode.change(fn=enable_2d_related_stuff, inputs=animation_mode, outputs=only_2d_motion_column)
    animation_mode.change(fn=disable_by_interpolation, inputs=animation_mode, outputs=color_force_grayscale)
    animation_mode.change(fn=disable_by_interpolation, inputs=animation_mode, outputs=noise_tab_column)
    pers_flip_hide_all_param_names = [per_f_th_row, per_f_ph_row, per_f_ga_row, per_f_f_row]
    for output in pers_flip_hide_all_param_names:
        animation_mode.change(fn=per_flip_handle, inputs=[animation_mode, enable_perspective_flip], outputs=output)
    # show/hide enable persp flip mode button. Shown only in 2d/3d anim modes
    animation_mode.change(fn=disable_pers_flip_accord, inputs=animation_mode, outputs=enable_per_f_row)
    animation_mode.change(fn=disable_pers_flip_accord, inputs=animation_mode, outputs=both_anim_mode_motion_params_column)
    aspect_ratio_use_old_formula.change(fn=hide_if_true, inputs=aspect_ratio_use_old_formula, outputs=aspect_ratio_schedule)
    # Hybrid related:
    animation_mode.change(fn=show_hybrid_html_msg, inputs=animation_mode, outputs=hybrid_msg_html)
    animation_mode.change(fn=change_hybrid_tab_status, inputs=animation_mode, outputs=hybrid_sch_accord)
    animation_mode.change(fn=change_hybrid_tab_status, inputs=animation_mode, outputs=hybrid_settings_accord)
    animation_mode.change(fn=change_hybrid_tab_status, inputs=animation_mode, outputs=humans_masking_accord)
    optical_flow_redo_generation.change(fn=hide_if_none, inputs=optical_flow_redo_generation, outputs=redo_flow_factor_schedule_column)
    override_settings_with_file.change(fn=hide_if_false, inputs=override_settings_with_file, outputs=custom_settings_file)
    hybrid_comp_mask_type.change(fn=hide_if_none, inputs=hybrid_comp_mask_type, outputs=hybrid_comp_mask_row)
    hybrid_motion_outputs = [hybrid_flow_method, hybrid_flow_factor_schedule, hybrid_flow_consistency, hybrid_consistency_blur, hybrid_motion_use_prev_img]
    for output in hybrid_motion_outputs:
        hybrid_motion.change(fn=disable_by_non_optical_flow, inputs=hybrid_motion, outputs=output)
    hybrid_flow_consistency.change(fn=hide_if_false, inputs=hybrid_flow_consistency, outputs=hybrid_consistency_blur)
    optical_flow_cadence.change(fn=hide_if_none, inputs=optical_flow_cadence, outputs=cadence_flow_factor_schedule_column)
    hybrid_composite.change(fn=disable_by_hybrid_composite_dynamic, inputs=[hybrid_composite, hybrid_comp_mask_type], outputs=hybrid_comp_mask_row)
    hybrid_composite_outputs = [humans_masking_accord, hybrid_sch_accord, hybrid_comp_mask_type, hybrid_use_first_frame_as_init_image, hybrid_use_init_image]
    for output in hybrid_composite_outputs:
        hybrid_composite.change(fn=hide_if_false, inputs=hybrid_composite, outputs=output)
    hybrid_comp_mask_type_outputs = [hybrid_comp_mask_blend_alpha_schedule_row, hybrid_comp_mask_contrast_schedule_row, hybrid_comp_mask_auto_contrast_cutoff_high_schedule_row,
                                     hybrid_comp_mask_auto_contrast_cutoff_low_schedule_row]
    for output in hybrid_comp_mask_type_outputs:
        hybrid_comp_mask_type.change(fn=hide_if_none, inputs=hybrid_comp_mask_type, outputs=output)
    # End of hybrid related
    seed_behavior.change(fn=change_seed_iter_visibility, inputs=seed_behavior, outputs=seed_iter_N_row)
    seed_behavior.change(fn=change_seed_schedule_visibility, inputs=seed_behavior, outputs=seed_schedule_row)
    color_coherence.change(fn=change_color_coherence_video_every_N_frames_visibility, inputs=color_coherence, outputs=color_coherence_video_every_N_frames_row)
    color_coherence.change(fn=change_color_coherence_image_path_visibility, inputs=color_coherence, outputs=color_coherence_image_path_row)
    noise_type.change(fn=change_perlin_visibility, inputs=noise_type, outputs=perlin_row)
    skip_video_creation_outputs = [fps_out_format_row, soundtrack_row, store_frames_in_ram, make_gif, r_upscale_row, delete_imgs]
    for output in skip_video_creation_outputs:
        skip_video_creation.change(fn=change_visibility_from_skip_video, inputs=skip_video_creation, outputs=output)
    frame_interpolation_slow_mo_enabled.change(fn=hide_if_false, inputs=frame_interpolation_slow_mo_enabled, outputs=frame_interp_slow_mo_amount_column)
    frame_interpolation_engine.change(fn=change_interp_x_max_limit, inputs=[frame_interpolation_engine, frame_interpolation_x_amount], outputs=frame_interpolation_x_amount)
    [change_fn.change(set_interp_out_fps, inputs=[frame_interpolation_x_amount, frame_interpolation_slow_mo_enabled, frame_interpolation_slow_mo_amount, in_vid_fps_ui_window],
                      outputs=out_interp_vid_estimated_fps) for change_fn in [frame_interpolation_x_amount, frame_interpolation_slow_mo_amount, frame_interpolation_slow_mo_enabled]]
    # Populate the FPS and FCount values as soon as a video is uploaded to the FileUploadBox (vid_to_interpolate_chosen_file)
    vid_to_interpolate_chosen_file.change(gradio_f_interp_get_fps_and_fcount,
                                          inputs=[vid_to_interpolate_chosen_file, frame_interpolation_x_amount, frame_interpolation_slow_mo_enabled, frame_interpolation_slow_mo_amount],
                                          outputs=[in_vid_fps_ui_window, in_vid_frame_count_window, out_interp_vid_estimated_fps])
    vid_to_interpolate_chosen_file.change(fn=hide_interp_stats, inputs=[vid_to_interpolate_chosen_file], outputs=[interp_live_stats_row])
    interp_hide_list = [frame_interpolation_slow_mo_enabled, frame_interpolation_keep_imgs, frame_interp_amounts_row, interp_existing_video_row]
    for output in interp_hide_list:
        frame_interpolation_engine.change(fn=hide_interp_by_interp_status, inputs=frame_interpolation_engine, outputs=output)
    diffusion_cadence.change(fn=hide_optical_flow_cadence, inputs=diffusion_cadence, outputs=optical_flow_cadence_row)
    depth_algorithm.change(fn=legacy_3d_mode, inputs=depth_algorithm, outputs=midas_weight)
    depth_algorithm.change(fn=show_leres_html_msg, inputs=depth_algorithm, outputs=leres_license_msg)
    ddim_outputs = [enable_ddim_eta_scheduling]
    for output in ddim_outputs:
        sampler.change(fn=show_when_ddim, inputs=sampler, outputs=output)
    ancestral_outputs = [enable_ancestral_eta_scheduling]
    for output in ancestral_outputs:
        sampler.change(fn=show_when_ancestral_samplers, inputs=sampler, outputs=output)
    enable_ancestral_eta_scheduling.change(fn=hide_if_false, inputs=enable_ancestral_eta_scheduling, outputs=ancestral_eta_schedule)
    enable_ddim_eta_scheduling.change(fn=hide_if_false, inputs=enable_ddim_eta_scheduling, outputs=ddim_eta_schedule)
    # END OF UI TABS
    stuff = locals()
    stuff = {**stuff, **controlnet_dict}
    stuff.pop('controlnet_dict')
    return stuff
