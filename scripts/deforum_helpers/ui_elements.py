import gradio as gr
import modules.shared as sh
from modules.ui_components import FormRow
from .defaults import get_gradio_html, DeforumAnimPrompts
from .video_audio_utilities import direct_stitch_vid_from_frames
from .gradio_funcs import upload_vid_to_interpolate, upload_pics_to_interpolate, ncnn_upload_vid_to_upscale, upload_vid_to_depth

def get_tab_run(d, da):
    with gr.TabItem('Run'):  # RUN TAB
        with gr.Row(variant='compact'):
            sampler = create_gr_elem(d.sampler)
            steps = create_gr_elem(d.steps)
        with gr.Row(variant='compact'):
            W = create_gr_elem(d.W)
            H = create_gr_elem(d.H)
        with gr.Row(variant='compact'):
            seed = create_gr_elem(d.seed)
            batch_name = create_gr_elem(d.batch_name)
        with gr.Row(variant='compact'):
            # Seed enable extras is INVISIBLE in the ui!
            seed_enable_extras = create_gr_elem(d.seed_enable_extras)
            restore_faces = create_gr_elem(d.restore_faces)
            tiling = create_gr_elem(d.tiling)
            enable_ddim_eta_scheduling = create_gr_elem(da.enable_ddim_eta_scheduling)
            enable_ancestral_eta_scheduling = create_gr_elem(da.enable_ancestral_eta_scheduling)
        with gr.Row(variant='compact') as eta_sch_row:
            ddim_eta_schedule = create_gr_elem(da.ddim_eta_schedule)
            ancestral_eta_schedule = create_gr_elem(da.ancestral_eta_schedule)
        # RUN FROM SETTING FILE ACCORD
        with gr.Accordion('Batch Mode, Resume and more', open=False):
            with gr.Tab('Batch Mode/ run from setting files'):
                with gr.Row(variant='compact'):  # TODO: handle this inside one of the args functions?
                    override_settings_with_file = gr.Checkbox(label="Enable batch mode", value=False, interactive=True, elem_id='override_settings',
                                                              info="run from a list of setting .txt files. Upload them to the box on the right (visible when enabled)")
                    custom_settings_file = gr.File(label="Setting files", interactive=True, file_count="multiple", file_types=[".txt"], elem_id="custom_setting_file", visible=False)
            # RESUME ANIMATION ACCORD
            with gr.Tab('Resume Animation'):
                with gr.Row(variant='compact'):
                    resume_from_timestring = create_gr_elem(da.resume_from_timestring)
                    resume_timestring = create_gr_elem(da.resume_timestring)
            with gr.Row(variant='compact') as pix2pix_img_cfg_scale_row:
                pix2pix_img_cfg_scale_schedule = create_gr_elem(da.pix2pix_img_cfg_scale_schedule)
    return {k: v for k, v in {**locals(), **vars()}.items()}

gradio_elements = {
    "number": gr.Number,
    "checkbox": gr.Checkbox,
    "slider": gr.Slider,
    "textbox": gr.Textbox,
    "dropdown": gr.Dropdown,
    "radio": gr.Radio
}

def create_gr_elem(d):
    obj_type_str = d["type"]
    obj_type = gradio_elements[obj_type_str]

    elem_params = {
        'label': d.get("label"),
        'value': d.get("value"),
        'minimum': d.get("min"),
        'maximum': d.get("max"),
        'step': d.get("step"),
        'precision': d.get("precision"),
        'choices': d.get("choices"),
        'visible': d.get("visible"),
        'info': d.get("info"),
        'lines': d.get("lines"),
        'type': d.get("radio_type")
    }
    return obj_type(**{k: v for k, v in elem_params.items() if v is not None})

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
                use_looper = create_gr_elem(dloopArgs.use_looper)
            with gr.Row(variant='compact'):
                init_images = create_gr_elem(dloopArgs.init_images)
            # GUIDED IMAGES SCHEDULES ACCORD
            with gr.Accordion('Guided images schedules', open=False):
                with gr.Row(variant='compact'):
                    image_strength_schedule = create_gr_elem(dloopArgs.image_strength_schedule)
                with gr.Row(variant='compact'):
                    blendFactorMax = create_gr_elem(dloopArgs.blendFactorMax)
                with gr.Row(variant='compact'):
                    blendFactorSlope = create_gr_elem(dloopArgs.blendFactorSlope)
                with gr.Row(variant='compact'):
                    tweening_frames_schedule = create_gr_elem(dloopArgs.tweening_frames_schedule)
                with gr.Row(variant='compact'):
                    color_correction_factor = create_gr_elem(dloopArgs.color_correction_factor)
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
                    seed_behavior = create_gr_elem(d.seed_behavior)
                with gr.Row(variant='compact') as seed_iter_N_row:
                    seed_iter_N = create_gr_elem(d.seed_iter_N)
                with gr.Row(visible=False) as seed_schedule_row:
                    seed_schedule = create_gr_elem(da.seed_schedule)
            with gr.TabItem('SubSeed', open=False) as subseed_sch_tab:
                with gr.Row(variant='compact'):
                    enable_subseed_scheduling = create_gr_elem(da.enable_subseed_scheduling)
                    subseed_schedule = create_gr_elem(da.subseed_schedule)
                    subseed_strength_schedule = create_gr_elem(da.subseed_strength_schedule)
                with gr.Row(variant='compact'):
                    seed_resize_from_w = create_gr_elem(d.seed_resize_from_w)
                    seed_resize_from_h = create_gr_elem(d.seed_resize_from_h)
            # Steps Scheduling
            with gr.TabItem('Step') as a13:
                with gr.Row(variant='compact'):
                    enable_steps_scheduling = create_gr_elem(da.enable_steps_scheduling)
                with gr.Row(variant='compact'):
                    steps_schedule = create_gr_elem(da.steps_schedule)
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
                    reroll_blank_frames = create_gr_elem(d.reroll_blank_frames)
                    reroll_patience = create_gr_elem(d.reroll_patience)
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

def get_tab_prompts(da):
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
    return {k: v for k, v in {**locals(), **vars()}.items()}

def get_tab_init(d, da, dp):
    with gr.TabItem('Init'):
        # IMAGE INIT INNER-TAB
        with gr.Tab('Image Init'):
            with gr.Row(variant='compact'):
                with gr.Column(min_width=150):
                    use_init = create_gr_elem(d.use_init)
                with gr.Column(min_width=150):
                    strength_0_no_init = create_gr_elem(d.strength_0_no_init)
                with gr.Column(min_width=170):
                    strength = create_gr_elem(d.strength)
            with gr.Row(variant='compact'):
                init_image = create_gr_elem(d.init_image)
        # VIDEO INIT INNER-TAB
        with gr.Tab('Video Init'):
            with gr.Row(variant='compact'):
                video_init_path = create_gr_elem(da.video_init_path)
            with gr.Row(variant='compact'):
                extract_from_frame = create_gr_elem(da.extract_from_frame)
                extract_to_frame = create_gr_elem(da.extract_to_frame)
                extract_nth_frame = create_gr_elem(da.extract_nth_frame)
                overwrite_extracted_frames = create_gr_elem(da.overwrite_extracted_frames)
                use_mask_video = create_gr_elem(da.use_mask_video)
            with gr.Row(variant='compact'):
                video_mask_path = create_gr_elem(da.video_mask_path)
        # MASK INIT INNER-TAB
        with gr.Tab('Mask Init'):
            with gr.Row(variant='compact'):
                use_mask = create_gr_elem(d.use_mask)
                use_alpha_as_mask = create_gr_elem(d.use_alpha_as_mask)
                invert_mask = create_gr_elem(d.invert_mask)
                overlay_mask = create_gr_elem(d.overlay_mask)
            with gr.Row(variant='compact'):
                mask_file = create_gr_elem(d.mask_file)
            with gr.Row(variant='compact'):
                mask_overlay_blur = create_gr_elem(d.mask_overlay_blur)
            with gr.Row(variant='compact'):
                fill = create_gr_elem(d.fill)
            with gr.Row(variant='compact'):
                full_res_mask = create_gr_elem(d.full_res_mask)
                full_res_mask_padding = create_gr_elem(d.full_res_mask_padding)
            with gr.Row(variant='compact'):
                with gr.Column(min_width=240):
                    mask_contrast_adjust = create_gr_elem(d.mask_contrast_adjust)
                with gr.Column(min_width=250):
                    mask_brightness_adjust = create_gr_elem(d.mask_brightness_adjust)
        # PARSEQ ACCORD
        with gr.Accordion('Parseq', open=False):
            gr.HTML(value=get_gradio_html('parseq'))
            with gr.Row(variant='compact'):
                parseq_manifest = create_gr_elem(dp.parseq_manifest)
            with gr.Row(variant='compact'):
                parseq_use_deltas = create_gr_elem(dp.parseq_use_deltas)
    return {k: v for k, v in {**locals(), **vars()}.items()}

def get_tab_hybrid(da):
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
    return {k: v for k, v in {**locals(), **vars()}.items()}

def get_tab_output(da, dv):
    with gr.TabItem('Output', elem_id='output_tab'):
        # VID OUTPUT ACCORD
        with gr.Accordion('Video Output Settings', open=True):
            with gr.Row(variant='compact') as fps_out_format_row:
                fps = create_gr_elem(dv.fps)
            with gr.Column(variant='compact'):
                with gr.Row(variant='compact') as soundtrack_row:
                    add_soundtrack = create_gr_elem(dv.add_soundtrack)
                    soundtrack_path = create_gr_elem(dv.soundtrack_path)
                with gr.Row(variant='compact'):
                    skip_video_creation = create_gr_elem(dv.skip_video_creation)
                    delete_imgs = create_gr_elem(dv.delete_imgs)
                    store_frames_in_ram = create_gr_elem(dv.store_frames_in_ram)
                    save_depth_maps = create_gr_elem(da.save_depth_maps)
                    make_gif = create_gr_elem(dv.make_gif)
            with gr.Row(equal_height=True, variant='compact', visible=True) as r_upscale_row:
                r_upscale_video = create_gr_elem(dv.r_upscale_video)
                r_upscale_model = create_gr_elem(dv.r_upscale_model)
                r_upscale_factor = create_gr_elem(dv.r_upscale_factor)
                r_upscale_keep_imgs = create_gr_elem(dv.r_upscale_keep_imgs)
        # FRAME INTERPOLATION TAB
        with gr.Tab('Frame Interpolation') as frame_interp_tab:
            with gr.Accordion('Important notes and Help', open=False, elem_id="f_interp_accord"):
                gr.HTML(value=get_gradio_html('frame_interpolation'))
            with gr.Column(variant='compact'):
                with gr.Row(variant='compact'):
                    # Interpolation Engine
                    with gr.Column(min_width=110, scale=3):
                        frame_interpolation_engine = create_gr_elem(dv.frame_interpolation_engine)
                    with gr.Column(min_width=30, scale=1):
                        frame_interpolation_slow_mo_enabled = create_gr_elem(dv.frame_interpolation_slow_mo_enabled)
                    with gr.Column(min_width=30, scale=1):
                        # If this is set to True, we keep all the interpolated frames in a folder. Default is False - means we delete them at the end of the run
                        frame_interpolation_keep_imgs = create_gr_elem(dv.frame_interpolation_keep_imgs)
                with gr.Row(variant='compact', visible=False) as frame_interp_amounts_row:
                    with gr.Column(min_width=180) as frame_interp_x_amount_column:
                        # How many times to interpolate (interp X)
                        frame_interpolation_x_amount = create_gr_elem(dv.frame_interpolation_x_amount)
                    with gr.Column(min_width=180, visible=False) as frame_interp_slow_mo_amount_column:
                        # Interp Slow-Mo (setting final output fps, not really doing anything directly with RIFE/FILM)
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
                                                                        value='---')  # Non-interactive textbox showing uploaded input vid Frame Count
                    ncnn_upscale_in_vid_fps_ui_window = gr.Textbox(label="In FPS", lines=1, interactive=False, value='---')  # Non-interactive textbox showing uploaded input vid FPS
                    ncnn_upscale_in_vid_res = gr.Textbox(label="In Res", lines=1, interactive=False, value='---')  # Non-interactive textbox showing uploaded input resolution
                    ncnn_upscale_out_vid_res = gr.Textbox(label="Out Res", value='---')  # Non-interactive textbox showing expected output resolution
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
    return {k: v for k, v in {**locals(), **vars()}.items()}