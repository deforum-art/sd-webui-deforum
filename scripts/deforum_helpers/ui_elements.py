import gradio as gr
from modules.ui_components import FormRow, FormColumn
from .defaults import get_gradio_html, DeforumAnimPrompts
from .video_audio_utilities import direct_stitch_vid_from_frames
from .gradio_funcs import upload_vid_to_interpolate, upload_pics_to_interpolate, ncnn_upload_vid_to_upscale, upload_vid_to_depth

def create_gr_elem(d):
    # Capitalize and CamelCase the orig value under "type", which defines gr.inputs.type in lower_case.
    # Examples: "dropdown" becomes gr.Dropdown, and "checkbox_group" becomes gr.CheckboxGroup.
    obj_type_str = ''.join(word.title() for word in d["type"].split('_'))
    obj_type = getattr(gr, obj_type_str)

    # Prepare parameters for gradio element creation
    params = {k: v for k, v in d.items() if k != "type" and v is not None}

    # If we're creating a Radio element and 'radio_type' is specified, then use it to set gr.radio's type
    if obj_type_str == 'Radio' and 'radio_type' in params:
        params['type'] = params.pop('radio_type')

    return obj_type(**params)

# ******** Important message ********
# All get_tab functions use FormRow()/ FormColumn() by default, unless we have a gr.File inside that row/column, then we use gr.Row()/gr.Column() instead
# ******** Important message ********
def get_tab_run(d, da):
    with gr.TabItem('Run'):  # RUN TAB
        with FormRow():
            sampler = create_gr_elem(d.sampler)
            steps = create_gr_elem(d.steps)
        with FormRow():
            W = create_gr_elem(d.W)
            H = create_gr_elem(d.H)
        with FormRow():
            seed = create_gr_elem(d.seed)
            batch_name = create_gr_elem(d.batch_name)
        with FormRow():
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
                with gr.Row():  # TODO: handle this inside one of the args functions?
                    override_settings_with_file = gr.Checkbox(label="Enable batch mode", value=False, interactive=True, elem_id='override_settings',
                                                              info="run from a list of setting .txt files. Upload them to the box on the right (visible when enabled)")
                    custom_settings_file = gr.File(label="Setting files", interactive=True, file_count="multiple", file_types=[".txt"], elem_id="custom_setting_file", visible=False)
            # RESUME ANIMATION ACCORD
            with gr.Tab('Resume Animation'):
                with FormRow():
                    resume_from_timestring = create_gr_elem(da.resume_from_timestring)
                    resume_timestring = create_gr_elem(da.resume_timestring)
            with gr.Row(variant='compact') as pix2pix_img_cfg_scale_row:
                pix2pix_img_cfg_scale_schedule = create_gr_elem(da.pix2pix_img_cfg_scale_schedule)
    return {k: v for k, v in {**locals(), **vars()}.items()}

def get_tab_keyframes(d, da, dloopArgs):
    with gr.TabItem('Keyframes'):  # TODO make a some sort of the original dictionary parsing
        with FormRow():
            with FormColumn(scale=2):
                animation_mode = create_gr_elem(da.animation_mode)
            with FormColumn(scale=1, min_width=180):
                border = create_gr_elem(da.border)
        with FormRow():
            diffusion_cadence = create_gr_elem(da.diffusion_cadence)
            max_frames = create_gr_elem(da.max_frames)
        # GUIDED IMAGES ACCORD
        with gr.Accordion('Guided Images', open=False, elem_id='guided_images_accord') as guided_images_accord:
            # GUIDED IMAGES INFO ACCORD
            with gr.Accordion('*READ ME before you use this mode!*', open=False):
                gr.HTML(value=get_gradio_html('guided_imgs'))
            with FormRow():
                use_looper = create_gr_elem(dloopArgs.use_looper)
            with FormRow():
                init_images = create_gr_elem(dloopArgs.init_images)
            # GUIDED IMAGES SCHEDULES ACCORD
            with gr.Accordion('Guided images schedules', open=False):
                with FormRow():
                    image_strength_schedule = create_gr_elem(dloopArgs.image_strength_schedule)
                with FormRow():
                    blendFactorMax = create_gr_elem(dloopArgs.blendFactorMax)
                with FormRow():
                    blendFactorSlope = create_gr_elem(dloopArgs.blendFactorSlope)
                with FormRow():
                    tweening_frames_schedule = create_gr_elem(dloopArgs.tweening_frames_schedule)
                with FormRow():
                    color_correction_factor = create_gr_elem(dloopArgs.color_correction_factor)
        # EXTRA SCHEDULES TABS
        with gr.Tabs():
            with gr.TabItem('Strength'):
                with FormRow():
                    strength_schedule = create_gr_elem(da.strength_schedule)
            with gr.TabItem('CFG'):
                with FormRow():
                    cfg_scale_schedule = create_gr_elem(da.cfg_scale_schedule)
                with FormRow():
                    enable_clipskip_scheduling = create_gr_elem(da.enable_clipskip_scheduling)
                with FormRow():
                    clipskip_schedule = create_gr_elem(da.clipskip_schedule)
            with gr.TabItem('Seed'):
                with FormRow():
                    seed_behavior = create_gr_elem(d.seed_behavior)
                with FormRow() as seed_iter_N_row:
                    seed_iter_N = create_gr_elem(d.seed_iter_N)
                with FormRow(visible=False) as seed_schedule_row:
                    seed_schedule = create_gr_elem(da.seed_schedule)
            with gr.TabItem('SubSeed', open=False) as subseed_sch_tab:
                with FormRow():
                    enable_subseed_scheduling = create_gr_elem(da.enable_subseed_scheduling)
                    subseed_schedule = create_gr_elem(da.subseed_schedule)
                    subseed_strength_schedule = create_gr_elem(da.subseed_strength_schedule)
                with FormRow():
                    seed_resize_from_w = create_gr_elem(d.seed_resize_from_w)
                    seed_resize_from_h = create_gr_elem(d.seed_resize_from_h)
            # Steps Scheduling
            with gr.TabItem('Step'):
                with FormRow():
                    enable_steps_scheduling = create_gr_elem(da.enable_steps_scheduling)
                with FormRow():
                    steps_schedule = create_gr_elem(da.steps_schedule)
            # Sampler Scheduling
            with gr.TabItem('Sampler'):
                with FormRow():
                    enable_sampler_scheduling = create_gr_elem(da.enable_sampler_scheduling)
                with FormRow():
                    sampler_schedule = create_gr_elem(da.sampler_schedule)
            # Checkpoint Scheduling
            with gr.TabItem('Checkpoint'):
                with FormRow():
                    enable_checkpoint_scheduling = create_gr_elem(da.enable_checkpoint_scheduling)
                with FormRow():
                    checkpoint_schedule = create_gr_elem(da.checkpoint_schedule)
        # MOTION INNER TAB
        with gr.Tabs(elem_id='motion_noise_etc'):
            with gr.TabItem('Motion') as motion_tab:
                with FormColumn() as only_2d_motion_column:
                    with FormRow():
                        zoom = create_gr_elem(da.zoom)
                    with FormRow():
                        angle = create_gr_elem(da.angle)
                    with FormRow():
                        transform_center_x = create_gr_elem(da.transform_center_x)
                    with FormRow():
                        transform_center_y = create_gr_elem(da.transform_center_y)
                with FormColumn() as both_anim_mode_motion_params_column:
                    with FormRow():
                        translation_x = create_gr_elem(da.translation_x)
                    with FormRow():
                        translation_y = create_gr_elem(da.translation_y)
                with FormColumn(visible=False) as only_3d_motion_column:
                    with FormRow():
                        translation_z = create_gr_elem(da.translation_z)
                    with FormRow():
                        rotation_3d_x = create_gr_elem(da.rotation_3d_x)
                    with FormRow():
                        rotation_3d_y = create_gr_elem(da.rotation_3d_y)
                    with FormRow():
                        rotation_3d_z = create_gr_elem(da.rotation_3d_z)
                # PERSPECTIVE FLIP - inner params are hidden if not enabled
                with FormRow() as enable_per_f_row:
                    enable_perspective_flip = create_gr_elem(da.enable_perspective_flip)
                with FormRow(visible=False) as per_f_th_row:
                    perspective_flip_theta = create_gr_elem(da.perspective_flip_theta)
                with FormRow(visible=False) as per_f_ph_row:
                    perspective_flip_phi = create_gr_elem(da.perspective_flip_phi)
                with FormRow(visible=False) as per_f_ga_row:
                    perspective_flip_gamma = create_gr_elem(da.perspective_flip_gamma)
                with FormRow(visible=False) as per_f_f_row:
                    perspective_flip_fv = create_gr_elem(da.perspective_flip_fv)
            # NOISE INNER TAB
            with gr.TabItem('Noise'):
                with FormColumn() as noise_tab_column:
                    with FormRow():
                        noise_type = create_gr_elem(da.noise_type)
                    with FormRow():
                        noise_schedule = create_gr_elem(da.noise_schedule)
                    with FormRow() as perlin_row:
                        with FormColumn(min_width=220):
                            perlin_octaves = create_gr_elem(da.perlin_octaves)
                        with FormColumn(min_width=220):
                            perlin_persistence = create_gr_elem(da.perlin_persistence)
                            # following two params are INVISIBLE IN UI as of 21-05-23
                            perlin_w = create_gr_elem(da.perlin_w)
                            perlin_h = create_gr_elem(da.perlin_h)
                    with FormRow():
                        enable_noise_multiplier_scheduling = create_gr_elem(da.enable_noise_multiplier_scheduling)
                    with FormRow():
                        noise_multiplier_schedule = create_gr_elem(da.noise_multiplier_schedule)
            # COHERENCE INNER TAB
            with gr.TabItem('Coherence', open=False) as coherence_accord:
                with FormRow():
                    color_coherence = create_gr_elem(da.color_coherence)
                    color_force_grayscale = create_gr_elem(da.color_force_grayscale)
                with FormRow():
                    legacy_colormatch = create_gr_elem(da.legacy_colormatch)
                with FormRow(visible=False) as color_coherence_image_path_row:
                    color_coherence_image_path = create_gr_elem(da.color_coherence_image_path)
                with FormRow(visible=False) as color_coherence_video_every_N_frames_row:
                    color_coherence_video_every_N_frames = create_gr_elem(da.color_coherence_video_every_N_frames)
                with FormRow() as optical_flow_cadence_row:
                    with FormColumn(min_width=220) as optical_flow_cadence_column:
                        optical_flow_cadence = create_gr_elem(da.optical_flow_cadence)
                    with FormColumn(min_width=220, visible=False) as cadence_flow_factor_schedule_column:
                        cadence_flow_factor_schedule = create_gr_elem(da.cadence_flow_factor_schedule)
                with FormRow():
                    with FormColumn(min_width=220):
                        optical_flow_redo_generation = create_gr_elem(da.optical_flow_redo_generation)
                    with FormColumn(min_width=220, visible=False) as redo_flow_factor_schedule_column:
                        redo_flow_factor_schedule = create_gr_elem(da.redo_flow_factor_schedule)
                with FormRow():
                    contrast_schedule = gr.Textbox(label="Contrast schedule", lines=1, value=da.contrast_schedule, interactive=True,
                                                   info="adjusts the overall contrast per frame [neutral at 1.0, recommended to *not* play with this param]")
                    diffusion_redo = gr.Slider(label="Redo generation", minimum=0, maximum=50, step=1, value=da.diffusion_redo, interactive=True,
                                               info="this option renders N times before the final render. it is suggested to lower your steps if you up your redo. seed is randomized during redo generations and restored afterwards")
                with FormRow():
                    # what to do with blank frames (they may result from glitches or the NSFW filter being turned on): reroll with +1 seed, interrupt the animation generation, or do nothing
                    reroll_blank_frames = create_gr_elem(d.reroll_blank_frames)
                    reroll_patience = create_gr_elem(d.reroll_patience)
            # ANTI BLUR INNER TAB
            with gr.TabItem('Anti Blur', elem_id='anti_blur_accord') as anti_blur_tab:
                with FormRow():
                    amount_schedule = create_gr_elem(da.amount_schedule)
                with FormRow():
                    kernel_schedule = create_gr_elem(da.kernel_schedule)
                with FormRow():
                    sigma_schedule = create_gr_elem(da.sigma_schedule)
                with FormRow():
                    threshold_schedule = create_gr_elem(da.threshold_schedule)
            with gr.TabItem('Depth Warping & FOV', elem_id='depth_warp_fov_tab') as depth_warp_fov_tab:
                # this html only shows when not in 2d/3d mode
                depth_warp_msg_html = gr.HTML(value='Please switch to 3D animation mode to view this section.', elem_id='depth_warp_msg_html')
                with FormRow(visible=False) as depth_warp_row_1:
                    use_depth_warping = create_gr_elem(da.use_depth_warping)
                    # *the following html only shows when LeReS depth is selected*
                    leres_license_msg = gr.HTML(value=get_gradio_html('leres'), visible=False, elem_id='leres_license_msg')
                    depth_algorithm = create_gr_elem(da.depth_algorithm)
                    midas_weight = create_gr_elem(da.midas_weight)
                with FormRow(visible=False) as depth_warp_row_2:
                    padding_mode = create_gr_elem(da.padding_mode)
                    sampling_mode = create_gr_elem(da.sampling_mode)
                with FormRow(visible=False) as depth_warp_row_3:
                    aspect_ratio_use_old_formula = create_gr_elem(da.aspect_ratio_use_old_formula)
                with FormRow(visible=False) as depth_warp_row_4:
                    aspect_ratio_schedule = create_gr_elem(da.aspect_ratio_schedule)
                with FormRow(visible=False) as depth_warp_row_5:
                    fov_schedule = create_gr_elem(da.fov_schedule)
                with FormRow(visible=False) as depth_warp_row_6:
                    near_schedule = create_gr_elem(da.near_schedule)
                with FormRow(visible=False) as depth_warp_row_7:
                    far_schedule = create_gr_elem(da.far_schedule)

    return {k: v for k, v in {**locals(), **vars()}.items()}

def get_tab_prompts(da):
    with gr.TabItem('Prompts'):
        # PROMPTS INFO ACCORD
        with gr.Accordion(label='*Important* notes on Prompts', elem_id='prompts_info_accord', open=False) as prompts_info_accord:
            gr.HTML(value=get_gradio_html('prompts'))
        with FormRow():
            animation_prompts = gr.Textbox(label="Prompts", lines=8, interactive=True, value=DeforumAnimPrompts(),
                                           info="full prompts list in a JSON format.  value on left side is the frame number")
        with FormRow():
            animation_prompts_positive = gr.Textbox(label="Prompts positive", lines=1, interactive=True, placeholder="words in here will be added to the start of all positive prompts")
        with FormRow():
            animation_prompts_negative = gr.Textbox(label="Prompts negative", value="nsfw, nude", lines=1, interactive=True,
                                                    placeholder="words in here will be added to the end of all negative prompts")
        # COMPOSABLE MASK SCHEDULING ACCORD
        with gr.Accordion('Composable Mask scheduling', open=False):
            gr.HTML(value=get_gradio_html('composable_masks'))
            with FormRow():
                mask_schedule = create_gr_elem(da.mask_schedule)
            with FormRow():
                use_noise_mask = create_gr_elem(da.use_noise_mask)
            with FormRow():
                noise_mask_schedule = create_gr_elem(da.noise_mask_schedule)

    return {k: v for k, v in {**locals(), **vars()}.items()}

def get_tab_init(d, da, dp):
    with gr.TabItem('Init'):
        # IMAGE INIT INNER-TAB
        with gr.Tab('Image Init'):
            with FormRow():
                with gr.Column(min_width=150):
                    use_init = create_gr_elem(d.use_init)
                with gr.Column(min_width=150):
                    strength_0_no_init = create_gr_elem(d.strength_0_no_init)
                with gr.Column(min_width=170):
                    strength = create_gr_elem(d.strength)
            with FormRow():
                init_image = create_gr_elem(d.init_image)
        # VIDEO INIT INNER-TAB
        with gr.Tab('Video Init'):
            with FormRow():
                video_init_path = create_gr_elem(da.video_init_path)
            with FormRow():
                extract_from_frame = create_gr_elem(da.extract_from_frame)
                extract_to_frame = create_gr_elem(da.extract_to_frame)
                extract_nth_frame = create_gr_elem(da.extract_nth_frame)
                overwrite_extracted_frames = create_gr_elem(da.overwrite_extracted_frames)
                use_mask_video = create_gr_elem(da.use_mask_video)
            with FormRow():
                video_mask_path = create_gr_elem(da.video_mask_path)
        # MASK INIT INNER-TAB
        with gr.Tab('Mask Init'):
            with FormRow():
                use_mask = create_gr_elem(d.use_mask)
                use_alpha_as_mask = create_gr_elem(d.use_alpha_as_mask)
                invert_mask = create_gr_elem(d.invert_mask)
                overlay_mask = create_gr_elem(d.overlay_mask)
            with FormRow():
                mask_file = create_gr_elem(d.mask_file)
            with FormRow():
                mask_overlay_blur = create_gr_elem(d.mask_overlay_blur)
            with FormRow():
                fill = create_gr_elem(d.fill)
            with FormRow():
                full_res_mask = create_gr_elem(d.full_res_mask)
                full_res_mask_padding = create_gr_elem(d.full_res_mask_padding)
            with FormRow():
                with FormColumn(min_width=240):
                    mask_contrast_adjust = create_gr_elem(d.mask_contrast_adjust)
                with FormColumn(min_width=250):
                    mask_brightness_adjust = create_gr_elem(d.mask_brightness_adjust)
        # PARSEQ ACCORD
        with gr.Accordion('Parseq', open=False):
            gr.HTML(value=get_gradio_html('parseq'))
            with FormRow():
                parseq_manifest = create_gr_elem(dp.parseq_manifest)
            with FormRow():
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
            with FormRow():
                hybrid_composite = gr.Radio(['None', 'Normal', 'Before Motion', 'After Generation'], label="Hybrid composite", value=da.hybrid_composite, elem_id="hybrid_composite")
            with FormRow():
                with FormColumn(min_width=340):
                    with FormRow():
                        hybrid_generate_inputframes = create_gr_elem(da.hybrid_generate_inputframes)
                        hybrid_use_first_frame_as_init_image = create_gr_elem(da.hybrid_use_first_frame_as_init_image)
                        hybrid_use_init_image = create_gr_elem(da.hybrid_use_init_image)
            with FormRow():
                with FormColumn():
                    with FormRow():
                        hybrid_motion = create_gr_elem(da.hybrid_motion)
                with FormColumn():
                    with FormRow():
                        with FormColumn(scale=1):
                            hybrid_flow_method = create_gr_elem(da.hybrid_flow_method)
                    with FormRow():
                        with FormColumn():
                            hybrid_flow_consistency = create_gr_elem(da.hybrid_flow_consistency)
                            hybrid_consistency_blur = create_gr_elem(da.hybrid_consistency_blur)
                        with FormColumn():
                            hybrid_motion_use_prev_img = create_gr_elem(da.hybrid_motion_use_prev_img)
            with FormRow():
                hybrid_comp_mask_type = create_gr_elem(da.hybrid_comp_mask_type)
            with gr.Row(visible=False, variant='compact') as hybrid_comp_mask_row:
                hybrid_comp_mask_equalize = create_gr_elem(da.hybrid_comp_mask_equalize)
                with FormColumn():
                    hybrid_comp_mask_auto_contrast = gr.Checkbox(label="Comp mask auto contrast", value=False, interactive=True)
                    hybrid_comp_mask_inverse = gr.Checkbox(label="Comp mask inverse", value=da.hybrid_comp_mask_inverse, interactive=True)
            with FormRow():
                hybrid_comp_save_extra_frames = gr.Checkbox(label="Comp save extra frames", value=False, interactive=True)
        # HYBRID SCHEDULES ACCORD
        with gr.Accordion("Hybrid Schedules", open=False, visible=False) as hybrid_sch_accord:
            with FormRow() as hybrid_comp_alpha_schedule_row:
                hybrid_comp_alpha_schedule = create_gr_elem(da.hybrid_comp_alpha_schedule)
            with FormRow() as hybrid_flow_factor_schedule_row:
                hybrid_flow_factor_schedule = create_gr_elem(da.hybrid_flow_factor_schedule)
            with FormRow(visible=False) as hybrid_comp_mask_blend_alpha_schedule_row:
                hybrid_comp_mask_blend_alpha_schedule = create_gr_elem(da.hybrid_comp_mask_blend_alpha_schedule)
            with FormRow(visible=False) as hybrid_comp_mask_contrast_schedule_row:
                hybrid_comp_mask_contrast_schedule = create_gr_elem(da.hybrid_comp_mask_contrast_schedule)
            with FormRow(visible=False) as hybrid_comp_mask_auto_contrast_cutoff_high_schedule_row:
                hybrid_comp_mask_auto_contrast_cutoff_high_schedule = create_gr_elem(da.hybrid_comp_mask_auto_contrast_cutoff_high_schedule)
            with FormRow(visible=False) as hybrid_comp_mask_auto_contrast_cutoff_low_schedule_row:
                hybrid_comp_mask_auto_contrast_cutoff_low_schedule = create_gr_elem(da.hybrid_comp_mask_auto_contrast_cutoff_low_schedule)
        # HUMANS MASKING ACCORD
        with gr.Accordion("Humans Masking", open=False, visible=False) as humans_masking_accord:
            with FormRow():
                hybrid_generate_human_masks = create_gr_elem(da.hybrid_generate_human_masks)

    return {k: v for k, v in {**locals(), **vars()}.items()}

def get_tab_output(da, dv):
    with gr.TabItem('Output', elem_id='output_tab'):
        # VID OUTPUT ACCORD
        with gr.Accordion('Video Output Settings', open=True):
            with FormRow() as fps_out_format_row:
                fps = create_gr_elem(dv.fps)
            with FormColumn():
                with FormRow() as soundtrack_row:
                    add_soundtrack = create_gr_elem(dv.add_soundtrack)
                    soundtrack_path = create_gr_elem(dv.soundtrack_path)
                with FormRow():
                    skip_video_creation = create_gr_elem(dv.skip_video_creation)
                    delete_imgs = create_gr_elem(dv.delete_imgs)
                    store_frames_in_ram = create_gr_elem(dv.store_frames_in_ram)
                    save_depth_maps = create_gr_elem(da.save_depth_maps)
                    make_gif = create_gr_elem(dv.make_gif)
            with FormRow(equal_height=True) as r_upscale_row:
                r_upscale_video = create_gr_elem(dv.r_upscale_video)
                r_upscale_model = create_gr_elem(dv.r_upscale_model)
                r_upscale_factor = create_gr_elem(dv.r_upscale_factor)
                r_upscale_keep_imgs = create_gr_elem(dv.r_upscale_keep_imgs)
        # FRAME INTERPOLATION TAB
        with gr.Tab('Frame Interpolation') as frame_interp_tab:
            with gr.Accordion('Important notes and Help', open=False, elem_id="f_interp_accord"):
                gr.HTML(value=get_gradio_html('frame_interpolation'))
            with gr.Column():
                with gr.Row():
                    # Interpolation Engine
                    with gr.Column(min_width=110, scale=3):
                        frame_interpolation_engine = create_gr_elem(dv.frame_interpolation_engine)
                    with gr.Column(min_width=30, scale=1):
                        frame_interpolation_slow_mo_enabled = create_gr_elem(dv.frame_interpolation_slow_mo_enabled)
                    with gr.Column(min_width=30, scale=1):
                        # If this is set to True, we keep all the interpolated frames in a folder. Default is False - means we delete them at the end of the run
                        frame_interpolation_keep_imgs = create_gr_elem(dv.frame_interpolation_keep_imgs)
                    with gr.Column(min_width=30, scale=1):
                        frame_interpolation_use_upscaled = create_gr_elem(dv.frame_interpolation_use_upscaled)                        
                with FormRow(visible=False) as frame_interp_amounts_row:
                    with gr.Column(min_width=180) as frame_interp_x_amount_column:
                        # How many times to interpolate (interp X)
                        frame_interpolation_x_amount = create_gr_elem(dv.frame_interpolation_x_amount)
                    with gr.Column(min_width=180, visible=False) as frame_interp_slow_mo_amount_column:
                        # Interp Slow-Mo (setting final output fps, not really doing anything directly with RIFE/FILM)
                        frame_interpolation_slow_mo_amount = create_gr_elem(dv.frame_interpolation_slow_mo_amount)
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
                        with FormRow(visible=False) as interp_live_stats_row:
                            # Non-interactive textbox showing uploaded input vid total Frame Count
                            in_vid_frame_count_window = gr.Textbox(label="In Frame Count", lines=1, interactive=False, value='---')
                            # Non-interactive textbox showing uploaded input vid FPS
                            in_vid_fps_ui_window = gr.Textbox(label="In FPS", lines=1, interactive=False, value='---')
                            # Non-interactive textbox showing expected output interpolated video FPS
                            out_interp_vid_estimated_fps = gr.Textbox(label="Interpolated Vid FPS", value='---')
                        with FormRow() as interp_buttons_row:
                            # This is the actual button that's pressed to initiate the interpolation:
                            interpolate_button = gr.Button(value="*Interpolate Video*")
                            interpolate_pics_button = gr.Button(value="*Interpolate Pics*")
                        # Show a text about CLI outputs:
                        gr.HTML("* check your CLI for outputs *", elem_id="below_interpolate_butts_msg")
                        # make the function call when the interpolation button is clicked
                        interpolate_button.click(fn=upload_vid_to_interpolate,
                                                 inputs=[vid_to_interpolate_chosen_file, frame_interpolation_engine, frame_interpolation_x_amount, frame_interpolation_slow_mo_enabled,
                                                         frame_interpolation_slow_mo_amount, frame_interpolation_keep_imgs, in_vid_fps_ui_window])
                        interpolate_pics_button.click(fn=upload_pics_to_interpolate,
                                                      inputs=[pics_to_interpolate_chosen_file, frame_interpolation_engine, frame_interpolation_x_amount, frame_interpolation_slow_mo_enabled,
                                                              frame_interpolation_slow_mo_amount, frame_interpolation_keep_imgs, fps, add_soundtrack, soundtrack_path])
        # VIDEO UPSCALE TAB - not built using our args.py at all - all data and params are here and in .upscaling file
        with gr.TabItem('Video Upscaling'):
            vid_to_upscale_chosen_file = gr.File(label="Video to Upscale", interactive=True, file_count="single", file_types=["video"], elem_id="vid_to_upscale_chosen_file")
            with gr.Column():
                # NCNN UPSCALE TAB
                with FormRow() as ncnn_upload_vid_stats_row:
                    ncnn_upscale_in_vid_frame_count_window = gr.Textbox(label="In Frame Count", lines=1, interactive=False,
                                                                        value='---')  # Non-interactive textbox showing uploaded input vid Frame Count
                    ncnn_upscale_in_vid_fps_ui_window = gr.Textbox(label="In FPS", lines=1, interactive=False, value='---')  # Non-interactive textbox showing uploaded input vid FPS
                    ncnn_upscale_in_vid_res = gr.Textbox(label="In Res", lines=1, interactive=False, value='---')  # Non-interactive textbox showing uploaded input resolution
                    ncnn_upscale_out_vid_res = gr.Textbox(label="Out Res", value='---')  # Non-interactive textbox showing expected output resolution
                with gr.Column():
                    with FormRow() as ncnn_actual_upscale_row:
                        ncnn_upscale_model = create_gr_elem(dv.r_upscale_model)  # note that we re-use *r_upscale_model* in here to create the gradio element as they are the same
                        ncnn_upscale_factor = create_gr_elem(dv.r_upscale_factor)  # note that we re-use *r_upscale_facto*r in here to create the gradio element as they are the same
                        ncnn_upscale_keep_imgs = create_gr_elem(dv.r_upscale_keep_imgs)  # note that we re-use *r_upscale_keep_imgs* in here to create the gradio element as they are the same
                ncnn_upscale_btn = gr.Button(value="*Upscale uploaded video*")
                ncnn_upscale_btn.click(fn=ncnn_upload_vid_to_upscale,
                                       inputs=[vid_to_upscale_chosen_file, ncnn_upscale_in_vid_fps_ui_window, ncnn_upscale_in_vid_res, ncnn_upscale_out_vid_res, ncnn_upscale_model,
                                               ncnn_upscale_factor, ncnn_upscale_keep_imgs])
        # Vid2Depth TAB - not built using our args.py at all - all data and params are here and in .vid2depth file
        with gr.TabItem('Vid2depth'):
            vid_to_depth_chosen_file = gr.File(label="Video to get Depth from", interactive=True, file_count="single", file_types=["video"], elem_id="vid_to_depth_chosen_file")
            with FormRow():
                mode = gr.Dropdown(label='Mode', elem_id="mode", choices=['Depth (Midas/Adabins)', 'Anime Remove Background', 'Mixed', 'None (just grayscale)'], value='Depth (Midas/Adabins)')
                threshold_value = gr.Slider(label="Threshold Value Lower", value=127, minimum=0, maximum=255, step=1)
                threshold_value_max = gr.Slider(label="Threshold Value Upper", value=255, minimum=0, maximum=255, step=1)
            with FormRow():
                thresholding = gr.Radio(['None', 'Simple', 'Simple (Auto-value)', 'Adaptive (Mean)', 'Adaptive (Gaussian)'], label="Thresholding Mode", value='None')
            with FormRow():
                adapt_block_size = gr.Number(label="Block size", value=11)
                adapt_c = gr.Number(label="C", value=2)
                invert = gr.Checkbox(label='Closer is brighter', value=True, elem_id="invert")
            with FormRow():
                end_blur = gr.Slider(label="End blur width", value=0, minimum=0, maximum=255, step=1)
                midas_weight_vid2depth = gr.Slider(label="MiDaS weight (vid2depth)", value=da.midas_weight, minimum=0, maximum=1, step=0.05, interactive=True,
                                                   info="sets a midpoint at which a depth-map is to be drawn: range [-1 to +1]")
                depth_keep_imgs = gr.Checkbox(label='Keep Imgs', value=True, elem_id="depth_keep_imgs")
            with FormRow():
                # This is the actual button that's pressed to initiate the Upscaling:
                depth_btn = gr.Button(value="*Get depth from uploaded video*")
            with FormRow():
                # Show a text about CLI outputs:
                gr.HTML("* check your CLI for outputs")
                # make the function call when the UPSCALE button is clicked
            depth_btn.click(fn=upload_vid_to_depth,
                            inputs=[vid_to_depth_chosen_file, mode, thresholding, threshold_value, threshold_value_max, adapt_block_size, adapt_c, invert, end_blur, midas_weight_vid2depth, depth_keep_imgs])
        # STITCH FRAMES TO VID TAB
        with gr.TabItem('Frames to Video') as stitch_imgs_to_vid_row:
            gr.HTML(value=get_gradio_html('frames_to_video'))
            with FormRow():
                image_path = create_gr_elem(dv.image_path)
            ffmpeg_stitch_imgs_but = gr.Button(value="*Stitch frames to video*")
            ffmpeg_stitch_imgs_but.click(fn=direct_stitch_vid_from_frames, inputs=[image_path, fps, add_soundtrack, soundtrack_path])

    return {k: v for k, v in {**locals(), **vars()}.items()}