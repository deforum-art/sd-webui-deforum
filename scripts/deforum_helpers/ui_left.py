from types import SimpleNamespace
from .defaults import get_gradio_html
from .gradio_funcs import *
from .frame_interpolation import gradio_f_interp_get_fps_and_fcount
from .args import DeforumArgs, DeforumAnimArgs, ParseqArgs, DeforumOutputArgs, RootArgs, LoopArgs
from .deforum_controlnet import setup_controlnet_ui
from .ui_elements import get_tab_run, get_tab_keyframes, get_tab_prompts, get_tab_init, get_tab_hybrid, get_tab_output

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
