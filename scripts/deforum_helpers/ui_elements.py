import gradio as gr
import modules.shared as sh
from modules.ui_components import FormRow
from .defaults import get_gradio_html, DeforumAnimPrompts
from .video_audio_utilities import direct_stitch_vid_from_frames
from .gradio_funcs import upload_vid_to_interpolate, upload_pics_to_interpolate, ncnn_upload_vid_to_upscale, upload_vid_to_depth

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
                        # If this is set to True, we keep all the interpolated frames in a folder. Default is False - means we delete them at the end of the run
                        frame_interpolation_keep_imgs = gr.Checkbox(label="Keep Imgs", elem_id="frame_interpolation_keep_imgs", value=dv.frame_interpolation_keep_imgs, interactive=True,
                                                                    visible=False)
                with gr.Row(variant='compact', visible=False) as frame_interp_amounts_row:
                    with gr.Column(min_width=180) as frame_interp_x_amount_column:
                        # How many times to interpolate (interp X)
                        frame_interpolation_x_amount = gr.Slider(minimum=2, maximum=10, step=1, label="Interp X", value=dv.frame_interpolation_x_amount, interactive=True)
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