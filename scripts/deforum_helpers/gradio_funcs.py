import gradio as gr
import modules.paths as ph
from .general_utils import get_os
from .upscaling import process_ncnn_upscale_vid_upload_logic
from .video_audio_utilities import extract_number, get_quick_vid_info, get_ffmpeg_params
from .frame_interpolation import process_interp_vid_upload_logic, process_interp_pics_upload_logic
from .vid2depth import process_depth_vid_upload_logic

f_models_path = ph.models_path + '/Deforum'

# START gradio-to-frame-interoplation/ upscaling functions
def upload_vid_to_interpolate(file, engine, x_am, sl_enabled, sl_am, keep_imgs, in_vid_fps):
    # print msg and do nothing if vid not uploaded or interp_x not provided
    if not file or engine == 'None':
        return print("Please upload a video and set a proper value for 'Interp X'. Can't interpolate x0 times :)")
    f_location, f_crf, f_preset = get_ffmpeg_params()

    process_interp_vid_upload_logic(file, engine, x_am, sl_enabled, sl_am, keep_imgs, f_location, f_crf, f_preset, in_vid_fps, f_models_path, file.orig_name)

def upload_pics_to_interpolate(pic_list, engine, x_am, sl_enabled, sl_am, keep_imgs, fps, add_audio, audio_track):
    from PIL import Image

    if pic_list is None or len(pic_list) < 2:
        return print("Please upload at least 2 pics for interpolation.")
    f_location, f_crf, f_preset = get_ffmpeg_params()
    # make sure all uploaded pics have the same resolution
    pic_sizes = [Image.open(picture_path.name).size for picture_path in pic_list]
    if len(set(pic_sizes)) != 1:
        return print("All uploaded pics need to be of the same Width and Height / resolution.")

    resolution = pic_sizes[0]

    process_interp_pics_upload_logic(pic_list, engine, x_am, sl_enabled, sl_am, keep_imgs, f_location, f_crf, f_preset, fps, f_models_path, resolution, add_audio, audio_track)

def ncnn_upload_vid_to_upscale(vid_path, in_vid_fps, in_vid_res, out_vid_res, upscale_model, upscale_factor, keep_imgs):
    if vid_path is None:
        print("Please upload a video :)")
        return
    f_location, f_crf, f_preset = get_ffmpeg_params()
    current_user = get_os()
    process_ncnn_upscale_vid_upload_logic(vid_path, in_vid_fps, in_vid_res, out_vid_res, f_models_path, upscale_model, upscale_factor, keep_imgs, f_location, f_crf, f_preset, current_user)

def upload_vid_to_depth(vid_to_depth_chosen_file, mode, thresholding, threshold_value, threshold_value_max, adapt_block_size, adapt_c, invert, end_blur, midas_weight_vid2depth, depth_keep_imgs):
    # print msg and do nothing if vid not uploaded
    if not vid_to_depth_chosen_file:
        return print("Please upload a video :()")
    f_location, f_crf, f_preset = get_ffmpeg_params()

    process_depth_vid_upload_logic(vid_to_depth_chosen_file, mode, thresholding, threshold_value, threshold_value_max, adapt_block_size, adapt_c, invert, end_blur, midas_weight_vid2depth,
                                   vid_to_depth_chosen_file.orig_name, depth_keep_imgs, f_location, f_crf, f_preset, f_models_path)

# END gradio-to-frame-interoplation/ upscaling functions

def change_visibility_from_skip_video(choice):
    return gr.update(visible=False) if choice else gr.update(visible=True)

def update_r_upscale_factor(choice):
    return gr.update(value='x4', choices=['x4']) if choice != 'realesr-animevideov3' else gr.update(value='x2', choices=['x2', 'x3', 'x4'])

def change_perlin_visibility(choice):
    return gr.update(visible=choice == "perlin")

def legacy_3d_mode(choice):
    return gr.update(visible=choice.lower() in ["midas+adabins (old)", 'zoe+adabins (old)'])

def change_color_coherence_image_path_visibility(choice):
    return gr.update(visible=choice == "Image")

def change_color_coherence_video_every_N_frames_visibility(choice):
    return gr.update(visible=choice == "Video Input")

def change_seed_iter_visibility(choice):
    return gr.update(visible=choice == "iter")

def change_seed_schedule_visibility(choice):
    return gr.update(visible=choice == "schedule")

def disable_pers_flip_accord(choice):
    return gr.update(visible=True) if choice in ['2D', '3D'] else gr.update(visible=False)

def per_flip_handle(anim_mode, per_f_enabled):
    if anim_mode in ['2D', '3D'] and per_f_enabled:
        return gr.update(visible=True)
    return gr.update(visible=False)

def change_max_frames_visibility(choice):
    return gr.update(visible=choice != "Video Input")

def change_diffusion_cadence_visibility(choice):
    return gr.update(visible=choice not in ['Video Input', 'Interpolation'])

def disble_3d_related_stuff(choice):
    return gr.update(visible=False) if choice != '3D' else gr.update(visible=True)

def only_show_in_non_3d_mode(choice):
    return gr.update(visible=False) if choice == '3D' else gr.update(visible=True)

def enable_2d_related_stuff(choice):
    return gr.update(visible=True) if choice == '2D' else gr.update(visible=False)

def disable_by_interpolation(choice):
    return gr.update(visible=False) if choice in ['Interpolation'] else gr.update(visible=True)

def disable_by_video_input(choice):
    return gr.update(visible=False) if choice in ['Video Input'] else gr.update(visible=True)

def hide_if_none(choice):
    return gr.update(visible=choice != "None")

def change_gif_button_visibility(choice):
    if choice is None or choice == "":
        return gr.update(visible=True)
    return gr.update(visible=False, value=False) if int(choice) > 30 else gr.update(visible=True)

def hide_if_false(choice):
    return gr.update(visible=True) if choice else gr.update(visible=False)

def hide_if_true(choice):
    return gr.update(visible=False) if choice else gr.update(visible=True)

def disable_by_hybrid_composite_dynamic(choice, comp_mask_type):
    if choice in ['Normal', 'Before Motion', 'After Generation']:
        if comp_mask_type != 'None':
            return gr.update(visible=True)
    return gr.update(visible=False)

def disable_by_non_optical_flow(choice):
    return gr.update(visible=False) if choice != 'Optical Flow' else gr.update(visible=True)

# Upscaling Gradio UI related funcs
def vid_upscale_gradio_update_stats(vid_path, upscale_factor):
    if not vid_path:
        return '---', '---', '---', '---'
    factor = extract_number(upscale_factor)
    fps, fcount, resolution = get_quick_vid_info(vid_path.name)
    in_res_str = f"{resolution[0]}*{resolution[1]}"
    out_res_str = f"{resolution[0] * factor}*{resolution[1] * factor}"
    return fps, fcount, in_res_str, out_res_str

def update_upscale_out_res(in_res, upscale_factor):
    if not in_res:
        return '---'
    factor = extract_number(upscale_factor)
    w, h = [int(x) * factor for x in in_res.split('*')]
    return f"{w}*{h}"

def update_upscale_out_res_by_model_name(in_res, upscale_model_name):
    if not upscale_model_name or in_res == '---':
        return '---'
    factor = 2 if upscale_model_name == 'realesr-animevideov3' else 4
    return f"{int(in_res.split('*')[0]) * factor}*{int(in_res.split('*')[1]) * factor}"

def hide_optical_flow_cadence(cadence_value):
    return gr.update(visible=True) if cadence_value > 1 else gr.update(visible=False)

def hide_interp_by_interp_status(choice):
    return gr.update(visible=False) if choice == 'None' else gr.update(visible=True)

def change_interp_x_max_limit(engine_name, current_value):
    if engine_name == 'FILM':
        return gr.update(maximum=300)
    elif current_value > 10:
        return gr.update(maximum=10, value=2)
    return gr.update(maximum=10)

def hide_interp_stats(choice):
    return gr.update(visible=True) if choice is not None else gr.update(visible=False)

def show_hybrid_html_msg(choice):
    return gr.update(visible=True) if choice not in ['2D', '3D'] else gr.update(visible=False)

def change_hybrid_tab_status(choice):
    return gr.update(visible=True) if choice in ['2D', '3D'] else gr.update(visible=False)

def show_leres_html_msg(choice):
    return gr.update(visible=True) if choice.lower() == 'leres' else gr.update(visible=False)

def show_when_ddim(sampler_name):
    return gr.update(visible=True) if sampler_name.lower() == 'ddim' else gr.update(visible=False)

def show_when_ancestral_samplers(sampler_name):
    return gr.update(visible=True) if sampler_name.lower() in ['euler a', 'dpm++ 2s a', 'dpm2 a', 'dpm2 a karras', 'dpm++ 2s a karras'] else gr.update(visible=False)

def change_css(checkbox_status):
    if checkbox_status:
        display = "block"
    else:
        display = "none"

    html_template = f'''
        <style>
            #tab_deforum_interface .svelte-e8n7p6, #f_interp_accord {{
                display: {display} !important;
            }}
        </style>
        '''
    return html_template
