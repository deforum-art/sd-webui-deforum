# This helper script is responsible for ControlNet/Deforum integration
# https://github.com/Mikubill/sd-webui-controlnet — controlnet repo

import os, sys
import gradio as gr
import scripts
import modules.scripts as scrpts
from PIL import Image
import numpy as np
from modules.processing import process_images
import importlib
from .rich import console
from rich.table import Table
from rich import box
from modules import scripts
from modules.shared import opts
from .deforum_controlnet_gradio import *
from .video_audio_utilities import vid2frames

# DEBUG_MODE = opts.data.get("deforum_debug_mode_enabled", False)

cnet = None

def find_controlnet():
    global cnet
    global cnet_import_failure_count
    if cnet is not None:
        return cnet
    try:
        cnet = importlib.import_module('extensions.sd-webui-controlnet.scripts.external_code', 'external_code')
        print(f"\033[0;32m*Deforum ControlNet support: enabled*\033[0m")
        return True
    except Exception as e: # the tab will be disactivated anyway, so we don't need the error message
        return None

svgsupport = False
try:
    import io
    import base64
    from svglib.svglib import svg2rlg
    from reportlab.graphics import renderPM
    svgsupport = True
except ImportError:
    pass

def setup_controlnet_ui_raw():
    cnet = find_controlnet()
    cn_models = cnet.get_models()
    max_models = opts.data.get("control_net_max_models_num", 1)
    cn_preprocessors = [ # since cn preprocessors don't seem to be provided in the API rn, hardcode the names list
        "none",
        "canny",
        "depth",
        "depth_leres",
        "hed",
        "mlsd",
        "normal_map",
        "openpose",
        "openpose_hand",
        "clip_vision",
        "color",
        "pidinet",
        "scribble",
        "fake_scribble",
        "segmentation",
        "binary",
    ]

    refresh_symbol = '\U0001f504'  # 🔄
    switch_values_symbol = '\U000021C5' # ⇅
    model_dropdowns = []
    infotext_fields = []

    def refresh_all_models(*inputs):
        cn_models = cnet.get_models(update=True)
        dd = inputs[0]
        selected = dd if dd in cn_models else "None"
    with gr.Tabs():
        with gr.Tab(f"ControlNet 1"):
            with gr.Row():
                cn_1_enabled = gr.Checkbox(label='Enable', value=False, interactive=True)
                cn_1_guess_mode = gr.Checkbox(label='Guess Mode', value=False, visible=False, interactive=True)
                cn_1_invert_image = gr.Checkbox(label='Invert colors', value=False, visible=False, interactive=True)
                cn_1_rgbbgr_mode = gr.Checkbox(label='RGB to BGR', value=False, visible=False, interactive=True)
                cn_1_low_vram = gr.Checkbox(label='Low VRAM', value=False, visible=False, interactive=True)

            with gr.Row(visible=False) as cn_1_mod_row:
                cn_1_module = gr.Dropdown(cn_preprocessors, label=f"Preprocessor", value="none", interactive=True)
                cn_1_model = gr.Dropdown(cn_models, label=f"Model", value="None", interactive=True)
                refresh_models = ToolButton(value=refresh_symbol)
                refresh_models.click(refresh_all_models, cn_1_model, cn_1_model)
            with gr.Row(visible=False) as cn_1_weight_row:
                cn_1_weight = gr.Slider(label=f"Weight", value=1.0, minimum=0.0, maximum=2.0, step=.05, interactive=True)
                cn_1_guidance_start =  gr.Slider(label="Guidance start", value=0.0, minimum=0.0, maximum=1.0, interactive=True)
                cn_1_guidance_end =  gr.Slider(label="Guidance end", value=1.0, minimum=0.0, maximum=1.0, interactive=True)
                model_dropdowns.append(cn_1_model)
            with gr.Column(visible=False) as cn_1_advanced:
                cn_1_processor_res = gr.Slider(label="Annotator resolution", value=64, minimum=64, maximum=2048, interactive=False)
                cn_1_threshold_a =  gr.Slider(label="Threshold A", value=64, minimum=64, maximum=1024, interactive=False)
                cn_1_threshold_b =  gr.Slider(label="Threshold B", value=64, minimum=64, maximum=1024, interactive=False)
            
            cn_1_module.change(build_sliders, inputs=[cn_1_module], outputs=[cn_1_processor_res, cn_1_threshold_a, cn_1_threshold_b, cn_1_advanced])
                
            infotext_fields.extend([
                (cn_1_module, f"ControlNet Preprocessor"),
                (cn_1_model, f"ControlNet Model"),
                (cn_1_weight, f"ControlNet Weight"),
            ])

            with gr.Row(visible=False) as cn_1_env_row:
                cn_1_resize_mode = gr.Radio(choices=["Envelope (Outer Fit)", "Scale to Fit (Inner Fit)", "Just Resize"], value="Scale to Fit (Inner Fit)", label="Resize Mode", interactive=True)
            
            with gr.Row(visible=False) as cn_1_vid_settings_row:
                cn_1_overwrite_frames = gr.Checkbox(label='Overwrite input frames', value=True, interactive=True)
                cn_1_vid_path = gr.Textbox(value='', label="ControlNet Input Video Path", interactive=True)
                cn_1_mask_vid_path = gr.Textbox(value='', label="ControlNet Mask Video Path", interactive=True)

            cn_1_input_video_chosen_file = gr.File(label="ControlNet Video Input", interactive=True, file_count="single", file_types=["video"], elem_id="controlnet_input_video_chosen_file", visible=False)
            cn_1_input_video_mask_chosen_file = gr.File(label="ControlNet Video Mask Input", interactive=True, file_count="single", file_types=["video"], elem_id="controlnet_input_video_mask_chosen_file", visible=False)
           
            cn_1_hide_output_list = [cn_1_guess_mode,cn_1_invert_image,cn_1_rgbbgr_mode,cn_1_low_vram,cn_1_mod_row,cn_1_weight_row,cn_1_env_row,cn_1_vid_settings_row,cn_1_input_video_chosen_file,cn_1_input_video_mask_chosen_file] 
            for cn_output in cn_1_hide_output_list:
                cn_1_enabled.change(fn=hide_ui_by_cn_status, inputs=cn_1_enabled,outputs=cn_output)
        with gr.Tab(f"ControlNet 2"):
            with gr.Row():
                cn_2_enabled = gr.Checkbox(label='Enable', value=False, interactive=True)
                cn_2_guess_mode = gr.Checkbox(label='Guess Mode', value=False, visible=False, interactive=True)
                cn_2_invert_image = gr.Checkbox(label='Invert colors', value=False, visible=False, interactive=True)
                cn_2_rgbbgr_mode = gr.Checkbox(label='RGB to BGR', value=False, visible=False, interactive=True)
                cn_2_low_vram = gr.Checkbox(label='Low VRAM', value=False, visible=False, interactive=True)
            with gr.Row(visible=False) as cn_2_mod_row:
                cn_2_module = gr.Dropdown(cn_preprocessors, label=f"Preprocessor", value="none", interactive=True)
                cn_2_model = gr.Dropdown(cn_models, label=f"Model", value="None", interactive=True)
                refresh_models = ToolButton(value=refresh_symbol)
                refresh_models.click(refresh_all_models, cn_2_model, cn_2_model)
            with gr.Row(visible=False) as cn_2_weight_row:
                cn_2_weight = gr.Slider(label=f"Weight", value=1.0, minimum=0.0, maximum=2.0, step=.05, interactive=True)
                cn_2_guidance_start =  gr.Slider(label="Guidance start", value=0.0, minimum=0.0, maximum=1.0, interactive=True)
                cn_2_guidance_end =  gr.Slider(label="Guidance end", value=1.0, minimum=0.0, maximum=1.0, interactive=True)
                model_dropdowns.append(cn_2_model)
            with gr.Column(visible=False) as cn_2_advanced:
                cn_2_processor_res = gr.Slider(label="Annotator resolution", value=64, minimum=64, maximum=2048, interactive=False)
                cn_2_threshold_a =  gr.Slider(label="Threshold A", value=64, minimum=64, maximum=1024, interactive=False)
                cn_2_threshold_b =  gr.Slider(label="Threshold B", value=64, minimum=64, maximum=1024, interactive=False)
            
            cn_2_module.change(build_sliders, inputs=[cn_2_module], outputs=[cn_2_processor_res, cn_2_threshold_a, cn_2_threshold_b, cn_2_advanced])
                
            infotext_fields.extend([
                (cn_2_module, f"ControlNet Preprocessor"),
                (cn_2_model, f"ControlNet Model"),
                (cn_2_weight, f"ControlNet Weight"),
            ])

            with gr.Row(visible=False) as cn_2_env_row:
                cn_2_resize_mode = gr.Radio(choices=["Envelope (Outer Fit)", "Scale to Fit (Inner Fit)", "Just Resize"], value="Scale to Fit (Inner Fit)", label="Resize Mode", interactive=True)
            
            with gr.Row(visible=False) as cn_2_vid_settings_row:
                cn_2_overwrite_frames = gr.Checkbox(label='Overwrite input frames', value=True, interactive=True)
                cn_2_vid_path = gr.Textbox(value='', label="ControlNet Input Video Path", interactive=True)
                cn_2_mask_vid_path = gr.Textbox(value='', label="ControlNet Mask Video Path", interactive=True)

            cn_2_input_video_chosen_file = gr.File(label="ControlNet Video Input", interactive=True, file_count="single", file_types=["video"], elem_id="controlnet_input_video_chosen_file", visible=False)
            cn_2_input_video_mask_chosen_file = gr.File(label="ControlNet Video Mask Input", interactive=True, file_count="single", file_types=["video"], elem_id="controlnet_input_video_mask_chosen_file", visible=False)
           
            cn_2_hide_output_list = [cn_2_guess_mode,cn_2_invert_image,cn_2_rgbbgr_mode,cn_2_low_vram,cn_2_mod_row,cn_2_weight_row,cn_2_env_row,cn_2_vid_settings_row,cn_2_input_video_chosen_file,cn_2_input_video_mask_chosen_file] 
            for cn_output in cn_2_hide_output_list:
                cn_2_enabled.change(fn=hide_ui_by_cn_status, inputs=cn_2_enabled,outputs=cn_output)
            
    return locals()
            
def setup_controlnet_ui():
    if not find_controlnet():
        gr.HTML("""<a style='target='_blank' href='https://github.com/Mikubill/sd-webui-controlnet'>ControlNet not found. Please install it :)</a>""", elem_id='controlnet_not_found_html_msg')
        return {}

    try:
        return setup_controlnet_ui_raw()
    except Exception as e:
        print(f"'ControlNet UI setup failed due to '{e}'!")
        gr.HTML(f"""
                Failed to setup ControlNet UI, check the reason in your commandline log. Please, downgrade your CN extension to <a style='color:Orange;' target='_blank' href='https://github.com/Mikubill/sd-webui-controlnet/archive/c9340671d6d59e5a79fc404f78f747f969f87374.zip'>c9340671d6d59e5a79fc404f78f747f969f87374</a> or report the problem <a style='color:Orange;' target='_blank' href='https://github.com/Mikubill/sd-webui-controlnet/issues'>here</a>.
                """, elem_id='controlnet_not_found_html_msg')
        return {}

def controlnet_component_names():
    if not find_controlnet():
        return []

    controlnet_args_names = str(r'''cn_1_input_video_chosen_file, cn_1_input_video_mask_chosen_file,
        cn_1_overwrite_frames,cn_1_vid_path,cn_1_mask_vid_path,
        cn_1_enabled, cn_1_guess_mode, cn_1_invert_image, cn_1_rgbbgr_mode, cn_1_low_vram,
        cn_1_module, cn_1_model,
        cn_1_weight, cn_1_guidance_start, cn_1_guidance_end,
        cn_1_processor_res, 
        cn_1_threshold_a, cn_1_threshold_b, cn_1_resize_mode,
        cn_2_input_video_chosen_file, cn_2_input_video_mask_chosen_file,
        cn_2_overwrite_frames,cn_2_vid_path,cn_2_mask_vid_path,
        cn_2_enabled, cn_2_guess_mode, cn_2_invert_image, cn_2_rgbbgr_mode, cn_2_low_vram,
        cn_2_module, cn_2_model,
        cn_2_weight, cn_2_guidance_start, cn_2_guidance_end,
        cn_2_processor_res, 
        cn_2_threshold_a, cn_2_threshold_b, cn_2_resize_mode'''
    ).replace("\n", "").replace("\r", "").replace(" ", "").split(',')
    
    return controlnet_args_names

def controlnet_infotext():
    return """Requires the <a style='color:SteelBlue;' target='_blank' href='https://github.com/Mikubill/sd-webui-controlnet'>ControlNet</a> extension to be installed.</p>
            <p">If Deforum crashes due to CN updates, go <a style='color:Orange;' target='_blank' href='https://github.com/Mikubill/sd-webui-controlnet/issues'>here</a> and report your problem.</p>
           """
           
def is_controlnet_enabled(controlnet_args):
    # return 'cn_1_enabled' in vars(controlnet_args) and controlnet_args.cn_1_enabled
    if 'cn_1_enabled' in vars(controlnet_args) and controlnet_args.cn_1_enabled:
        return True
    elif 'cn_2_enabled' in vars(controlnet_args) and controlnet_args.cn_2_enabled:
        return True
    else:
        return False

def process_with_controlnet(p, args, anim_args, loop_args, controlnet_args, root, is_img2img=True, frame_idx=1):
    def read_cn_data(cn_idx):
        cn_mask_np, cn_image_np = None, None
        cn_inputframes = os.path.join(args.outdir, f'controlnet_{cn_idx}_inputframes')
        if os.path.exists(cn_inputframes):
            cn_frame_path = os.path.join(cn_inputframes, f"{frame_idx:09}.jpg")
            cn_mask_frame_path = os.path.join(args.outdir, f'controlnet_{cn_idx}_maskframes', f"{frame_idx:09}.jpg")

            print(f'Reading ControlNet {cn_idx} base frame {frame_idx} at {cn_frame_path}')
            print(f'Reading ControlNet {cn_idx} mask frame {frame_idx} at {cn_mask_frame_path}')

            if os.path.exists(cn_frame_path):
                cn_image_np = np.array(Image.open(cn_frame_path).convert("RGB")).astype('uint8')

            if os.path.exists(cn_mask_frame_path):
                cn_mask_np = np.array(Image.open(cn_mask_frame_path).convert("RGB")).astype('uint8')
        return cn_mask_np, cn_image_np

    cnet = find_controlnet()
    cn_1_mask_np, cn_1_image_np = read_cn_data(1)
    cn_2_mask_np, cn_2_image_np = read_cn_data(2)

    cn_1_inputframes = os.path.join(args.outdir, 'controlnet_1_inputframes')
    cn_2_inputframes = os.path.join(args.outdir, 'controlnet_2_inputframes')

    if not os.path.exists(cn_1_inputframes) and not os.path.exists(cn_2_inputframes):
        print(f'\033[33mNeither the base nor the masking frames for ControlNet were found. Using the regular pipeline\033[0m')

    p.scripts = scripts.scripts_img2img if is_img2img else scripts.scripts_txt2img

    def create_cnu_dict(cn_args, prefix, img_np, mask_np):
        keys = [
            "enabled", "module", "model", "weight", "invert_image",
            "guess_mode", "resize_mode", "rgbbgr_mode", "low_vram",
            "processor_res", "threshold_a", "threshold_b", "guidance_start", "guidance_end"
        ]
        cnu = {k: getattr(cn_args, f"{prefix}_{k}") for k in keys}
        cnu['image'] = {'image': img_np, 'mask': mask_np} if mask_np is not None else img_np
        return cnu

    images_np = [cn_1_image_np, cn_2_image_np]
    masks_np = [cn_1_mask_np, cn_2_mask_np]
    prefixes = ["cn_1", "cn_2"]

    cn_units = [
        cnet.ControlNetUnit(**create_cnu_dict(controlnet_args, prefix, img_np, mask_np))
        for prefix, img_np, mask_np in zip(prefixes, images_np, masks_np)
    ]

    p.script_args = {"enabled": True}
    cnet.update_cn_script_in_processing(p, cn_units, is_img2img=is_img2img, is_ui=False)

def process_controlnet_video(args, anim_args, controlnet_args, video_path, mask_path, outdir_suffix, id):
    if (video_path or mask_path) and getattr(controlnet_args, f'cn_{id}_enabled'):
        print(f'Unpacking ControlNet {id} {"video mask" if mask_path else "base video"}')
        frame_path = os.path.join(args.outdir, f'controlnet_{id}_{outdir_suffix}')
        os.makedirs(frame_path, exist_ok=True)

        print(f"Exporting Video Frames (1 every {anim_args.extract_nth_frame}) frames to {frame_path}...")
        vid2frames(
            video_path=video_path or mask_path,
            video_in_frame_path=frame_path,
            n=anim_args.extract_nth_frame,
            overwrite=getattr(controlnet_args, f'cn_{id}_overwrite_frames'),
            extract_from_frame=anim_args.extract_from_frame,
            extract_to_frame=anim_args.extract_to_frame,
            numeric_files_output=True
        )
        print(f"Loading {anim_args.max_frames} input frames from {frame_path} and saving video frames to {args.outdir}")
        print(f'ControlNet {id} {"video mask" if mask_path else "base video"} unpacked!')

def unpack_controlnet_vids(args, anim_args, video_args, parseq_args, loop_args, controlnet_args, animation_prompts, root):
    for i in range(1, 3):
        process_controlnet_video(
            args, anim_args, controlnet_args,
            getattr(controlnet_args, f'cn_{i}_vid_path') or getattr(controlnet_args, f'cn_{i}_input_video_chosen_file', None) and controlnet_args[f'cn_{i}_input_video_chosen_file'].name,
            getattr(controlnet_args, f'cn_{i}_mask_vid_path') or getattr(controlnet_args, f'cn_{i}_input_video_mask_chosen_file', None) and controlnet_args[f'cn_{i}_input_video_mask_chosen_file'].name,
            'inputframes' if not getattr(controlnet_args, f'cn_{i}_mask_vid_path') else 'maskframes',
            i
        )

def hide_ui_by_cn_status(choice):
    return gr.update(visible=True) if choice else gr.update(visible=False)