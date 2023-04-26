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
from .general_utils import count_files_in_folder # TODO: do it another way
from .video_audio_utilities import vid2frames, convert_image

# DEBUG_MODE = opts.data.get("deforum_debug_mode_enabled", False)
cnet = None
# number of CN model tabs to show in the deforum gui
num_of_models = 5

def find_controlnet():
    global cnet
    if cnet: return cnet
    try:
        cnet = importlib.import_module('extensions.sd-webui-controlnet.scripts.external_code', 'external_code')
    except:
        try:
            cnet = importlib.import_module('extensions-builtin.sd-webui-controlnet.scripts.external_code', 'external_code')
        except: 
            pass
    if cnet:
        print(f"\033[0;32m*Deforum ControlNet support: enabled*\033[0m")
        return True
    return None
    
def controlnet_infotext():
    return """Requires the <a style='color:SteelBlue;' target='_blank' href='https://github.com/Mikubill/sd-webui-controlnet'>ControlNet</a> extension to be installed.</p>
            <p">If Deforum crashes due to CN updates, go <a style='color:Orange;' target='_blank' href='https://github.com/Mikubill/sd-webui-controlnet/issues'>here</a> and report your problem.</p>
           """
   
def is_controlnet_enabled(controlnet_args):
    for i in range(1, num_of_models+1):
        if getattr(controlnet_args, f'cn_{i}_enabled', False):
            return True
    return False

def setup_controlnet_ui_raw():
    cnet = find_controlnet()
    cn_models = cnet.get_models()
    cn_preprocessors = cnet.get_modules()

    refresh_symbol = '\U0001f504'  # 🔄
    switch_values_symbol = '\U000021C5' # ⇅
    model_dropdowns = []
    infotext_fields = []

    def create_model_in_tab_ui(cn_id):
        with gr.Row():
            enabled = gr.Checkbox(label="Enable", value=False, interactive=True)
            pixel_perfect = gr.Checkbox(label="Pixel Perfect", value=False, visible=False, interactive=True)
            low_vram = gr.Checkbox(label="Low VRAM", value=False, visible=False, interactive=True)
            overwrite_frames = gr.Checkbox(label='Overwrite input frames', value=True, visible=False, interactive=True)
        with gr.Row(visible=False) as mod_row:
            module = gr.Dropdown(cn_preprocessors, label=f"Preprocessor", value="none", interactive=True)
            model = gr.Dropdown(cn_models, label=f"Model", value="None", interactive=True)
            refresh_models = ToolButton(value=refresh_symbol)
            refresh_models.click(refresh_all_models, model, model)
        with gr.Row(visible=False) as weight_row:
            weight = gr.Slider(label=f"Weight", value=1.0, minimum=0.0, maximum=2.0, step=.05, interactive=True)
            guidance_start =  gr.Slider(label="Guidance start", value=0.0, minimum=0.0, maximum=1.0, interactive=True)
            guidance_end =  gr.Slider(label="Guidance end", value=1.0, minimum=0.0, maximum=1.0, interactive=True)
            model_dropdowns.append(model)
        with gr.Column(visible=False) as advanced_column:
            processor_res = gr.Slider(label="Annotator resolution", value=64, minimum=64, maximum=2048, interactive=False)
            threshold_a =  gr.Slider(label="Threshold A", value=64, minimum=64, maximum=1024, interactive=False)
            threshold_b =  gr.Slider(label="Threshold B", value=64, minimum=64, maximum=1024, interactive=False)
        with gr.Row(visible=False) as vid_path_row:
            vid_path = gr.Textbox(value='', label="ControlNet Input Video/ Image Path", interactive=True)
        with gr.Row(visible=False) as mask_vid_path_row: # invisible temporarily since 26-04-23 until masks are fixed
            mask_vid_path = gr.Textbox(value='', label="ControlNet Mask Video/ Image Path", interactive=True)
        with gr.Row(visible=False) as control_mode_row:
            control_mode = gr.Radio(choices=["Balanced", "My prompt is more important", "ControlNet is more important"], value="Balanced", label="Control Mode", interactive=True)            
        with gr.Row(visible=False) as env_row:
            resize_mode = gr.Radio(choices=["Outer Fit (Shrink to Fit)", "Inner Fit (Scale to Fit)", "Just Resize"], value="Inner Fit (Scale to Fit)", label="Resize Mode", interactive=True)
        hide_output_list = [pixel_perfect,low_vram,mod_row,module,weight_row,env_row,overwrite_frames,vid_path_row,control_mode_row] # add mask_vid_path_row when masks are working again
        for cn_output in hide_output_list:
            enabled.change(fn=hide_ui_by_cn_status, inputs=enabled,outputs=cn_output)
        module.change(build_sliders, inputs=[module, pixel_perfect], outputs=[processor_res, threshold_a, threshold_b, advanced_column])
        pixel_perfect.change(build_sliders, inputs=[module, pixel_perfect], outputs=[processor_res, threshold_a, threshold_b, advanced_column])
        infotext_fields.extend([
                (module, f"ControlNet Preprocessor"),
                (model, f"ControlNet Model"),
                (weight, f"ControlNet Weight"),
        ])
        
        return {key: value for key, value in locals().items() if key in [
            "enabled", "pixel_perfect","low_vram", "module", "model", "weight",
            "guidance_start", "guidance_end", "processor_res", "threshold_a", "threshold_b", "resize_mode", "control_mode",
            "overwrite_frames", "vid_path", "mask_vid_path"
        ]}
        
    def refresh_all_models(*inputs):
        cn_models = cnet.get_models(update=True)
        dd = inputs[0]
        selected = dd if dd in cn_models else "None"
        return gr.Dropdown.update(value=selected, choices=cn_models)
    with gr.Tabs():
        model_params = {}
        for i in range(1, num_of_models+1):
            with gr.Tab(f"CN Model {i}"):
                model_params[i] = create_model_in_tab_ui(i)

                for key, value in model_params[i].items():
                    locals()[f"cn_{i}_{key}"] = value

    return locals()
            
def setup_controlnet_ui():
    if not find_controlnet():
        gr.HTML("""<a style='target='_blank' href='https://github.com/Mikubill/sd-webui-controlnet'>ControlNet not found. Please install it :)</a>""", elem_id='controlnet_not_found_html_msg')
        return {}

    try:
        return setup_controlnet_ui_raw()
    except Exception as e:
        print(f"'ControlNet UI setup failed with error: '{e}'!")
        gr.HTML(f"""
                Failed to setup ControlNet UI, check the reason in your commandline log. Please, downgrade your CN extension to <a style='color:Orange;' target='_blank' href='https://github.com/Mikubill/sd-webui-controlnet/archive/c9340671d6d59e5a79fc404f78f747f969f87374.zip'>c9340671d6d59e5a79fc404f78f747f969f87374</a> or report the problem <a style='color:Orange;' target='_blank' href='https://github.com/Mikubill/sd-webui-controlnet/issues'>here</a>.
                """, elem_id='controlnet_not_found_html_msg')
        return {}

def controlnet_component_names():
    if not find_controlnet():
        return []

    return [f'cn_{i}_{component}' for i in range(1, num_of_models+1) for component in [
        'overwrite_frames', 'vid_path', 'mask_vid_path', 'enabled',
        'low_vram', 'pixel_perfect',
        'module', 'model', 'weight', 'guidance_start', 'guidance_end',
        'processor_res', 'threshold_a', 'threshold_b', 'resize_mode', 'control_mode'
    ]]
    
def process_with_controlnet(p, args, anim_args, loop_args, controlnet_args, root, is_img2img=True, frame_idx=1):
    def read_cn_data(cn_idx):
        cn_mask_np, cn_image_np = None, None
        cn_inputframes = os.path.join(args.outdir, f'controlnet_{cn_idx}_inputframes') # set input frames folder path
        if os.path.exists(cn_inputframes):
            if count_files_in_folder(cn_inputframes) == 1:
                cn_frame_path = os.path.join(cn_inputframes, "000000001.jpg")
                print(f'Reading ControlNet *static* base frame at {cn_frame_path}')
            else:
                cn_frame_path = os.path.join(cn_inputframes, f"{frame_idx:09}.jpg")
                print(f'Reading ControlNet {cn_idx} base frame #{frame_idx} at {cn_frame_path}')
            if os.path.exists(cn_frame_path):
                cn_image_np = np.array(Image.open(cn_frame_path).convert("RGB")).astype('uint8')
        cn_maskframes = os.path.join(args.outdir, f'controlnet_{cn_idx}_maskframes') # set mask frames folder path        
        if os.path.exists(cn_maskframes):
            cn_mask_frame_path = os.path.join(args.outdir, f'controlnet_{cn_idx}_maskframes', f"{frame_idx:09}.jpg")
            if os.path.exists(cn_mask_frame_path):
                cn_mask_np = np.array(Image.open(cn_mask_frame_path).convert("RGB")).astype('uint8')
        return cn_mask_np, cn_image_np

    cnet = find_controlnet()
    cn_data = [read_cn_data(i) for i in range(1, num_of_models+1)]
    cn_inputframes_list = [os.path.join(args.outdir, f'controlnet_{i}_inputframes') for i in range(1, num_of_models+1)]

    if not any(os.path.exists(cn_inputframes) for cn_inputframes in cn_inputframes_list):
        print(f'\033[33mNeither the base nor the masking frames for ControlNet were found. Using the regular pipeline\033[0m')

    p.scripts = scripts.scripts_img2img if is_img2img else scripts.scripts_txt2img

    def create_cnu_dict(cn_args, prefix, img_np, mask_np):
        keys = [
            "enabled", "module", "model", "weight", "resize_mode", "control_mode", "low_vram","pixel_perfect",
            "processor_res", "threshold_a", "threshold_b", "guidance_start", "guidance_end"
        ]
        cnu = {k: getattr(cn_args, f"{prefix}_{k}") for k in keys}
        cnu['image'] = {'image': img_np, 'mask': mask_np} if mask_np is not None else img_np
        return cnu

    masks_np, images_np = zip(*cn_data)

    cn_units = [cnet.ControlNetUnit(**create_cnu_dict(controlnet_args, f"cn_{i+1}", img_np, mask_np))
            for i, (img_np, mask_np) in enumerate(zip(images_np, masks_np))]

    p.script_args = {"enabled": True} 
    cnet.update_cn_script_in_processing(p, cn_units, is_img2img=is_img2img, is_ui=False)

def process_controlnet_input_frames(args, anim_args, controlnet_args, video_path, mask_path, outdir_suffix, id):
    if (video_path or mask_path) and getattr(controlnet_args, f'cn_{id}_enabled'):
        frame_path = os.path.join(args.outdir, f'controlnet_{id}_{outdir_suffix}')
        os.makedirs(frame_path, exist_ok=True)
        
        # TODO: handle masks too
        accepted_image_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
        if video_path.lower().endswith(accepted_image_extensions):
            convert_image(video_path, os.path.join(frame_path, '000000001.jpg'))
            print(f"Copied CN Model {id}'s single input image to inputframes folder!")
        else:
            print(f'Unpacking ControlNet {id} {"video mask" if mask_path else "base video"}')
            print(f"Exporting Video Frames to {frame_path}...") # future todo, add an if for vid input mode to show actual extract nth param
            vid2frames(
                video_path=video_path or mask_path,
                video_in_frame_path=frame_path,
                n=1 if anim_args.animation_mode != 'Video Input' else anim_args.extract_nth_frame,
                overwrite=getattr(controlnet_args, f'cn_{id}_overwrite_frames'),
                extract_from_frame=0 if anim_args.animation_mode != 'Video Input' else anim_args.extract_from_frame,
                extract_to_frame=(anim_args.max_frames-1) if anim_args.animation_mode != 'Video Input' else anim_args.extract_to_frame,
                numeric_files_output=True
            )
            print(f"Loading {anim_args.max_frames} input frames from {frame_path} and saving video frames to {args.outdir}")
            print(f'ControlNet {id} {"video mask" if mask_path else "base video"} unpacked!')

def unpack_controlnet_vids(args, anim_args, video_args, parseq_args, loop_args, controlnet_args, animation_prompts, root):
    # this func gets called from render.py once for an entire animation run -->
    # tries to trigger an extraction of CN input frames (regular + masks) from video or image
    for i in range(1, num_of_models+1):
        vid_path = getattr(controlnet_args, f'cn_{i}_vid_path', None)
        mask_path = getattr(controlnet_args, f'cn_{i}_mask_vid_path', None)
        
        if vid_path: # Process base video, if available
            process_controlnet_input_frames(args, anim_args, controlnet_args, vid_path, None, 'inputframes', i)
        
        if mask_path: # Process mask video, if available
            process_controlnet_input_frames(args, anim_args, controlnet_args, None, mask_path, 'maskframes', i)