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

    def create_model_in_tab_ui(cn_id):
        with gr.Row():
            enabled = gr.Checkbox(label="Enable", value=False, interactive=True)
            guess_mode = gr.Checkbox(label="Guess Mode", value=False, visible=False, interactive=True)
            invert_image = gr.Checkbox(label="Invert colors", value=False, visible=False, interactive=True)
            rgbbgr_mode = gr.Checkbox(label="RGB to BGR", value=False, visible=False, interactive=True)
            low_vram = gr.Checkbox(label="Low VRAM", value=False, visible=False, interactive=True)
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
        with gr.Row(visible=False) as env_row:
            resize_mode = gr.Radio(choices=["Envelope (Outer Fit)", "Scale to Fit (Inner Fit)", "Just Resize"], value="Scale to Fit (Inner Fit)", label="Resize Mode", interactive=True)
        with gr.Row(visible=False) as vid_settings_row:
            overwrite_frames = gr.Checkbox(label='Overwrite input frames', value=True, interactive=True)
            vid_path = gr.Textbox(value='', label="ControlNet Input Video Path", interactive=True)
            mask_vid_path = gr.Textbox(value='', label="ControlNet Mask Video Path", interactive=True)
        input_video_chosen_file = gr.File(label="ControlNet Video Input", interactive=True, file_count="single", file_types=["video"], elem_id="controlnet_input_video_chosen_file", visible=False)
        input_video_mask_chosen_file = gr.File(label="ControlNet Video Mask Input", interactive=True, file_count="single", file_types=["video"], elem_id="controlnet_input_video_mask_chosen_file", visible=False)
        hide_output_list = [guess_mode,invert_image,rgbbgr_mode,low_vram,mod_row,module,weight_row,env_row,vid_settings_row,input_video_chosen_file,input_video_mask_chosen_file, advanced_column] 
        for cn_output in hide_output_list:
            enabled.change(fn=hide_ui_by_cn_status, inputs=enabled,outputs=cn_output)
        module.change(build_sliders, inputs=[module], outputs=[processor_res, threshold_a, threshold_b, advanced_column])
        infotext_fields.extend([
                (module, f"ControlNet Preprocessor"),
                (model, f"ControlNet Model"),
                (weight, f"ControlNet Weight"),
        ])
        
        return {key: value for key, value in locals().items() if key in [
            "enabled", "guess_mode", "invert_image", "rgbbgr_mode", "low_vram", "module", "model", "weight",
            "guidance_start", "guidance_end", "processor_res", "threshold_a", "threshold_b", "resize_mode",
            "overwrite_frames", "vid_path", "mask_vid_path", "input_video_chosen_file", "input_video_mask_chosen_file"
        ]}
        
    def refresh_all_models(*inputs):
        cn_models = cnet.get_models(update=True)
        dd = inputs[0]
        selected = dd if dd in cn_models else "None"
        return gr.Dropdown.update(value=selected, choices=cn_models)
    with gr.Tabs():
        with gr.Tab(f"CN Model 1"):
            model_params = {}
            model_params[1] = create_model_in_tab_ui(1)

            cn_1_enabled = model_params[1]["enabled"]
            cn_1_guess_mode = model_params[1]["guess_mode"]
            cn_1_invert_image = model_params[1]["invert_image"]
            cn_1_rgbbgr_mode = model_params[1]["rgbbgr_mode"]
            cn_1_low_vram = model_params[1]["low_vram"]
            cn_1_module = model_params[1]["module"]
            cn_1_model = model_params[1]["model"]
            cn_1_weight = model_params[1]["weight"]
            cn_1_guidance_start = model_params[1]["guidance_start"]
            cn_1_guidance_end = model_params[1]["guidance_end"]
            cn_1_processor_res = model_params[1]["processor_res"]
            cn_1_threshold_a = model_params[1]["threshold_a"]
            cn_1_threshold_b = model_params[1]["threshold_b"]
            cn_1_resize_mode = model_params[1]["resize_mode"]
            cn_1_overwrite_frames = model_params[1]["overwrite_frames"]
            cn_1_vid_path = model_params[1]["vid_path"]
            cn_1_mask_vid_path = model_params[1]["mask_vid_path"]
            cn_1_input_video_chosen_file = model_params[1]["input_video_chosen_file"]
            cn_1_input_video_mask_chosen_file = model_params[1]["input_video_mask_chosen_file"]
                
        with gr.Tab(f"CN Model 2"):
            model_params = {}
            model_params[2] = create_model_in_tab_ui(2)

            cn_2_enabled = model_params[2]["enabled"]
            cn_2_guess_mode = model_params[2]["guess_mode"]
            cn_2_invert_image = model_params[2]["invert_image"]
            cn_2_rgbbgr_mode = model_params[2]["rgbbgr_mode"]
            cn_2_low_vram = model_params[2]["low_vram"]
            cn_2_module = model_params[2]["module"]
            cn_2_model = model_params[2]["model"]
            cn_2_weight = model_params[2]["weight"]
            cn_2_guidance_start = model_params[2]["guidance_start"]
            cn_2_guidance_end = model_params[2]["guidance_end"]
            cn_2_processor_res = model_params[2]["processor_res"]
            cn_2_threshold_a = model_params[2]["threshold_a"]
            cn_2_threshold_b = model_params[2]["threshold_b"]
            cn_2_resize_mode = model_params[2]["resize_mode"]
            cn_2_overwrite_frames = model_params[2]["overwrite_frames"]
            cn_2_vid_path = model_params[2]["vid_path"]
            cn_2_mask_vid_path = model_params[2]["mask_vid_path"]
            cn_2_input_video_chosen_file = model_params[2]["input_video_chosen_file"]
            cn_2_input_video_mask_chosen_file = model_params[2]["input_video_mask_chosen_file"]

        with gr.Tab(f"CN Model 3"):
            model_params = {}
            model_params[3] = create_model_in_tab_ui(3)

            cn_3_enabled = model_params[3]["enabled"]
            cn_3_guess_mode = model_params[3]["guess_mode"]
            cn_3_invert_image = model_params[3]["invert_image"]
            cn_3_rgbbgr_mode = model_params[3]["rgbbgr_mode"]
            cn_3_low_vram = model_params[3]["low_vram"]
            cn_3_module = model_params[3]["module"]
            cn_3_model = model_params[3]["model"]
            cn_3_weight = model_params[3]["weight"]
            cn_3_guidance_start = model_params[3]["guidance_start"]
            cn_3_guidance_end = model_params[3]["guidance_end"]
            cn_3_processor_res = model_params[3]["processor_res"]
            cn_3_threshold_a = model_params[3]["threshold_a"]
            cn_3_threshold_b = model_params[3]["threshold_b"]
            cn_3_resize_mode = model_params[3]["resize_mode"]
            cn_3_overwrite_frames = model_params[3]["overwrite_frames"]
            cn_3_vid_path = model_params[3]["vid_path"]
            cn_3_mask_vid_path = model_params[3]["mask_vid_path"]
            cn_3_input_video_chosen_file = model_params[3]["input_video_chosen_file"]
            cn_3_input_video_mask_chosen_file = model_params[3]["input_video_mask_chosen_file"]
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

    prefix_list = ['cn_1_', 'cn_2_', 'cn_3_']
    arg_names = []
    for prefix in prefix_list:
        arg_names.extend([f'{prefix}input_video_chosen_file',
                          f'{prefix}input_video_mask_chosen_file',
                          f'{prefix}overwrite_frames',
                          f'{prefix}vid_path',
                          f'{prefix}mask_vid_path',
                          f'{prefix}enabled',
                          f'{prefix}guess_mode',
                          f'{prefix}invert_image',
                          f'{prefix}rgbbgr_mode',
                          f'{prefix}low_vram',
                          f'{prefix}module',
                          f'{prefix}model',
                          f'{prefix}weight',
                          f'{prefix}guidance_start',
                          f'{prefix}guidance_end',
                          f'{prefix}processor_res',
                          f'{prefix}threshold_a',
                          f'{prefix}threshold_b',
                          f'{prefix}resize_mode'])
    return arg_names

def controlnet_infotext():
    return """Requires the <a style='color:SteelBlue;' target='_blank' href='https://github.com/Mikubill/sd-webui-controlnet'>ControlNet</a> extension to be installed.</p>
            <p">If Deforum crashes due to CN updates, go <a style='color:Orange;' target='_blank' href='https://github.com/Mikubill/sd-webui-controlnet/issues'>here</a> and report your problem.</p>
           """
           
def is_controlnet_enabled(controlnet_args):
    for i in range(1, 4):
        if getattr(controlnet_args, f'cn_{i}_enabled', False):
            return True
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
    cn_3_mask_np, cn_3_image_np = read_cn_data(3)

    cn_1_inputframes = os.path.join(args.outdir, 'controlnet_1_inputframes')
    cn_2_inputframes = os.path.join(args.outdir, 'controlnet_2_inputframes')
    cn_3_inputframes = os.path.join(args.outdir, 'controlnet_3_inputframes')

    if not os.path.exists(cn_1_inputframes) and not os.path.exists(cn_2_inputframes) and not os.path.exists(cn_3_inputframes):
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

    images_np = [cn_1_image_np, cn_2_image_np, cn_3_image_np]
    masks_np = [cn_1_mask_np, cn_2_mask_np, cn_3_mask_np]
    prefixes = ["cn_1", "cn_2", "cn_3"]

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
    for i in range(1, 4):
        vid_path = getattr(controlnet_args, f'cn_{i}_vid_path', None)
        vid_chosen_file = getattr(controlnet_args, f'cn_{i}_input_video_chosen_file', None)
        vid_name = None
        if vid_chosen_file is not None:
            vid_name = getattr(getattr(controlnet_args, f'cn_{i}_input_video_chosen_file'), 'name', None)
        
        mask_path = getattr(controlnet_args, f'cn_{i}_mask_vid_path', None)
        mask_chosen_file = getattr(controlnet_args, f'cn_{i}_input_video_mask_chosen_file', None)
        mask_name = None
        if mask_chosen_file is not None:
            mask_name = getattr(getattr(controlnet_args, f'cn_{i}_input_video_mask_chosen_file'), 'name', None)

        process_controlnet_video(
            args, anim_args, controlnet_args,
            vid_path or vid_name,
            mask_path or mask_name,
            'inputframes' if not mask_path else 'maskframes',
            i
        )

def hide_ui_by_cn_status(choice):
    return gr.update(visible=True) if choice else gr.update(visible=False)