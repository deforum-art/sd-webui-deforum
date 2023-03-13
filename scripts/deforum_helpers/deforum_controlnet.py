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
    except Exception as e:
        # the tab will be disactivated anyway, so we don't need the error message
        return None

gradio_compat = True
try:
    from distutils.version import LooseVersion
    from importlib_metadata import version
    if LooseVersion(version("gradio")) < LooseVersion("3.10"):
        gradio_compat = False
except ImportError:
    pass

# svgsupports
svgsupport = False
try:
    import io
    import base64
    from svglib.svglib import svg2rlg
    from reportlab.graphics import renderPM
    svgsupport = True
except ImportError:
    pass

def ControlnetArgs():
    controlnet_enabled = False
    controlnet_guess_mode = False
    controlnet_rgbbgr_mode = False
    controlnet_lowvram = False
    controlnet_module = "none"
    controlnet_model = "None"
    controlnet_weight = 1.0
    controlnet_guidance_strength = 1.0
    blendFactorMax = "0:(0.35)"
    blendFactorSlope = "0:(0.25)"
    tweening_frames_schedule = "0:(20)"
    color_correction_factor = "0:(0.075)"
    return locals()

def setup_controlnet_ui_raw():
    cnet = find_controlnet()
    cn_models = cnet.get_models()
    # since cn preprocessors don't seem to be provided in the API rn, hardcode the names list
    cn_preprocessors = [
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

    # Already under an accordion
    refresh_symbol = '\U0001f504'  # 🔄
    switch_values_symbol = '\U000021C5' # ⇅
    model_dropdowns = []
    infotext_fields = []
    # Main part
    class ToolButton(gr.Button, gr.components.FormComponent):
        """Small button with single emoji as text, fits inside gradio forms"""

        def __init__(self, **kwargs):
            super().__init__(variant="tool", **kwargs)

        def get_block_name(self):
            return "button"

    # # Copying the main ControlNet widgets while getting rid of static elements such as the scribble pad
    with gr.Row():
        controlnet_enabled = gr.Checkbox(label='Enable', value=False, interactive=True)
        controlnet_guess_mode = gr.Checkbox(label='Guess Mode', value=False, visible=False, interactive=True)
        controlnet_invert_image = gr.Checkbox(label='Invert colors', value=False, visible=False, interactive=True)
        controlnet_rgbbgr_mode = gr.Checkbox(label='RGB to BGR', value=False, visible=False, interactive=True)
        controlnet_lowvram = gr.Checkbox(label='Low VRAM', value=False, visible=False, interactive=True)

    def refresh_all_models(*inputs):
        cn_models = cnet.get_models(update=True)
        dd = inputs[0]
        selected = dd if dd in cn_models else "None"
        return gr.Dropdown.update(value=selected, choices=cn_models)

    with gr.Row(visible=False) as cn_mod_row:
        controlnet_module = gr.Dropdown(cn_preprocessors, label=f"Preprocessor", value="none", interactive=True)
        controlnet_model = gr.Dropdown(cn_models, label=f"Model", value="None", interactive=True)
        refresh_models = ToolButton(value=refresh_symbol)
        refresh_models.click(refresh_all_models, controlnet_model, controlnet_model)
        #ctrls += (refresh_models, )
    with gr.Row(visible=False) as cn_weight_row:
        controlnet_weight = gr.Slider(label=f"Weight", value=1.0, minimum=0.0, maximum=2.0, step=.05, interactive=True)
        controlnet_guidance_start =  gr.Slider(label="Guidance start", value=0.0, minimum=0.0, maximum=1.0, interactive=True)
        controlnet_guidance_end =  gr.Slider(label="Guidance end", value=1.0, minimum=0.0, maximum=1.0, interactive=True)
        #ctrls += (controlnet_module, controlnet_model, controlnet_weight,)
        model_dropdowns.append(controlnet_model)
  
    # advanced options    
    controlnet_advanced = gr.Column(visible=False)
    with controlnet_advanced:
        controlnet_processor_res = gr.Slider(label="Annotator resolution", value=64, minimum=64, maximum=2048, interactive=False)
        controlnet_threshold_a =  gr.Slider(label="Threshold A", value=64, minimum=64, maximum=1024, interactive=False)
        controlnet_threshold_b =  gr.Slider(label="Threshold B", value=64, minimum=64, maximum=1024, interactive=False)
    
    if gradio_compat:    
        controlnet_module.change(build_sliders, inputs=[controlnet_module], outputs=[controlnet_processor_res, controlnet_threshold_a, controlnet_threshold_b, controlnet_advanced])
        
    infotext_fields.extend([
        (controlnet_module, f"ControlNet Preprocessor"),
        (controlnet_model, f"ControlNet Model"),
        (controlnet_weight, f"ControlNet Weight"),
    ])

    with gr.Row(visible=False) as cn_env_row:
        controlnet_resize_mode = gr.Radio(choices=["Envelope (Outer Fit)", "Scale to Fit (Inner Fit)", "Just Resize"], value="Scale to Fit (Inner Fit)", label="Resize Mode", interactive=True)
    
    with gr.Row(visible=False) as cn_vid_settings_row:
        controlnet_overwrite_frames = gr.Checkbox(label='Overwrite input frames', value=True, interactive=True)
        controlnet_vid_path = gr.Textbox(value='', label="ControlNet Input Video Path", interactive=True)
        controlnet_mask_vid_path = gr.Textbox(value='', label="ControlNet Mask Video Path", interactive=True)

    # Video input to be fed into ControlNet
    #input_video_url = gr.Textbox(source='upload', type='numpy', tool='sketch') # TODO
    controlnet_input_video_chosen_file = gr.File(label="ControlNet Video Input", interactive=True, file_count="single", file_types=["video"], elem_id="controlnet_input_video_chosen_file", visible=False)
    controlnet_input_video_mask_chosen_file = gr.File(label="ControlNet Video Mask Input", interactive=True, file_count="single", file_types=["video"], elem_id="controlnet_input_video_mask_chosen_file", visible=False)
   
    cn_hide_output_list = [controlnet_guess_mode,controlnet_invert_image,controlnet_rgbbgr_mode,controlnet_lowvram,cn_mod_row,cn_weight_row,cn_env_row,cn_vid_settings_row,controlnet_input_video_chosen_file,controlnet_input_video_mask_chosen_file] 
    for cn_output in cn_hide_output_list:
        controlnet_enabled.change(fn=hide_ui_by_cn_status, inputs=controlnet_enabled,outputs=cn_output)
        
    return locals()

            
def setup_controlnet_ui():
    if not find_controlnet():
        gr.HTML("""
                <a style='target='_blank' href='https://github.com/Mikubill/sd-webui-controlnet'>ControlNet not found. Please install it :)</a>
                """, elem_id='controlnet_not_found_html_msg')
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

    controlnet_args_names = str(r'''controlnet_input_video_chosen_file, controlnet_input_video_mask_chosen_file,
controlnet_overwrite_frames,controlnet_vid_path,controlnet_mask_vid_path,
controlnet_enabled, controlnet_guess_mode, controlnet_invert_image, controlnet_rgbbgr_mode, controlnet_lowvram,
controlnet_module, controlnet_model,
controlnet_weight, controlnet_guidance_start, controlnet_guidance_end,
controlnet_processor_res, 
controlnet_threshold_a, controlnet_threshold_b, controlnet_resize_mode'''
    ).replace("\n", "").replace("\r", "").replace(" ", "").split(',')
    
    return controlnet_args_names

def controlnet_infotext():
    return """Requires the <a style='color:SteelBlue;' target='_blank' href='https://github.com/Mikubill/sd-webui-controlnet'>ControlNet</a> extension to be installed.</p>
            <p style="margin-top:0.2em">
                *Work In Progress*. All params below are going to be keyframable at some point. If you want to speedup the integration, *especially* if you want to bring new features sooner, <a style='color:Violet;' target='_blank' href='https://github.com/deforum-art/deforum-for-automatic1111-webui/'>join Deforum's development</a>. &#128521;
            </p>
            <p">
                If you previously downgraded the CN extension to use it in Deforum, upgrade it to the latest version for the API communication to work; also, you don't need to store your CN models in deforum/models anymore. We got complaints that fp16 models sometimes don't work for an unknown reason, so use <a style='color:DarkGreen;' target='_blank' href='https://huggingface.co/lllyasviel/ControlNet/tree/main/models'>the full precision ones</a> until it's fixed. Note that CN's API may breakingly change at any time. The recommended CN version is <a style='color:Orange;' target='_blank' href='https://github.com/Mikubill/sd-webui-controlnet/archive/c9340671d6d59e5a79fc404f78f747f969f87374.zip'>c9340671d6d59e5a79fc404f78f747f969f87374</a>. If Deforum crashes due to CN updates, go <a style='color:Orange;' target='_blank' href='https://github.com/Mikubill/sd-webui-controlnet/issues'>here</a> and report your problem.
            </p>
           """

def is_controlnet_enabled(controlnet_args):
    return 'controlnet_enabled' in vars(controlnet_args) and controlnet_args.controlnet_enabled

def process_with_controlnet(p, args, anim_args, loop_args, controlnet_args, root, is_img2img = True, frame_idx = 1):
    cnet = find_controlnet()

    controlnet_frame_path = os.path.join(args.outdir, 'controlnet_inputframes', f"{frame_idx:09}.jpg")
    controlnet_mask_frame_path = os.path.join(args.outdir, 'controlnet_maskframes', f"{frame_idx:09}.jpg")

    print(f'Reading ControlNet base frame {frame_idx} at {controlnet_frame_path}')
    print(f'Reading ControlNet mask frame {frame_idx} at {controlnet_mask_frame_path}')

    cn_mask_np = None
    cn_image_np = None

    if not os.path.exists(controlnet_frame_path) and not os.path.exists(controlnet_mask_frame_path):
        print(f'\033[33mNeither the base nor the masking frames for ControlNet were found. Using the regular pipeline\033[0m')
        return
    
    if os.path.exists(controlnet_frame_path):
        cn_image_np = np.array(Image.open(controlnet_frame_path).convert("RGB")).astype('uint8')
    
    if os.path.exists(controlnet_mask_frame_path):
        cn_mask_np = np.array(Image.open(controlnet_mask_frame_path).convert("RGB")).astype('uint8')
    
    table = Table(title="ControlNet params",padding=0, box=box.ROUNDED)

    # TODO: auto infer the names and the values for the table
    field_names = []
    field_names += ["module", "model", "weight", "inv", "guide_start", "guide_end", "guess", "resize", "rgb_bgr", "proc res", "thr a", "thr b"]
    for field_name in field_names:
        table.add_column(field_name, justify="center")
    
    cn_model_name = str(controlnet_args.controlnet_model)

    rows = []
    rows += [controlnet_args.controlnet_module, cn_model_name[len('control_'):] if 'control_' in cn_model_name else cn_model_name, controlnet_args.controlnet_weight, controlnet_args.controlnet_invert_image, controlnet_args.controlnet_guidance_start, controlnet_args.controlnet_guidance_end, controlnet_args.controlnet_guess_mode, controlnet_args.controlnet_resize_mode, controlnet_args.controlnet_rgbbgr_mode, controlnet_args.controlnet_processor_res, controlnet_args.controlnet_threshold_a, controlnet_args.controlnet_threshold_b]
    rows = [str(x) for x in rows]

    table.add_row(*rows)
    
    console.print(table)

    p.scripts = scripts.scripts_img2img if is_img2img else scripts.scripts_txt2img

    cnu = {
        "enabled":True,
        "module":controlnet_args.controlnet_module,
        "model":controlnet_args.controlnet_model,
        "weight":controlnet_args.controlnet_weight,
        "image":{'image': cn_image_np, 'mask': cn_mask_np} if cn_mask_np is not None else cn_image_np,
        "invert_image":controlnet_args.controlnet_invert_image,
        "guess_mode":controlnet_args.controlnet_guess_mode,
        "resize_mode":controlnet_args.controlnet_resize_mode,
        "rgbbgr_mode":controlnet_args.controlnet_rgbbgr_mode,
        "low_vram":controlnet_args.controlnet_lowvram,
        "processor_res":controlnet_args.controlnet_processor_res,
        "threshold_a":controlnet_args.controlnet_threshold_a,
        "threshold_b":controlnet_args.controlnet_threshold_b,
        "guidance_start":controlnet_args.controlnet_guidance_start,
        "guidance_end":controlnet_args.controlnet_guidance_end,
    }

    p.script_args = (
        cnu["enabled"],
        cnu["module"],
        cnu["model"],
        cnu["weight"],
        cnu["invert_image"],
        cnu["image"],
        cnu["guess_mode"],
        cnu["resize_mode"],
        cnu["rgbbgr_mode"],
        cnu["low_vram"],
        cnu["processor_res"],
        cnu["threshold_a"],
        cnu["threshold_b"],
        cnu["guidance_start"],
        cnu["guidance_end"],
    )

    cn_units = [
        cnet.ControlNetUnit(**cnu),
    ]

    cnet.update_cn_script_in_processing(p, cn_units, is_img2img=is_img2img, is_ui=False)

import pathlib
from .video_audio_utilities import vid2frames

def unpack_controlnet_vids(args, anim_args, video_args, parseq_args, loop_args, controlnet_args, animation_prompts, root):
    if controlnet_args.controlnet_input_video_chosen_file is not None and len(controlnet_args.controlnet_input_video_chosen_file.name) > 0 or len(controlnet_args.controlnet_vid_path) > 0:
        print(f'Unpacking ControlNet base video')
        # create a folder for the video input frames to live in
        mask_in_frame_path = os.path.join(args.outdir, 'controlnet_inputframes') 
        os.makedirs(mask_in_frame_path, exist_ok=True)

        # save the video frames from mask video
        print(f"Exporting Video Frames (1 every {anim_args.extract_nth_frame}) frames to {mask_in_frame_path}...")
        vid2frames(video_path=controlnet_args.controlnet_vid_path if len(controlnet_args.controlnet_vid_path) > 0 else controlnet_args.controlnet_input_video_chosen_file.name, video_in_frame_path=mask_in_frame_path, n=anim_args.extract_nth_frame, overwrite=controlnet_args.controlnet_overwrite_frames, extract_from_frame=anim_args.extract_from_frame, extract_to_frame=anim_args.extract_to_frame, numeric_files_output=True)

        print(f"Loading {anim_args.max_frames} input frames from {mask_in_frame_path} and saving video frames to {args.outdir}")
        print(f'ControlNet base video unpacked!')
    
    if controlnet_args.controlnet_input_video_mask_chosen_file is not None and len(controlnet_args.controlnet_input_video_mask_chosen_file.name) > 0 or len(controlnet_args.controlnet_mask_vid_path) > 0:
        print(f'Unpacking ControlNet video mask')
        # create a folder for the video input frames to live in
        mask_in_frame_path = os.path.join(args.outdir, 'controlnet_maskframes') 
        os.makedirs(mask_in_frame_path, exist_ok=True)

        # save the video frames from mask video
        print(f"Exporting Video Frames (1 every {anim_args.extract_nth_frame}) frames to {mask_in_frame_path}...")
        vid2frames(video_path=controlnet_args.controlnet_mask_vid_path if len(controlnet_args.controlnet_mask_vid_path) > 0 else controlnet_args.controlnet_input_video_mask_chosen_file.name, video_in_frame_path=mask_in_frame_path, n=anim_args.extract_nth_frame, overwrite=controlnet_args.controlnet_overwrite_frames, extract_from_frame=anim_args.extract_from_frame, extract_to_frame=anim_args.extract_to_frame, numeric_files_output=True)

        print(f"Loading {anim_args.max_frames} input frames from {mask_in_frame_path} and saving video frames to {args.outdir}")
        print(f'ControlNet video mask unpacked!')

def hide_ui_by_cn_status(choice):
    return gr.update(visible=True) if choice else gr.update(visible=False)
    
def build_sliders(module):
    if module == "canny":
        return [
            gr.update(label="Annotator resolution", value=512, minimum=64, maximum=2048, step=1, interactive=True),
            gr.update(label="Canny low threshold", minimum=1, maximum=255, value=100, step=1, interactive=True),
            gr.update(label="Canny high threshold", minimum=1, maximum=255, value=200, step=1, interactive=True),
            gr.update(visible=True)
        ]
    elif module == "mlsd": #Hough
        return [
            gr.update(label="Hough Resolution", minimum=64, maximum=2048, value=512, step=1, interactive=True),
            gr.update(label="Hough value threshold (MLSD)", minimum=0.01, maximum=2.0, value=0.1, step=0.01, interactive=True),
            gr.update(label="Hough distance threshold (MLSD)", minimum=0.01, maximum=20.0, value=0.1, step=0.01, interactive=True),
            gr.update(visible=True)
        ]
    elif module in ["hed", "fake_scribble"]:
        return [
            gr.update(label="HED Resolution", minimum=64, maximum=2048, value=512, step=1, interactive=True),
            gr.update(label="Threshold A", value=64, minimum=64, maximum=1024, interactive=False),
            gr.update(label="Threshold B", value=64, minimum=64, maximum=1024, interactive=False),
            gr.update(visible=True)
        ]
    elif module in ["openpose", "openpose_hand", "segmentation"]:
        return [
            gr.update(label="Annotator Resolution", minimum=64, maximum=2048, value=512, step=1, interactive=True),
            gr.update(label="Threshold A", value=64, minimum=64, maximum=1024, interactive=False),
            gr.update(label="Threshold B", value=64, minimum=64, maximum=1024, interactive=False),
            gr.update(visible=True)
        ]
    elif module == "depth":
        return [
            gr.update(label="Midas Resolution", minimum=64, maximum=2048, value=384, step=1, interactive=True),
            gr.update(label="Threshold A", value=64, minimum=64, maximum=1024, interactive=False),
            gr.update(label="Threshold B", value=64, minimum=64, maximum=1024, interactive=False),
            gr.update(visible=True)
        ]
    elif module in ["depth_leres", "depth_leres_boost"]:
        return [
            gr.update(label="LeReS Resolution", minimum=64, maximum=2048, value=512, step=1, interactive=True),
            gr.update(label="Remove Near %", value=0, minimum=0, maximum=100, step=0.1, interactive=True),
            gr.update(label="Remove Background %", value=0, minimum=0, maximum=100, step=0.1, interactive=True),
            gr.update(visible=True)
        ]
    elif module == "normal_map":
        return [
            gr.update(label="Normal Resolution", minimum=64, maximum=2048, value=512, step=1, interactive=True),
            gr.update(label="Normal background threshold", minimum=0.0, maximum=1.0, value=0.4, step=0.01, interactive=True),
            gr.update(label="Threshold B", value=64, minimum=64, maximum=1024, interactive=False),
            gr.update(visible=True)
        ]
    elif module == "binary":
        return [
            gr.update(label="Annotator resolution", value=512, minimum=64, maximum=2048, step=1, interactive=True),
            gr.update(label="Binary threshold", minimum=0, maximum=255, value=0, step=1, interactive=True),
            gr.update(label="Threshold B", value=64, minimum=64, maximum=1024, interactive=False),
            gr.update(visible=True)
        ]
    elif module == "color":
        return [
            gr.update(label="Annotator Resolution", value=512, minimum=64, maximum=2048, step=8, interactive=True),
            gr.update(label="Threshold A", value=64, minimum=64, maximum=1024, interactive=False),
            gr.update(label="Threshold B", value=64, minimum=64, maximum=1024, interactive=False),
            gr.update(visible=True)
        ]
    elif module == "none":
        return [
            gr.update(label="Normal Resolution", value=64, minimum=64, maximum=2048, interactive=False),
            gr.update(label="Threshold A", value=64, minimum=64, maximum=1024, interactive=False),
            gr.update(label="Threshold B", value=64, minimum=64, maximum=1024, interactive=False),
            gr.update(visible=False)
        ]
    else:
        return [
            gr.update(label="Annotator resolution", value=512, minimum=64, maximum=2048, step=1, interactive=True),
            gr.update(label="Threshold A", value=64, minimum=64, maximum=1024, interactive=False),
            gr.update(label="Threshold B", value=64, minimum=64, maximum=1024, interactive=False),
            gr.update(visible=True)
        ]

