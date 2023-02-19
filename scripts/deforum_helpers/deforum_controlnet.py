# This helper script is responsible for ControlNet/Deforum integration
# https://github.com/Mikubill/sd-webui-controlnet — controlnet repo

import os, sys
import gradio as gr

has_controlnet = None

def find_controlnet():
    global has_controlnet
    if has_controlnet is not None:
        return has_controlnet
    
    try:
        from scripts import controlnet
    except Exception as e:
        print(f'\033[33mFailed to import controlnet! The exact error is {e}. Deforum support for ControlNet will not be activated\033[0m')
        has_controlnet = False
        return False
    has_controlnet = True
    print(f'\033[0;32mCongratulations! You have ControlNet support for Deforum enabled!\033[0m')
    return True

# The most parts below are plainly copied from controlnet.py
# TODO: come up with a cleaner way

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
    controlnet_scribble_mode = False
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
    from scripts import controlnet
    from scripts.controlnet import update_cn_models, cn_models, cn_models_names

    model_dropdowns = []
    infotext_fields = []

    # Already under an accordion

    # Copying the main ControlNet widgets while getting rid of static elements such as the scribble pad
    with gr.Row():
        controlnet_enabled = gr.Checkbox(label='Enable', value=False)
        controlnet_scribble_mode = gr.Checkbox(label='Scribble Mode (Invert colors)', value=False)
        controlnet_rgbbgr_mode = gr.Checkbox(label='RGB to BGR', value=False)
        controlnet_lowvram = gr.Checkbox(label='Low VRAM', value=False)

    # Main part
    class ToolButton(gr.Button, gr.components.FormComponent):
        """Small button with single emoji as text, fits inside gradio forms"""

        def __init__(self, **kwargs):
            super().__init__(variant="tool", **kwargs)

        def get_block_name(self):
            return "button"

    from scripts.processor import canny, midas, midas_normal, leres, hed, mlsd, openpose, pidinet, simple_scribble, fake_scribble, uniformer

    preprocessor = {
        "none": lambda x, *args, **kwargs: x,
        "canny": canny,
        "depth": midas,
        "depth_leres": leres,
        "hed": hed,
        "mlsd": mlsd,
        "normal_map": midas_normal,
        "openpose": openpose,
        # "openpose_hand": openpose_hand,
        "pidinet": pidinet,
        "scribble": simple_scribble,
        "fake_scribble": fake_scribble,
        "segmentation": uniformer,
    }

    def refresh_all_models(*inputs):
        update_cn_models()
        
        dd = inputs[0]
        selected = dd if dd in cn_models else "None"
        return gr.Dropdown.update(value=selected, choices=list(cn_models.keys()))

    refresh_symbol = '\U0001f504'  # 🔄
    switch_values_symbol = '\U000021C5' # ⇅

    with gr.Row():
        controlnet_module = gr.Dropdown(list(preprocessor.keys()), label=f"Preprocessor", value="none")
        controlnet_model = gr.Dropdown(list(cn_models.keys()), label=f"Model", value="None")
        refresh_models = ToolButton(value=refresh_symbol)
        refresh_models.click(refresh_all_models, controlnet_model, controlnet_model)
        # ctrls += (refresh_models, )
    with gr.Row():
        controlnet_weight = gr.Slider(label=f"Weight", value=1.0, minimum=0.0, maximum=2.0, step=.05)
        controlnet_guidance_strength =  gr.Slider(label="Guidance strength (T)", value=1.0, minimum=0.0, maximum=1.0, interactive=True)

        # ctrls += (module, model, weight,)
        # model_dropdowns.append(model)
        
    def build_sliders(module):
        if controlnet_module == "canny":
            return [
                gr.update(label="Annotator resolution", value=512, minimum=64, maximum=2048, step=1, interactive=True),
                gr.update(label="Canny low threshold", minimum=1, maximum=255, value=100, step=1, interactive=True),
                gr.update(label="Canny high threshold", minimum=1, maximum=255, value=200, step=1, interactive=True),
                gr.update(visible=True)
            ]
        elif controlnet_module == "mlsd": #Hough
            return [
                gr.update(label="Hough Resolution", minimum=64, maximum=2048, value=512, step=1, interactive=True),
                gr.update(label="Hough value threshold (MLSD)", minimum=0.01, maximum=2.0, value=0.1, step=0.01, interactive=True),
                gr.update(label="Hough distance threshold (MLSD)", minimum=0.01, maximum=20.0, value=0.1, step=0.01, interactive=True),
                gr.update(visible=True)
            ]
        elif controlnet_module in ["hed", "fake_scribble"]:
            return [
                gr.update(label="HED Resolution", minimum=64, maximum=2048, value=512, step=1, interactive=True),
                gr.update(label="Threshold A", value=64, minimum=64, maximum=1024, interactive=False),
                gr.update(label="Threshold B", value=64, minimum=64, maximum=1024, interactive=False),
                gr.update(visible=True)
            ]
        elif controlnet_module in ["openpose", "openpose_hand", "segmentation"]:
            return [
                gr.update(label="Annotator Resolution", minimum=64, maximum=2048, value=512, step=1, interactive=True),
                gr.update(label="Threshold A", value=64, minimum=64, maximum=1024, interactive=False),
                gr.update(label="Threshold B", value=64, minimum=64, maximum=1024, interactive=False),
                gr.update(visible=True)
            ]
        elif controlnet_module == "depth":
            return [
                gr.update(label="Midas Resolution", minimum=64, maximum=2048, value=384, step=1, interactive=True),
                gr.update(label="Threshold A", value=64, minimum=64, maximum=1024, interactive=False),
                gr.update(label="Threshold B", value=64, minimum=64, maximum=1024, interactive=False),
                gr.update(visible=True)
            ]
        elif controlnet_module == "depth_leres":
            return [
                gr.update(label="LeReS Resolution", minimum=64, maximum=2048, value=512, step=1, interactive=True),
                gr.update(label="Remove Near %", value=0, minimum=0, maximum=100, step=0.1, interactive=True),
                gr.update(label="Remove Background %", value=0, minimum=0, maximum=100, step=0.1, interactive=True),
                gr.update(visible=True)
            ]
        elif controlnet_module == "normal_map":
            return [
                gr.update(label="Normal Resolution", minimum=64, maximum=2048, value=512, step=1, interactive=True),
                gr.update(label="Normal background threshold", minimum=0.0, maximum=1.0, value=0.4, step=0.01, interactive=True),
                gr.update(label="Threshold B", value=64, minimum=64, maximum=1024, interactive=False),
                gr.update(visible=True)
            ]
        elif controlnet_module == "none":
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

    # def svgPreprocess(inputs):
    #     if (inputs):
    #         if (inputs['image'].startswith("data:image/svg+xml;base64,") and svgsupport):
    #             svg_data = base64.b64decode(inputs['image'].replace('data:image/svg+xml;base64,',''))
    #             drawing = svg2rlg(io.BytesIO(svg_data))
    #             png_data = renderPM.drawToString(drawing, fmt='PNG')
    #             encoded_string = base64.b64encode(png_data)
    #             base64_str = str(encoded_string, "utf-8")
    #             base64_str = "data:image/png;base64,"+ base64_str
    #             inputs['image'] = base64_str
    #         return input_image.orgpreprocess(inputs)
    #     return None

    with gr.Row():
        controlnet_resize_mode = gr.Radio(choices=["Envelope (Outer Fit)", "Scale to Fit (Inner Fit)", "Just Resize"], value="Scale to Fit (Inner Fit)", label="Resize Mode")

    # Video input to be fed into ControlNet
    #input_video_url = gr.Textbox(source='upload', type='numpy', tool='sketch') # TODO
    controlnet_input_video_chosen_file = gr.File(label="ControlNet Video Input", interactive=True, file_count="single", file_types=["video"], elem_id="controlnet_input_video_chosen_file")
    controlnet_input_video_mask_chosen_file = gr.File(label="ControlNet Video Mask Input", interactive=True, file_count="single", file_types=["video"], elem_id="controlnet_input_video_mask_chosen_file")

    return locals()

def setup_controlnet_ui():
    if not find_controlnet():
        gr.HTML("""
                <a style='color:red;' target='_blank' href='https://github.com/Mikubill/sd-webui-controlnet'>ControlNet not found! Please, install it</a>
                """)
        return {}

    return setup_controlnet_ui_raw()


def controlnet_component_names():
    if not find_controlnet():
        return []

    controlnet_args_names = str(r'''controlnet_input_video_chosen_file, controlnet_input_video_mask_chosen_file,
controlnet_enabled, controlnet_scribble_mode, controlnet_rgbbgr_mode, controlnet_lowvram,
controlnet_module, controlnet_model,
controlnet_weight, controlnet_guidance_strength,
controlnet_processor_res, 
controlnet_threshold_a, controlnet_threshold_b, controlnet_resize_mode'''
    ).replace("\n", "").replace("\r", "").replace(" ", "").split(',')
    
    return controlnet_args_names

def augment_txt2img_pipeline(p):
    pass

def augment_img2img_pipeline(p):
    pass

import pathlib
from video_audio_utilities import vid2frames

def unpack_controlnet_vids(args, anim_args, video_args, parseq_args, loop_args, controlnet_args, animation_prompts, root):
    if controlnet_args.controlnet_input_video_chosen_file is not None and len(controlnet_args.controlnet_input_video_chosen_file.name) > 0:
        print(f'Unpacking ControlNet base video')
        # create a folder for the video input frames to live in
        mask_in_frame_path = os.path.join(args.outdir, 'controlnet_inputframes') 
        os.makedirs(mask_in_frame_path, exist_ok=True)

        # save the video frames from mask video
        print(f"Exporting Video Frames (1 every {anim_args.extract_nth_frame}) frames to {mask_in_frame_path}...")
        vid2frames(video_path=controlnet_args.controlnet_input_video_chosen_file.name, video_in_frame_path=mask_in_frame_path, n=anim_args.extract_nth_frame, overwrite=anim_args.overwrite_extracted_frames, extract_from_frame=anim_args.extract_from_frame, extract_to_frame=anim_args.extract_to_frame)

        print(f"Loading {anim_args.max_frames} input frames from {mask_in_frame_path} and saving video frames to {args.outdir}")
        print(f'ControlNet base video unpacked!')
    
    if controlnet_args.controlnet_input_video_mask_chosen_file is not None and len(controlnet_args.controlnet_input_video_mask_chosen_file.name) > 0:
        print(f'Unpacking ControlNet video mask')
        # create a folder for the video input frames to live in
        mask_in_frame_path = os.path.join(args.outdir, 'controlnet_maskframes') 
        os.makedirs(mask_in_frame_path, exist_ok=True)

        # save the video frames from mask video
        print(f"Exporting Video Frames (1 every {anim_args.extract_nth_frame}) frames to {mask_in_frame_path}...")
        vid2frames(video_path=controlnet_args.controlnet_input_video_mask_chosen_file.name, video_in_frame_path=mask_in_frame_path, n=anim_args.extract_nth_frame, overwrite=anim_args.overwrite_extracted_frames, extract_from_frame=anim_args.extract_from_frame, extract_to_frame=anim_args.extract_to_frame)

        print(f"Loading {anim_args.max_frames} input frames from {mask_in_frame_path} and saving video frames to {args.outdir}")
        print(f'ControlNet video mask unpacked!')
    