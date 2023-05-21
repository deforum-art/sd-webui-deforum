# This helper script is responsible for ControlNet/Deforum integration
# https://github.com/Mikubill/sd-webui-controlnet — controlnet repo

import os
import gradio as gr
import scripts
from PIL import Image
import numpy as np
import importlib
from modules import scripts
from .deforum_controlnet_gradio import hide_ui_by_cn_status, hide_file_textboxes, ToolButton
from .general_utils import count_files_in_folder, clean_gradio_path_strings  # TODO: do it another way
from .video_audio_utilities import vid2frames, convert_image
from .animation_key_frames import ControlNetKeys
from .load_images import load_image

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
    for i in range(1, num_of_models + 1):
        if getattr(controlnet_args, f'cn_{i}_enabled', False):
            return True
    return False

def setup_controlnet_ui_raw():
    cnet = find_controlnet()
    cn_models = cnet.get_models()
    cn_preprocessors = cnet.get_modules()

    cn_modules = cnet.get_modules_detail()
    preprocessor_sliders_config = {}

    for config_name, config_values in cn_modules.items():
        sliders = config_values.get('sliders', [])
        preprocessor_sliders_config[config_name] = sliders

    model_free_preprocessors = ["reference_only", "reference_adain", "reference_adain+attn"]
    flag_preprocessor_resolution = "Preprocessor Resolution"

    def build_sliders(module, pp):
        grs = []
        if module not in preprocessor_sliders_config:
            grs += [
                gr.update(label=flag_preprocessor_resolution, value=512, minimum=64, maximum=2048, step=1, visible=not pp, interactive=not pp),
                gr.update(visible=False, interactive=False),
                gr.update(visible=False, interactive=False),
                gr.update(visible=True)
            ]
        else:
            for slider_config in preprocessor_sliders_config[module]:
                if isinstance(slider_config, dict):
                    visible = True
                    if slider_config['name'] == flag_preprocessor_resolution:
                        visible = not pp
                    grs.append(gr.update(
                        label=slider_config['name'],
                        value=slider_config['value'],
                        minimum=slider_config['min'],
                        maximum=slider_config['max'],
                        step=slider_config['step'] if 'step' in slider_config else 1,
                        visible=visible,
                        interactive=visible))
                else:
                    grs.append(gr.update(visible=False, interactive=False))
            while len(grs) < 3:
                grs.append(gr.update(visible=False, interactive=False))
            grs.append(gr.update(visible=True))
        if module in model_free_preprocessors:
            grs += [gr.update(visible=False, value='None'), gr.update(visible=False)]
        else:
            grs += [gr.update(visible=True), gr.update(visible=True)]
        return grs

    refresh_symbol = '\U0001f504'  # 🔄
    switch_values_symbol = '\U000021C5'  # ⇅
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
            weight = gr.Textbox(label="Weight schedule", lines=1, value='0:(1)', interactive=True)
        with gr.Row(visible=False) as start_cs_row:
            guidance_start = gr.Textbox(label="Starting Control Step schedule", lines=1, value='0:(0.0)', interactive=True)
        with gr.Row(visible=False) as end_cs_row:
            guidance_end = gr.Textbox(label="Ending Control Step schedule", lines=1, value='0:(1.0)', interactive=True)
            model_dropdowns.append(model)
        with gr.Column(visible=False) as advanced_column:
            processor_res = gr.Slider(label="Annotator resolution", value=64, minimum=64, maximum=2048, interactive=False)
            threshold_a = gr.Slider(label="Threshold A", value=64, minimum=64, maximum=1024, interactive=False)
            threshold_b = gr.Slider(label="Threshold B", value=64, minimum=64, maximum=1024, interactive=False)
        with gr.Row(visible=False) as vid_path_row:
            vid_path = gr.Textbox(value='', label="ControlNet Input Video/ Image Path", interactive=True)
        with gr.Row(visible=False) as mask_vid_path_row:  # invisible temporarily since 26-04-23 until masks are fixed
            mask_vid_path = gr.Textbox(value='', label="ControlNet Mask Video/ Image Path (*NOT WORKING, kept in UI for CN's devs testing!*)", interactive=True)
        with gr.Row(visible=False) as control_mode_row:
            control_mode = gr.Radio(choices=["Balanced", "My prompt is more important", "ControlNet is more important"], value="Balanced", label="Control Mode", interactive=True)
        with gr.Row(visible=False) as env_row:
            resize_mode = gr.Radio(choices=["Outer Fit (Shrink to Fit)", "Inner Fit (Scale to Fit)", "Just Resize"], value="Inner Fit (Scale to Fit)", label="Resize Mode", interactive=True)
        with gr.Row(visible=False) as control_loopback_row:
            loopback_mode = gr.Checkbox(label="LoopBack mode", value=False, interactive=True)
        hide_output_list = [pixel_perfect, low_vram, mod_row, module, weight_row, start_cs_row, end_cs_row, env_row, overwrite_frames, vid_path_row, control_mode_row, mask_vid_path_row,
                            control_loopback_row]  # add mask_vid_path_row when masks are working again
        for cn_output in hide_output_list:
            enabled.change(fn=hide_ui_by_cn_status, inputs=enabled, outputs=cn_output)
        module.change(build_sliders, inputs=[module, pixel_perfect], outputs=[processor_res, threshold_a, threshold_b, advanced_column, model, refresh_models])
        # hide vid/image input fields
        loopback_outs = [vid_path_row, mask_vid_path_row]
        for loopback_output in loopback_outs:
            loopback_mode.change(fn=hide_file_textboxes, inputs=loopback_mode, outputs=loopback_output)
        # handle pixel perfect ui changes
        pixel_perfect.change(build_sliders, inputs=[module, pixel_perfect], outputs=[processor_res, threshold_a, threshold_b, advanced_column, model, refresh_models])
        infotext_fields.extend([
            (module, f"ControlNet Preprocessor"),
            (model, f"ControlNet Model"),
            (weight, f"ControlNet Weight"),
        ])

        return {key: value for key, value in locals().items() if key in [
            "enabled", "pixel_perfect", "low_vram", "module", "model", "weight",
            "guidance_start", "guidance_end", "processor_res", "threshold_a", "threshold_b", "resize_mode", "control_mode",
            "overwrite_frames", "vid_path", "mask_vid_path", "loopback_mode"
        ]}

    def refresh_all_models(*inputs):
        cn_models = cnet.get_models(update=True)
        dd = inputs[0]
        selected = dd if dd in cn_models else "None"
        return gr.Dropdown.update(value=selected, choices=cn_models)

    with gr.TabItem('ControlNet'):
        gr.HTML(controlnet_infotext())
        with gr.Tabs():
            model_params = {}
            for i in range(1, num_of_models + 1):
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

    return [f'cn_{i}_{component}' for i in range(1, num_of_models + 1) for component in [
        'overwrite_frames', 'vid_path', 'mask_vid_path', 'enabled',
        'low_vram', 'pixel_perfect',
        'module', 'model', 'weight', 'guidance_start', 'guidance_end',
        'processor_res', 'threshold_a', 'threshold_b', 'resize_mode', 'control_mode', 'loopback_mode'
    ]]

def process_with_controlnet(p, args, anim_args, controlnet_args, root, is_img2img=True, frame_idx=0):
    CnSchKeys = ControlNetKeys(anim_args, controlnet_args)

    def read_cn_data(cn_idx):
        cn_mask_np, cn_image_np = None, None
        # Loopback mode ENABLED:
        if getattr(controlnet_args, f'cn_{cn_idx}_loopback_mode'):
            # On very first frame, check if use init enabled, and if init image is provided
            if frame_idx == 0 and args.use_init and args.init_image is not None:
                cn_image_np = load_image(args.init_image)
                # convert to uint8 for compatibility with CN
                cn_image_np = np.array(cn_image_np).astype('uint8')
            # Not first frame, use previous img (init_sample)
            elif frame_idx > 0 and root.init_sample:
                cn_image_np = np.array(root.init_sample).astype('uint8')
        else:  # loopback mode is DISABLED
            cn_inputframes = os.path.join(args.outdir, f'controlnet_{cn_idx}_inputframes')  # set input frames folder path
            if os.path.exists(cn_inputframes):
                if count_files_in_folder(cn_inputframes) == 1:
                    cn_frame_path = os.path.join(cn_inputframes, "000000000.jpg")
                    print(f'Reading ControlNet *static* base frame at {cn_frame_path}')
                else:
                    cn_frame_path = os.path.join(cn_inputframes, f"{frame_idx:09}.jpg")
                    print(f'Reading ControlNet {cn_idx} base frame #{frame_idx} at {cn_frame_path}')
                if os.path.exists(cn_frame_path):
                    cn_image_np = np.array(Image.open(cn_frame_path).convert("RGB")).astype('uint8')
            cn_maskframes = os.path.join(args.outdir, f'controlnet_{cn_idx}_maskframes')  # set mask frames folder path
            if os.path.exists(cn_maskframes):
                if count_files_in_folder(cn_maskframes) == 1:
                    cn_mask_frame_path = os.path.join(cn_inputframes, "000000000.jpg")
                    print(f'Reading ControlNet *static* mask frame at {cn_mask_frame_path}')
                else:
                    cn_mask_frame_path = os.path.join(args.outdir, f'controlnet_{cn_idx}_maskframes', f"{frame_idx:09}.jpg")
                    print(f'Reading ControlNet {cn_idx} mask frame #{frame_idx} at {cn_mask_frame_path}')
                if os.path.exists(cn_mask_frame_path):
                    cn_mask_np = np.array(Image.open(cn_mask_frame_path).convert("RGB")).astype('uint8')

        return cn_mask_np, cn_image_np

    cnet = find_controlnet()
    cn_data = [read_cn_data(i) for i in range(1, num_of_models + 1)]

    # Check if any loopback_mode is set to True
    any_loopback_mode = any(getattr(controlnet_args, f'cn_{i}_loopback_mode') for i in range(1, num_of_models + 1))

    cn_inputframes_list = [os.path.join(args.outdir, f'controlnet_{i}_inputframes') for i in range(1, num_of_models + 1)]

    if not any(os.path.exists(cn_inputframes) for cn_inputframes in cn_inputframes_list) and not any_loopback_mode:
        print(f'\033[33mNeither the base nor the masking frames for ControlNet were found. Using the regular pipeline\033[0m')

    p.scripts = scripts.scripts_img2img if is_img2img else scripts.scripts_txt2img

    def create_cnu_dict(cn_args, prefix, img_np, mask_np, frame_idx, CnSchKeys):

        keys = [
            "enabled", "module", "model", "weight", "resize_mode", "control_mode", "low_vram", "pixel_perfect",
            "processor_res", "threshold_a", "threshold_b", "guidance_start", "guidance_end"
        ]
        cnu = {k: getattr(cn_args, f"{prefix}_{k}") for k in keys}
        model_num = int(prefix.split('_')[-1])  # Extract model number from prefix (e.g., "cn_1" -> 1)
        if 1 <= model_num <= 5:
            # if in loopmode and no init image (img_np, after processing in this case) provided, disable CN unit for the very first frame. Will be enabled in the next frame automatically
            if getattr(cn_args, f"cn_{model_num}_loopback_mode") and frame_idx == 0 and img_np is None:
                cnu['enabled'] = False
            cnu['weight'] = getattr(CnSchKeys, f"cn_{model_num}_weight_schedule_series")[frame_idx]
            cnu['guidance_start'] = getattr(CnSchKeys, f"cn_{model_num}_guidance_start_schedule_series")[frame_idx]
            cnu['guidance_end'] = getattr(CnSchKeys, f"cn_{model_num}_guidance_end_schedule_series")[frame_idx]
        cnu['image'] = {'image': img_np, 'mask': mask_np} if mask_np is not None else img_np

        return cnu

    masks_np, images_np = zip(*cn_data)

    cn_units = [cnet.ControlNetUnit(**create_cnu_dict(controlnet_args, f"cn_{i + 1}", img_np, mask_np, frame_idx, CnSchKeys))
                for i, (img_np, mask_np) in enumerate(zip(images_np, masks_np))]

    p.script_args = {"enabled": True}
    cnet.update_cn_script_in_processing(p, cn_units, is_img2img=is_img2img, is_ui=False)

def process_controlnet_input_frames(args, anim_args, controlnet_args, video_path, mask_path, outdir_suffix, id):
    if (video_path or mask_path) and getattr(controlnet_args, f'cn_{id}_enabled'):
        frame_path = os.path.join(args.outdir, f'controlnet_{id}_{outdir_suffix}')
        os.makedirs(frame_path, exist_ok=True)

        accepted_image_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
        if video_path and video_path.lower().endswith(accepted_image_extensions):
            convert_image(video_path, os.path.join(frame_path, '000000000.jpg'))
            print(f"Copied CN Model {id}'s single input image to inputframes folder!")
        elif mask_path and mask_path.lower().endswith(accepted_image_extensions):
            convert_image(mask_path, os.path.join(frame_path, '000000000.jpg'))
            print(f"Copied CN Model {id}'s single input image to inputframes *mask* folder!")
        else:
            print(f'Unpacking ControlNet {id} {"video mask" if mask_path else "base video"}')
            print(f"Exporting Video Frames to {frame_path}...")
            vid2frames(
                video_path=video_path or mask_path,
                video_in_frame_path=frame_path,
                n=1 if anim_args.animation_mode != 'Video Input' else anim_args.extract_nth_frame,
                overwrite=getattr(controlnet_args, f'cn_{id}_overwrite_frames'),
                extract_from_frame=0 if anim_args.animation_mode != 'Video Input' else anim_args.extract_from_frame,
                extract_to_frame=(anim_args.max_frames - 1) if anim_args.animation_mode != 'Video Input' else anim_args.extract_to_frame,
                numeric_files_output=True
            )
            print(f"Loading {anim_args.max_frames} input frames from {frame_path} and saving video frames to {args.outdir}")
            print(f'ControlNet {id} {"video mask" if mask_path else "base video"} unpacked!')

def unpack_controlnet_vids(args, anim_args, controlnet_args):
    # this func gets called from render.py once for an entire animation run -->
    # tries to trigger an extraction of CN input frames (regular + masks) from video or image
    for i in range(1, num_of_models + 1):
        # LoopBack mode is enabled, no need to extract a video or copy an init image
        if getattr(controlnet_args, f'cn_{i}_loopback_mode'):
            print(f"ControlNet #{i} is in LoopBack mode, skipping video/ image extraction stage.")
            continue
        vid_path = clean_gradio_path_strings(getattr(controlnet_args, f'cn_{i}_vid_path', None))
        mask_path = clean_gradio_path_strings(getattr(controlnet_args, f'cn_{i}_mask_vid_path', None))

        if vid_path:  # Process base video, if available
            process_controlnet_input_frames(args, anim_args, controlnet_args, vid_path, None, 'inputframes', i)

        if mask_path:  # Process mask video, if available
            process_controlnet_input_frames(args, anim_args, controlnet_args, None, mask_path, 'maskframes', i)
