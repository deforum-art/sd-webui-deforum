# Copyright (C) 2023 Deforum LLC
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, version 3 of the License.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

# Contact the authors: https://deforum.github.io/

# This helper script is responsible for AnimateDiff/Deforum integration
# https://github.com/continue-revolution/sd-webui-animatediff — animatediff repo

import os
import copy
import gradio as gr
import scripts
from PIL import Image
import numpy as np
import importlib
from modules import scripts, shared
from .deforum_controlnet_gradio import hide_ui_by_cn_status, hide_file_textboxes, ToolButton
from .general_utils import count_files_in_folder, clean_gradio_path_strings  # TODO: do it another way
from .video_audio_utilities import vid2frames, convert_image
from .animation_key_frames import AnimateDiffKeys
from .load_images import load_image
from .general_utils import debug_print

#self.last_frame = last_frame
#self.latent_power_last = latent_power_last
#self.latent_scale_last = latent_scale_last

cnet = None

def find_animatediff():
    global cnet
    if cnet: return cnet
    try:
        cnet = importlib.import_module('extensions.sd-webui-animatediff.scripts', 'animatediff')
    except:
        try:
            cnet = importlib.import_module('extensions-builtin.sd-webui-animatediff.scripts', 'animatediff')
        except:
            pass
    if cnet:
        print(f"\033[0;32m*Deforum AnimateDiff support: enabled*\033[0m")
        return True
    return None

def is_animatediff_enabled(animatediff_args):
    if getattr(animatediff_args, f'animatediff_enabled', False):
        return True
    return False

def animatediff_infotext():
    return """Requires the <a style='color:SteelBlue;' target='_blank' href='https://github.com/continue-revolution/sd-webui-animatediff'>AnimateDiff</a> extension to be installed.</p>
            <p">If Deforum crashes due to AnimateDiff updates, go <a style='color:Orange;' target='_blank' href='https://github.com/continue-revolution/sd-webui-animatediff/issues'>here</a> and report your problem.</p>
           """

def animatediff_component_names():
    if not find_animatediff():
        return []

    return [f'animatediff_{i}' for i in [
        'enabled', 'model', 'video_length', 'fps', 'loop_number',
        'closed_loop', 'stride', 'overlap', 'interp', 'interp_x', 'reverse', 
        'latent_power', 'latent_scale', 'threshold_b', 'resize_mode', 'control_mode', 'loopback_mode'
    ]]

def setup_animatediff_ui_raw():

    cnet = find_animatediff()

    model_dir = shared.opts.data.get("animatediff_model_path", os.path.join(scripts.basedir(), "model"))

    if not os.path.isdir(model_dir):
        os.mkdir(model_dir)

    cn_models = [f for f in os.listdir(model_dir) if f != ".gitkeep"]

    def refresh_all_models(*inputs):
        new_model_list = [
            f for f in os.listdir(model_dir) if f != ".gitkeep"
        ]
        dd = inputs[0]
        if dd in new_model_list:
            selected = dd
        elif len(new_model_list) > 0:
            selected = new_model_list[0]
        else:
            selected = None
        return gr.Dropdown.update(choices=new_model_list, value=selected)

    refresh_symbol = '\U0001f504'  # 🔄
    switch_values_symbol = '\U000021C5'  # ⇅
    model_dropdowns = []
    infotext_fields = []

    # TODO: unwrap
    def create_model_in_tab_ui(cn_id):
        with gr.Row():
            enabled = gr.Checkbox(label="Enable", value=False, interactive=True)
            pixel_perfect = gr.Checkbox(label="Pixel Perfect", value=False, visible=False, interactive=True)
            low_vram = gr.Checkbox(label="Low VRAM", value=False, visible=False, interactive=True)
            overwrite_frames = gr.Checkbox(label='Overwrite input frames', value=True, visible=False, interactive=True)
        with gr.Row(visible=False) as mod_row:
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
        hide_output_list = [pixel_perfect, low_vram, mod_row, weight_row, start_cs_row, end_cs_row, env_row, overwrite_frames, vid_path_row, control_mode_row, mask_vid_path_row,
                            control_loopback_row]  # add mask_vid_path_row when masks are working again
        for cn_output in hide_output_list:
            enabled.change(fn=hide_ui_by_cn_status, inputs=enabled, outputs=cn_output)

        # hide vid/image input fields
        loopback_outs = [vid_path_row, mask_vid_path_row]
        for loopback_output in loopback_outs:
            loopback_mode.change(fn=hide_file_textboxes, inputs=loopback_mode, outputs=loopback_output)
        infotext_fields.extend([
            (model, f"AnimateDiff Model"),
        ])

        return {key: value for key, value in locals().items() if key in 
            animatediff_component_names()
        }

    with gr.TabItem('AnimateDiff'):
        gr.HTML(animatediff_infotext())
        model_params = create_model_in_tab_ui(0)

        for key, value in model_params.items():
            locals()[f"animatediff_{key}"] = value

    return locals()