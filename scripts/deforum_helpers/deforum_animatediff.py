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
import shutil
from modules import scripts, shared
from .deforum_controlnet_gradio import hide_ui_by_cn_status, hide_file_textboxes, ToolButton
from .general_utils import count_files_in_folder, clean_gradio_path_strings  # TODO: do it another way
from .video_audio_utilities import vid2frames, convert_image
from .animation_key_frames import AnimateDiffKeys
from .load_images import load_image
from .general_utils import debug_print
from modules.shared import opts, cmd_opts, state, sd_model

import modules.paths as ph

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
    return """**Experimental!**
Requires the <a style='color:SteelBlue;' target='_blank' href='https://github.com/continue-revolution/sd-webui-animatediff'>AnimateDiff</a> extension to be installed.</p>
"""

def animatediff_component_names_raw():
    return [
        'enabled', 'model', 'activation_schedule',
        'motion_lora_schedule',
        'video_length_schedule',
        'batch_size_schedule',
        'stride_schedule',
        'overlap_schedule',
        'latent_power_schedule', 'latent_scale_schedule',
        'closed_loop_schedule'
    ]

def animatediff_component_names():
    if not find_animatediff():
        return []

    return [f'animatediff_{i}' for i in animatediff_component_names_raw()]

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
    infotext_fields = []

    # TODO: unwrap
    def create_model_in_tab_ui(cn_id):
        with gr.Row():
            gr.Markdown('Note: AnimateDiff will work only if you have ControlNet installed as well')
            enabled = gr.Checkbox(label="Enable AnimateDiff", value=False, interactive=True)
        with gr.Row(visible=False) as mod_row:
            model = gr.Dropdown(cn_models, label=f"Motion module", value="None", interactive=True, tooltip="Choose which motion module will be injected into the generation process.")
            refresh_models = ToolButton(value=refresh_symbol)
            refresh_models.click(refresh_all_models, model, model)
        with gr.Row(visible=False) as inforow:
            gr.Markdown('**Important!** This schedule sets up when AnimateDiff should run on the generated N previous frames. At the moment this is made with binary values: when the expression value is 0, it will make a pass, otherwise normal Deforum frames will be made')
        with gr.Row(visible=False) as activation_row:
            activation_schedule = gr.Textbox(label="AnimateDiff activation schedule", lines=1, value='0:(1), 2:((t-1) % 16)', interactive=True)
        gr.Markdown('Internal AnimateDiff settings, see its script in normal tabs')
        with gr.Row(visible=False) as motion_lora_row:
            motion_lora_schedule = gr.Textbox(label="Motion lora schedule", lines=1, value='0:("")', interactive=True)
        with gr.Row(visible=False) as length_row:
            video_length_schedule = gr.Textbox(label="N-back video length schedule", lines=1, value='0:(16)', interactive=True)
        with gr.Row(visible=False) as window_row:
            batch_size_schedule = gr.Textbox(label="Batch size", lines=1, value='0:(16)', interactive=True)
        with gr.Row(visible=False) as stride_row:
            stride_schedule = gr.Textbox(label="Stride", lines=1, value='0:(1)', interactive=True)
        with gr.Row(visible=False) as overlap_row:
            overlap_schedule = gr.Textbox(label="Overlap", lines=1, value='0:(-1)', interactive=True)
        with gr.Row(visible=False) as latent_power_row:
            latent_power_schedule = gr.Textbox(label="Latent power schedule", lines=1, value='0:(1)', interactive=True)
        with gr.Row(visible=False) as latent_scale_row:
            latent_scale_schedule = gr.Textbox(label="Latent scale schedule", lines=1, value='0:(32)', interactive=True)
        with gr.Row(visible=False) as rp_row:
            closed_loop_schedule = gr.Textbox(label="Closed loop", lines=1, value='0:("R-P")', interactive=True)
        hide_output_list = [enabled, inforow, activation_row, motion_lora_row, mod_row, length_row, window_row, stride_row, overlap_row, latent_power_row, latent_scale_row, rp_row]
        for cn_output in hide_output_list:
            enabled.change(fn=hide_ui_by_cn_status, inputs=enabled, outputs=cn_output)

        infotext_fields.extend([
            (model, f"AnimateDiff Model"),
        ])

        return {key: value for key, value in locals().items() if key in 
            animatediff_component_names_raw()
        }

    with gr.TabItem('AnimateDiff'):
        gr.HTML(animatediff_infotext())
        model_params = create_model_in_tab_ui(0)

        for key, value in model_params.items():
            locals()[f"animatediff_{key}"] = value

    return locals()

def setup_animatediff_ui():
    if not find_animatediff():
        gr.HTML("""<a style='target='_blank' href='https://github.com/continue-revolution/sd-webui-animatediff'>AnimateDiff not found. Please install it :)</a>""", elem_id='animatediff_not_found_html_msg')
        return {}

    try:
        return setup_animatediff_ui_raw()
    except Exception as e:
        print(f"'AnimateDiff UI setup failed with error: '{e}'!")
        gr.HTML(f"""
                Failed to setup AnimateDiff UI, check the reason in your commandline log. Please, downgrade your AnimateDiff extension to <a style='color:Orange;' target='_blank' href='https://github.com/continue-revolution/sd-webui-animatediff/archive/b192a2551a5ed66d4a3ce58d5d19a8872abc87ca.zip'>b192a2551a5ed66d4a3ce58d5d19a8872abc87ca</a> and report the problem <a style='color:Orange;' target='_blank' href='https://github.com/deforum-art/sd-webui-deforum'>here</a> (Deforum) or <a style='color:Orange;' target='_blank' href='https://github.com/continue-revolution/sd-webui-animatediff'>here</a> (AnimateDiff).
                """, elem_id='animatediff_not_found_html_msg')
        return {}

def find_animatediff_script(p):
    animatediff_script = next((script for script in p.scripts.alwayson_scripts if "animatediff" in script.title().lower()), None)
    if not animatediff_script:
        raise Exception("AnimateDiff script not found.")
    return animatediff_script

def get_animatediff_temp_dir(args):
    return os.path.join(args.outdir, 'animatediff_temp')

def need_animatediff(animatediff_args):
    return find_animatediff() is not None and is_animatediff_enabled(animatediff_args)

def seed_animatediff(p, animatediff_args, args, anim_args, root, frame_idx):
    if not need_animatediff(animatediff_args):
        return

    keys = AnimateDiffKeys(animatediff_args, anim_args) # if not parseq_adapter.use_parseq else parseq_adapter.cn_keys
    
    # Will do the back-render only on target frames
    if int(keys.activation_schedule_series[frame_idx]) != 0:
        return
    
    video_length = int(keys.video_length_schedule_series[frame_idx])
    assert video_length > 1

    # Managing the frames to be fed into AD:
    # Create a temporal directory
    animatediff_temp_dir = get_animatediff_temp_dir(args)
    if os.path.exists(animatediff_temp_dir):
        shutil.rmtree(animatediff_temp_dir)
    os.makedirs(animatediff_temp_dir)
    # Copy the frames (except for the one which is being CN-made) into that dir
    for offset in range(video_length - 1):
        filename = f"{root.timestring}_{frame_idx - offset - 1:09}.png"
        Image.open(os.path.join(args.outdir, filename)).save(os.path.join(animatediff_temp_dir, f"{offset:09}.png"), "PNG")

    animatediff_script = find_animatediff_script(p)
    # let's put it before ControlNet to cause less problems
    p.scripts.alwayson_scripts = [animatediff_script] + p.scripts.alwayson_scripts

    args_dict = {
        'model': keys.model,   # Motion module
        'format': ['Frame'],      # Save format, 'GIF' | 'MP4' | 'PNG' | 'WEBP' | 'WEBM' | 'TXT' | 'Frame'
        'enable': keys.enable,         # Enable AnimateDiff
        'video_length': video_length,     # Number of frames
        'fps': 8,               # FPS - don't care
        'loop_number': 0,       # Display loop number
        'closed_loop': keys.closed_loop_schedule_series[frame_idx],   # Closed loop, 'N' | 'R-P' | 'R+P' | 'A'
        'batch_size': int(keys.batch_size_schedule_series[frame_idx]),       # Context batch size
        'stride': int(keys.stride_schedule_series[frame_idx]),            # Stride 
        'overlap': int(keys.overlap_schedule_series[frame_idx]),          # Overlap
        'interp': 'Off',        # Frame interpolation, 'Off' | 'FILM' - don't care
        'interp_x': 10,          # Interp X - don't care
        'video_source': '',  # We don't use a video
        'video_path': animatediff_temp_dir, # Path with our selected video_length input frames
        'latent_power': keys.latent_power_schedule_series[frame_idx],      # Latent power
        'latent_scale': keys.latent_scale_schedule_series[frame_idx],     # Latent scale
        'last_frame': None,     # Optional last frame
        'latent_power_last': 1, # Optional latent power for last frame
        'latent_scale_last': 32,# Optional latent scale for last frame
        'request_id': ''        # Optional request id. If provided, outputs will have request id as filename suffix
    }

    args = list(args_dict.values())

    p.script_args_value = args + p.script_args_value

def reap_animatediff(images, animatediff_args, args, root, frame_idx):
    if not need_animatediff(animatediff_args):
        return
    
    animatediff_temp_dir = get_animatediff_temp_dir(args)
    assert os.path.exists(animatediff_temp_dir)

    for offset in range(len(images)):
        frame = images[-offset-1]
        cur_frame_idx = frame_idx - offset

        # overwrite the results
        filename = f"{root.timestring}_{cur_frame_idx:09}.png"
        frame.save(os.path.join(args.outdir, filename), "PNG")
