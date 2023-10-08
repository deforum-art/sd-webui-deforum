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
#from .animation_key_frames import AnimateDiffKeys
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
    return """Requires the <a style='color:SteelBlue;' target='_blank' href='https://github.com/continue-revolution/sd-webui-animatediff'>AnimateDiff</a> extension to be installed.</p>
            <p">If Deforum crashes due to AnimateDiff updates, go <a style='color:Orange;' target='_blank' href='https://github.com/continue-revolution/sd-webui-animatediff/issues'>here</a> and report your problem.</p>
           """

def animatediff_component_names_raw():
    return [
        'enabled', 'model', 'motion_lora_schedule',
        'window_length', # sliding window length (context batch size)
        'window_overlap', # how much do the contexts overlap. if -1, then batch_size // 4
        'latent_power', 'latent_scale',
    ]

def animatediff_component_names():
    if not find_animatediff():
        return []

    return [f'animatediff_{i}' for i in animatediff_component_names_raw()]

def setup_animatediff_ui_raw():

    cnet = find_animatediff()

    model_dir = ph.models_path + '/AnimateDiff' # shared.opts.data.get("animatediff_model_path", os.path.join(scripts.basedir(), "model"))

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
            enabled = gr.Checkbox(label="Enable AnimateDiff", value=False, interactive=True)
        with gr.Row(visible=False) as mod_row:
            model = gr.Dropdown(cn_models, label=f"Motion module", value="None", interactive=True, tooltip="Choose which motion module will be injected into the generation process.")
            refresh_models = ToolButton(value=refresh_symbol)
            refresh_models.click(refresh_all_models, model, model)
        with gr.Row(visible=False) as motion_lora_row:
            motion_lora_schedule = gr.Textbox(label="Motion lora schedule", lines=1, value='0:("")', interactive=True)
        with gr.Row(visible=False) as window_row:
            window_length = gr.Textbox(label="Number of sliding window frames", lines=1, value='0:(16)', interactive=True)
        with gr.Row(visible=False) as overlap_row:
            # TODO: expose cadence as a variable
            window_overlap = gr.Textbox(label="Number of overlapping frames", lines=1, value='0:(15)', interactive=True)
        with gr.Row(visible=False) as latent_power_row:
            latent_power = gr.Textbox(label="Latent power schedule", lines=1, value='0:(1)', interactive=True)
        with gr.Row(visible=False) as latent_scale_row:
            latent_scale = gr.Textbox(label="Latent scale schedule", lines=1, value='0:(32)', interactive=True)
        hide_output_list = [enabled, motion_lora_row, mod_row, window_row, overlap_row, latent_power_row, latent_scale_row]
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

lora_hacker = None
cfg_hacker = None
cn_hacker = None

# HACK: writing temp frames to a dir for animatediff processing
# TODO: 
def write_temp_frames():
    #tmp_frame_dir = Path(f'{data_path}/tmp/animatediff-frames/')
    #tmp_frame_dir.mkdir(parents=True, exist_ok=True)
    ...

def before_process(p, animatediff_args, temp_video_path):
    global lora_hacker, cfg_hacker, cn_hacker
    
    from scripts.animatediff_lora import AnimateDiffLora
    from scripts.animatediff_infv2v import AnimateDiffInfV2V
    from scripts.animatediff_cn import AnimateDiffControl

    from scripts.animatediff_mm import mm_animatediff as motion_module

    animatediff_args = animatediff_process_from_args(animatediff_args, temp_video_path)

    animatediff_args.set_p(p)
    motion_module.inject(p.sd_model, animatediff_args.animatediff_model)
    lora_hacker = AnimateDiffLora(motion_module.mm.using_v2)
    lora_hacker.hack()
    cfg_hacker = AnimateDiffInfV2V(p)
    cfg_hacker.hack(animatediff_args)
    cn_hacker = AnimateDiffControl(p)
    cn_hacker.hack(animatediff_args)

def postprocess(
        p, animatediff_args
    ):
    global lora_hacker, cfg_hacker, cn_hacker
    cn_hacker.restore()
    cfg_hacker.restore()
    lora_hacker.restore()
    from scripts.animatediff_mm import mm_animatediff as motion_module
    motion_module.restore(p.sd_model)

def animatediff_process_from_args(animatediff_args, temp_video_path):
    args = animatediff_args
    from scripts.animatediff_ui import AnimateDiffProcess

    return AnimateDiffProcess(
        model=args.animatediff_model,
        enable=args.animatediff_enabled,
        video_length=args.animatediff_window_length,
        fps=8, # Irrelevant?
        loop_number=0,
        closed_loop=False,
        batch_size=args.animatediff_window_length,
        stride=1, # From Deforum settings
        overlap=-1,
        format=['MP4'],
        video_source=None,
        video_path=temp_video_path,
        latent_power=args.animatediff_latent_power,
        latent_scale=args.animatediff_latent_scale,
    )

#def set_p(p, animatediff_args):
#    if animatediff_args.animatediff_window_length < 1:
#        return
#    p.batch_size = animatediff_args.animatediff_window_length
#    if animatediff_args.animatediff_window_overlap == -1:
#        animatediff_args.animatediff_window_overlap = animatediff_args.animatediff_window_length // 4
