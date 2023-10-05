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
