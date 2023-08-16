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

import gradio as gr
from modules import ui_components
from modules.shared import opts, cmd_opts, OptionInfo
from .video_audio_utilities import find_ffmpeg_binary
from .subtitle_handler import get_user_values

def on_ui_settings():
    srt_ui_params = get_user_values()
    section = ('deforum', "Deforum")
    opts.add_option("deforum_keep_3d_models_in_vram", OptionInfo(False, "Keep 3D models in VRAM between runs", gr.Checkbox, {"interactive": True, "visible": True if not (cmd_opts.lowvram or cmd_opts.medvram) else False}, section=section))
    opts.add_option("deforum_enable_persistent_settings", OptionInfo(False, "Keep settings persistent upon relaunch of webui", gr.Checkbox, {"interactive": True}, section=section))
    opts.add_option("deforum_persistent_settings_path", OptionInfo("models/Deforum/deforum_persistent_settings.txt", "Path for saving your persistent settings file:", section=section))
    opts.add_option("deforum_ffmpeg_location", OptionInfo(find_ffmpeg_binary(), "FFmpeg path/ location", section=section))
    opts.add_option("deforum_ffmpeg_crf", OptionInfo(17, "FFmpeg CRF value", gr.Slider, {"interactive": True, "minimum": 0, "maximum": 51}, section=section))
    opts.add_option("deforum_ffmpeg_preset", OptionInfo('slow', "FFmpeg Preset", gr.Dropdown, {"interactive": True, "choices": ['veryslow', 'slower', 'slow', 'medium', 'fast', 'faster', 'veryfast', 'superfast', 'ultrafast']}, section=section))
    opts.add_option("deforum_debug_mode_enabled", OptionInfo(False, "Enable Dev mode - adds extra reporting in console", gr.Checkbox, {"interactive": True}, section=section))
    opts.add_option("deforum_save_gen_info_as_srt", OptionInfo(False, "Save an .srt (subtitles) file with the generation info along with each animation", gr.Checkbox, {"interactive": True}, section=section))  
    opts.add_option("deforum_embed_srt", OptionInfo(False, "If .srt file is saved, soft-embed the subtitles into the rendered video file", gr.Checkbox, {"interactive": True}, section=section))  
    opts.add_option("deforum_save_gen_info_as_srt_params", OptionInfo(['Noise Schedule'], "Choose which animation params are to be saved to the .srt file (Frame # and Seed will always be saved):", ui_components.DropdownMulti, lambda: {"interactive": True, "choices": srt_ui_params}, section=section)) 
    opts.add_option("deforum_preview", OptionInfo("Off", "Generate preview video during generation? (Preview does not include frame interpolation or upscaling.)", gr.Dropdown, {"interactive": True, "choices": ['Off', 'On', 'On, concurrent (don\'t pause generation)']}, section=section))
    opts.add_option("deforum_preview_interval_frames", OptionInfo(100, "Generate preview every N frames", gr.Slider, {"interactive": True, "minimum": 10, "maximum": 500}, section=section))
