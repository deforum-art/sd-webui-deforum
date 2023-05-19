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