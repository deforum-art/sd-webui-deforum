# Detach 'deforum_helpers' from 'scripts' to prevent "No module named 'scripts.deforum_helpers'"  error 
# causing Deforum's tab not show up in some cases when you've might've broken the environment with webui packages updates
import sys, os, shutil

basedirs = [os.getcwd()]
if 'google.colab' in sys.modules:
    basedirs.append('/content/gdrive/MyDrive/sd/stable-diffusion-webui') #hardcode as TheLastBen's colab seems to be the primal source

for basedir in basedirs:
    deforum_paths_to_ensure = [basedir + '/extensions/deforum-for-automatic1111-webui/scripts', basedir + '/extensions/sd-webui-controlnet', basedir + '/extensions/deforum/scripts', basedir + '/scripts/deforum_helpers/src', basedir + '/extensions/deforum/scripts/deforum_helpers/src', basedir +'/extensions/deforum-for-automatic1111-webui/scripts/deforum_helpers/src',basedir]

    for deforum_scripts_path_fix in deforum_paths_to_ensure:
        if not deforum_scripts_path_fix in sys.path:
            sys.path.extend([deforum_scripts_path_fix])

# Main deforum stuff
import deforum_helpers.args as deforum_args
import deforum_helpers.settings as deforum_settings
from deforum_helpers.save_images import dump_frames_cache, reset_frames_cache
from deforum_helpers.frame_interpolation import process_video_interpolation

import modules.scripts as wscripts
from modules import script_callbacks
import gradio as gr
import json

from modules.processing import Processed, StableDiffusionProcessingImg2Img, process_images
from PIL import Image
from deforum_helpers.video_audio_utilities import ffmpeg_stitch_video, make_gifski_gif, handle_imgs_deletion, find_ffmpeg_binary, get_ffmpeg_params
from deforum_helpers.general_utils import get_deforum_version
from deforum_helpers.upscaling import make_upscale_v2
import gc
import torch
from webui import wrap_gradio_gpu_call
import modules.shared as shared
from modules.shared import opts, cmd_opts, state
from modules.ui import create_output_panel, plaintext_to_html, wrap_gradio_call
from types import SimpleNamespace
from deforum_helpers.subtitle_handler import get_user_values
from deforum_helpers.run_deforum import run_deforum

DEBUG_MODE = opts.data.get("deforum_debug_mode_enabled", False)


def on_ui_tabs():
    with gr.Blocks(analytics_enabled=False) as deforum_interface:
        components = {}
        dummy_component = gr.Label(visible=False)
        with gr.Row(elem_id='deforum_progress_row').style(equal_height=False, variant='compact'):
            with gr.Column(scale=1, variant='panel'):
                components = deforum_args.setup_deforum_setting_dictionary(None, True, True)
        
            with gr.Column(scale=1, variant='compact'):
                with gr.Row(variant='compact'):
                    btn = gr.Button("Click here after the generation to show the video")
                    components['btn'] = btn
                    close_btn = gr.Button("Close the video", visible=False)
                with gr.Row(variant='compact'):
                    i1 = gr.HTML(deforum_args.i1_store, elem_id='deforum_header')
                    components['i1'] = i1
                    # Show video
                    def show_vid():
                        return {
                            i1: gr.update(value=deforum_args.i1_store, visible=True),
                            close_btn: gr.update(visible=True),
                            btn: gr.update(value="Update the video", visible=True),
                        }
                
                    btn.click(
                        show_vid,
                        [],
                        [i1, close_btn, btn],
                        )
                    # Close video
                    def close_vid():
                        return {
                            i1: gr.update(value=deforum_args.i1_store_backup, visible=True),
                            close_btn: gr.update(visible=False),
                            btn: gr.update(value="Click here after the generation to show the video", visible=True),
                        }
                    
                    close_btn.click(
                        close_vid,
                        [],
                        [i1, close_btn, btn],
                        )
                id_part = 'deforum'
                with gr.Row(elem_id=f"{id_part}_generate_box", variant='compact'):
                    skip = gr.Button('Pause/Resume', elem_id=f"{id_part}_skip", visible=False)
                    interrupt = gr.Button('Interrupt', elem_id=f"{id_part}_interrupt", visible=True)
                    submit = gr.Button('Generate', elem_id=f"{id_part}_generate", variant='primary')

                    skip.click(
                        fn=lambda: state.skip(),
                        inputs=[],
                        outputs=[],
                    )

                    interrupt.click(
                        fn=lambda: state.interrupt(),
                        inputs=[],
                        outputs=[],
                    )
                
                deforum_gallery, generation_info, html_info, html_log = create_output_panel("deforum", opts.outdir_img2img_samples)

                with gr.Row(variant='compact'):
                    settings_path = gr.Textbox("deforum_settings.txt", elem_id='deforum_settings_path', label="Settings File", info="settings file path can be relative to webui folder OR full - absolute")
                    #reuse_latest_settings_btn = gr.Button('Reuse Latest', elem_id='deforum_reuse_latest_settings_btn')#TODO
                with gr.Row(variant='compact'):
                    save_settings_btn = gr.Button('Save Settings', elem_id='deforum_save_settings_btn')
                    load_settings_btn = gr.Button('Load All Settings', elem_id='deforum_load_settings_btn')
                    load_video_settings_btn = gr.Button('Load Video Settings', elem_id='deforum_load_video_settings_btn')

        component_list = [components[name] for name in deforum_args.get_component_names()]

        submit.click(
                    fn=wrap_gradio_gpu_call(run_deforum, extra_outputs=[None, '', '']),
                    _js="submit_deforum",
                    inputs=[dummy_component, dummy_component] + component_list,
                    outputs=[
                         deforum_gallery,
                         components["resume_timestring"],
                         generation_info,
                         html_info,
                         html_log,
                    ],
                )
        
        settings_component_list = [components[name] for name in deforum_args.get_settings_component_names()]
        video_settings_component_list = [components[name] for name in deforum_args.video_args_names]
        stuff = gr.HTML("") # wrap gradio call garbage
        stuff.visible = False

        save_settings_btn.click(
            fn=wrap_gradio_call(deforum_settings.save_settings),
            inputs=[settings_path] + settings_component_list + video_settings_component_list,
            outputs=[stuff],
        )
        
        load_settings_btn.click(
        fn=wrap_gradio_call(lambda *args, **kwargs: deforum_settings.load_all_settings(*args, ui_launch=False, **kwargs)),
        inputs=[settings_path] + settings_component_list,
        outputs=settings_component_list + [stuff],
        )

        load_video_settings_btn.click(
            fn=wrap_gradio_call(deforum_settings.load_video_settings),
            inputs=[settings_path] + video_settings_component_list,
            outputs=video_settings_component_list + [stuff],
        )
        
    def trigger_load_general_settings():
        print("Loading general settings...")
        wrapped_fn = wrap_gradio_call(lambda *args, **kwargs: deforum_settings.load_all_settings(*args, ui_launch=True, **kwargs))
        inputs = [settings_path.value] + [component.value for component in settings_component_list]
        outputs = settings_component_list + [stuff]
        updated_values = wrapped_fn(*inputs, *outputs)[0]

        settings_component_name_to_obj = {name: component for name, component in zip(deforum_args.get_settings_component_names(), settings_component_list)}
        for key, value in updated_values.items():
            settings_component_name_to_obj[key].value = value['value']

            
    if opts.data.get("deforum_enable_persistent_settings"):
        trigger_load_general_settings()
        
    return [(deforum_interface, "Deforum", "deforum_interface")]

def on_ui_settings():
    srt_ui_params = get_user_values()
    section = ('deforum', "Deforum")
    shared.opts.add_option("deforum_keep_3d_models_in_vram", shared.OptionInfo(False, "Keep 3D models in VRAM between runs", gr.Checkbox, {"interactive": True, "visible": True if not (cmd_opts.lowvram or cmd_opts.medvram) else False}, section=section))
    shared.opts.add_option("deforum_enable_persistent_settings", shared.OptionInfo(False, "Keep settings persistent upon relaunch of webui", gr.Checkbox, {"interactive": True}, section=section))
    shared.opts.add_option("deforum_persistent_settings_path", shared.OptionInfo("models/Deforum/deforum_persistent_settings.txt", "Path for saving your persistent settings file:", section=section))
    shared.opts.add_option("deforum_ffmpeg_location", shared.OptionInfo(find_ffmpeg_binary(), "FFmpeg path/ location", section=section))
    shared.opts.add_option("deforum_ffmpeg_crf", shared.OptionInfo(17, "FFmpeg CRF value", gr.Slider, {"interactive": True, "minimum": 0, "maximum": 51}, section=section))
    shared.opts.add_option("deforum_ffmpeg_preset", shared.OptionInfo('slow', "FFmpeg Preset", gr.Dropdown, {"interactive": True, "choices": ['veryslow', 'slower', 'slow', 'medium', 'fast', 'faster', 'veryfast', 'superfast', 'ultrafast']}, section=section))
    shared.opts.add_option("deforum_debug_mode_enabled", shared.OptionInfo(False, "Enable Dev mode - adds extra reporting in console", gr.Checkbox, {"interactive": True}, section=section))
    shared.opts.add_option("deforum_save_gen_info_as_srt", shared.OptionInfo(False, "Save an .srt (subtitles) file with the generation info along with each animation", gr.Checkbox, {"interactive": True}, section=section))  
    shared.opts.add_option("deforum_save_gen_info_as_srt_params", shared.OptionInfo(['Noise Schedule'], "Choose which animation params are to be saved to the .srt file (Frame # and Seed will always be saved):", gr.CheckboxGroup, {"interactive": True, "choices": srt_ui_params}, section=section))  
        
script_callbacks.on_ui_tabs(on_ui_tabs)
script_callbacks.on_ui_settings(on_ui_settings)