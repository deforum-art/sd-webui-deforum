# Detach 'deforum_helpers' from 'scripts' to prevent "No module named 'scripts.deforum_helpers'"  error 
# causing Deforum's tab not show up in some cases when you've might've broken the environment with webui packages updates
import sys, os, shutil

deforum_folder_name = os.path.sep.join(os.path.abspath(__file__).split(os.path.sep)[:-2])

basedirs = [os.getcwd()]
if 'google.colab' in sys.modules:
    basedirs.append('/content/gdrive/MyDrive/sd/stable-diffusion-webui') #hardcode as TheLastBen's colab seems to be the primal source

for basedir in basedirs:
    deforum_paths_to_ensure = [
        os.path.join(deforum_folder_name, 'scripts'),
        os.path.join(deforum_folder_name, 'scripts', 'deforum_helpers', 'src')
        ]

    for deforum_scripts_path_fix in deforum_paths_to_ensure:
        if not deforum_scripts_path_fix in sys.path:
            sys.path.extend([deforum_scripts_path_fix])

# Main deforum stuff
import deforum_helpers.args as deforum_args
import deforum_helpers.settings as deforum_settings
from deforum_helpers.save_images import dump_frames_cache, reset_frames_cache
from deforum_helpers.frame_interpolation import process_video_interpolation

import modules.scripts as wscripts
from modules import script_callbacks, ui_components
import gradio as gr
import json
import traceback

from modules.processing import Processed, StableDiffusionProcessingImg2Img, process_images
from PIL import Image
from deforum_helpers.video_audio_utilities import ffmpeg_stitch_video, make_gifski_gif, handle_imgs_deletion, find_ffmpeg_binary, get_ffmpeg_params
from deforum_helpers.general_utils import get_deforum_version
from deforum_helpers.upscaling import make_upscale_v2
import gc
import numpy as np
import torch
from webui import wrap_gradio_gpu_call
import modules.shared as shared
from modules.shared import opts, cmd_opts, state
from modules.ui import create_output_panel, plaintext_to_html, wrap_gradio_call
from types import SimpleNamespace
from deforum_helpers.subtitle_handler import get_user_values

DEBUG_MODE = opts.data.get("deforum_debug_mode_enabled", False)

def run_deforum(*args, **kwargs):
    f_location, f_crf, f_preset = get_ffmpeg_params() # get params for ffmpeg exec
    component_names = deforum_args.get_component_names()
    args_dict = {component_names[i]: args[i+2] for i in range(0, len(component_names))}
    p = StableDiffusionProcessingImg2Img(
        sd_model=shared.sd_model,
        outpath_samples = opts.outdir_samples or opts.outdir_img2img_samples,
        outpath_grids = opts.outdir_grids or opts.outdir_img2img_grids
    ) #we'll setup the rest later

    times_to_run = 1
    if args_dict['custom_settings_file'] is not None and len(args_dict['custom_settings_file']) > 1:
        times_to_run = len(args_dict['custom_settings_file'])
        
    for i in range(times_to_run):
        print(f"\033[4;33mDeforum extension for auto1111 webui, v2.4b\033[0m")
        print(f"Git commit: {get_deforum_version()}")
        args_dict['self'] = None
        args_dict['p'] = p
        try:
            args_loaded_ok, root, args, anim_args, video_args, parseq_args, loop_args, controlnet_args = deforum_args.process_args(args_dict, i)
        except Exception as e:
            print("\n*START OF TRACEBACK*")
            traceback.print_exc()
            print("*END OF TRACEBACK*\n")
            print("User friendly error message:")
            print(f"Error: {e}. Check your prompts with a JSON validator please.")
            return None, None, None, None, f"Error: '{e}'. Check your prompts with a JSON validator please. Full error message is in your terminal/ cli.", plaintext_to_html('') 
        if args_loaded_ok is False:
            if times_to_run > 1:
                print(f"\033[31mWARNING:\033[0m skipped running from the following setting file, as it contains an invalid JSON: {os.path.basename(args_dict['custom_settings_file'][i].name)}")
                continue
            else:
                print(f"\033[31mERROR!\033[0m Couldn't load data from '{os.path.basename(args_dict['custom_settings_file'][i].name)}'. Make sure it's a valid JSON using a JSON validator")
                return None, None, None, None, f"Couldn't load data from '{os.path.basename(args_dict['custom_settings_file'][i].name)}'. Make sure it's a valid JSON using a JSON validator", plaintext_to_html('')

        root.clipseg_model = None
        
        root.initial_clipskip = opts.data.get("CLIP_stop_at_last_layers", 1)
        root.initial_img2img_fix_steps = opts.data.get("img2img_fix_steps", False)
        root.initial_noise_multiplier = opts.data.get("initial_noise_multiplier", 1.0)
        root.initial_ddim_eta = opts.data.get("eta_ddim", 0.0)
        root.initial_ancestral_eta = opts.data.get("eta_ancestral", 1.0)

        root.basedirs = basedirs
        for basedir in basedirs:
            sys.path.extend([os.path.join(deforum_folder_name, 'scripts', 'deforum_helpers', 'src')])
        
        # clean up unused memory
        reset_frames_cache(root)
        gc.collect()
        torch.cuda.empty_cache()

        from deforum_helpers.render import render_animation
        from deforum_helpers.render_modes import render_input_video, render_animation_with_video_mask, render_interpolation

        tqdm_backup = shared.total_tqdm
        shared.total_tqdm = deforum_settings.DeforumTQDM(args, anim_args, parseq_args, video_args)
        try: # dispatch to appropriate renderer
            if anim_args.animation_mode == '2D' or anim_args.animation_mode == '3D':
                if anim_args.use_mask_video: 
                    render_animation_with_video_mask(args, anim_args, video_args, parseq_args, loop_args, controlnet_args, root.animation_prompts, root) # allow mask video without an input video
                else:    
                    render_animation(args, anim_args, video_args, parseq_args, loop_args, controlnet_args, root.animation_prompts, root)
            elif anim_args.animation_mode == 'Video Input':
                render_input_video(args, anim_args, video_args, parseq_args, loop_args, controlnet_args, root.animation_prompts, root)#TODO: prettify code
            elif anim_args.animation_mode == 'Interpolation':
                render_interpolation(args, anim_args, video_args, parseq_args, loop_args, controlnet_args, root.animation_prompts, root)
            else:
                print('Other modes are not available yet!')
        except Exception as e:
            print("\n*START OF TRACEBACK*")
            traceback.print_exc()
            print("*END OF TRACEBACK*\n")
            print("User friendly error message:")
            print(f"Error: {e}. Check your schedules/ init values please. Also make sure you don't have a backwards slash in any of your PATHs - use / instead of \\.")
            return None, None, None, None, f"Error: '{e}'. Check your schedules/ init values please. Also make sure you don't have a backwards slash in any of your PATHs - use / instead of \\. Full error message is in your terminal/ cli.", plaintext_to_html('') 
        finally:
            shared.total_tqdm = tqdm_backup
            # reset opts.data vals to what they were before we started the animation. Else they will stick to the last value - it actually updates webui settings (config.json)
            opts.data["CLIP_stop_at_last_layers"] = root.initial_clipskip
            opts.data["img2img_fix_steps"] = root.initial_img2img_fix_steps
            opts.data["initial_noise_multiplier"] = root.initial_noise_multiplier
            opts.data["eta_ddim"] = root.initial_ddim_eta
            opts.data["eta_ancestral"] = root.initial_ancestral_eta
            
            
        
        if video_args.store_frames_in_ram:
            dump_frames_cache(root)
        
        from base64 import b64encode
        
        real_audio_track = None
        if video_args.add_soundtrack != 'None':
            real_audio_track = anim_args.video_init_path if video_args.add_soundtrack == 'Init Video' else video_args.soundtrack_path
        
        # Establish path of subtitles file
        if opts.data.get("deforum_save_gen_info_as_srt", False) and opts.data.get("deforum_embed_srt", False):
            srt_path = os.path.join(args.outdir, f"{args.timestring}.srt")
        else:
            srt_path = None

        # Delete folder with duplicated imgs from OS temp folder
        shutil.rmtree(root.tmp_deforum_run_duplicated_folder, ignore_errors=True)

        # Decide whether or not we need to try and frame interpolate laters
        need_to_frame_interpolate = False
        if video_args.frame_interpolation_engine != "None" and not video_args.skip_video_creation and not video_args.store_frames_in_ram:
            need_to_frame_interpolate = True
            
        if video_args.skip_video_creation:
            print("\nSkipping video creation, uncheck 'Skip video creation' in 'Output' tab if you want to get a video too :)")
        else:
            import subprocess # DO WE USE IT?

            image_path = os.path.join(args.outdir, f"{args.timestring}_%09d.png")
            mp4_path = os.path.join(args.outdir, f"{args.timestring}.mp4")
            max_video_frames = anim_args.max_frames

            # Stitch video using ffmpeg!
            try:
                f_location, f_crf, f_preset = get_ffmpeg_params() # get params for ffmpeg exec
                ffmpeg_stitch_video(ffmpeg_location=f_location, fps=video_args.fps, outmp4_path=mp4_path, stitch_from_frame=0, stitch_to_frame=max_video_frames, imgs_path=image_path, add_soundtrack=video_args.add_soundtrack, audio_path=real_audio_track, crf=f_crf, preset=f_preset, srt_path=srt_path)
                mp4 = open(mp4_path,'rb').read()
                data_url = "data:video/mp4;base64," + b64encode(mp4).decode()
                deforum_args.i1_store = f'<p style=\"font-weight:bold;margin-bottom:0em\">Deforum extension for auto1111 — version 2.2b </p><video controls loop><source src="{data_url}" type="video/mp4"></video>'
            except Exception as e:
                if need_to_frame_interpolate:
                    print(f"FFMPEG DID NOT STITCH ANY VIDEO. However, you requested to frame interpolate  - so we will continue to frame interpolation, but you'll be left only with the interpolated frames and not a video, since ffmpeg couldn't run. Original ffmpeg error: {e}")
                else:
                    print(f"** FFMPEG DID NOT STITCH ANY VIDEO ** Error: {e}")
                pass
                
        if root.initial_info is None:
            root.initial_info = "An error has occured and nothing has been generated!"
            root.initial_info += "\nPlease, report the bug to https://github.com/deforum-art/deforum-for-automatic1111-webui/issues"
            a = np.random.rand(args.W, args.H, 3)*255
            root.first_frame = Image.fromarray(a.astype('uint8')).convert('RGB')
            root.initial_seed = 6934
        # FRAME INTERPOLATION TIME
        if need_to_frame_interpolate: 
            print(f"Got a request to *frame interpolate* using {video_args.frame_interpolation_engine}")
            process_video_interpolation(frame_interpolation_engine=video_args.frame_interpolation_engine, frame_interpolation_x_amount=video_args.frame_interpolation_x_amount,frame_interpolation_slow_mo_enabled=video_args.frame_interpolation_slow_mo_enabled, frame_interpolation_slow_mo_amount=video_args.frame_interpolation_slow_mo_amount, orig_vid_fps=video_args.fps, deforum_models_path=root.models_path, real_audio_track=real_audio_track, raw_output_imgs_path=args.outdir, img_batch_id=args.timestring, ffmpeg_location=f_location, ffmpeg_crf=f_crf, ffmpeg_preset=f_preset, keep_interp_imgs=video_args.frame_interpolation_keep_imgs, orig_vid_name=None, resolution=None, srt_path=srt_path)
        
        if video_args.make_gif and not video_args.skip_video_creation and not video_args.store_frames_in_ram:
            make_gifski_gif(imgs_raw_path = args.outdir, imgs_batch_id = args.timestring, fps = video_args.fps, models_folder = root.models_path, current_user_os = root.current_user_os)
        
        # Upscale video once generation is done:
        if video_args.r_upscale_video and not video_args.skip_video_creation and not video_args.store_frames_in_ram:
            # out mp4 path is defined in make_upscale func
            make_upscale_v2(upscale_factor = video_args.r_upscale_factor, upscale_model = video_args.r_upscale_model, keep_imgs = video_args.r_upscale_keep_imgs, imgs_raw_path = args.outdir, imgs_batch_id = args.timestring, fps = video_args.fps, deforum_models_path = root.models_path, current_user_os = root.current_user_os, ffmpeg_location=f_location, stitch_from_frame=0, stitch_to_frame=max_video_frames, ffmpeg_crf=f_crf, ffmpeg_preset=f_preset, add_soundtrack = video_args.add_soundtrack ,audio_path=real_audio_track, srt_path=srt_path)
            
        if video_args.delete_imgs and not video_args.skip_video_creation:
            handle_imgs_deletion(vid_path=mp4_path, imgs_folder_path=args.outdir, batch_id=args.timestring)
            
        root.initial_info += "\n The animation is stored in " + args.outdir
        reset_frames_cache(root) # cleanup the RAM in any case
        processed = Processed(p, [root.first_frame], root.initial_seed, root.initial_info)
        
        if processed is None:
            processed = process_images(p)

        shared.total_tqdm.clear()

        generation_info_js = processed.js()
        if getattr(opts, 'samples_log_stdout', False):
            print(generation_info_js)

        if opts.do_not_show_images:
            processed.images = []
            
        if opts.data.get("deforum_enable_persistent_settings"):
            persistent_sett_path = opts.data.get("deforum_persistent_settings_path")
            deforum_settings.save_settings_from_animation_run(args, anim_args, parseq_args, loop_args, controlnet_args, video_args, root, persistent_sett_path)

    return processed.images, args.timestring, generation_info_js, plaintext_to_html(processed.info), plaintext_to_html('')

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
    shared.opts.add_option("deforum_embed_srt", shared.OptionInfo(False, "If .srt file is saved, soft-embed the subtitles into the rendered video file", gr.Checkbox, {"interactive": True}, section=section))  
    shared.opts.add_option("deforum_save_gen_info_as_srt_params", shared.OptionInfo(['Noise Schedule'], "Choose which animation params are to be saved to the .srt file (Frame # and Seed will always be saved):", ui_components.DropdownMulti, lambda: {"interactive": True, "choices": srt_ui_params}, section=section)) 
        
script_callbacks.on_ui_tabs(on_ui_tabs)
script_callbacks.on_ui_settings(on_ui_settings)
