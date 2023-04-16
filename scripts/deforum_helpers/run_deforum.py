# TODO: deduplicate basedirs
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
import json, sys, os, shutil

from modules.processing import Processed, StableDiffusionProcessingImg2Img, process_images
from PIL import Image
from deforum_helpers.video_audio_utilities import ffmpeg_stitch_video, make_gifski_gif, handle_imgs_deletion, find_ffmpeg_binary, get_ffmpeg_params
from deforum_helpers.general_utils import get_deforum_version
from deforum_helpers.upscaling import make_upscale_v2
import gc
import torch
import modules.shared as shared
from modules.shared import opts, cmd_opts, state
from modules.ui import create_output_panel, plaintext_to_html, wrap_gradio_call
from types import SimpleNamespace
from deforum_helpers.subtitle_handler import get_user_values

def run_deforum(*args):
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
        print(f"\033[4;33mDeforum extension for auto1111 webui, v2.3b\033[0m")
        print(f"Git commit: {get_deforum_version()}")
        args_dict['self'] = None
        args_dict['p'] = p

        args_loaded_ok, root, args, anim_args, video_args, parseq_args, loop_args, controlnet_args = deforum_args.process_args(args_dict, i)
        if args_loaded_ok is False:
            if times_to_run > 1:
                print(f"\033[31mWARNING:\033[0m skipped running from the following setting file, as it contains an invalid JSON: {os.path.basename(args_dict['custom_settings_file'][i].name)}")
                continue
            else:
                print(f"\033[31mERROR!\033[0m Couldn't load data from '{os.path.basename(args_dict['custom_settings_file'][i].name)}'. Make sure it's a valid JSON using a JSON validator")
                return None, None, None, None, f"Couldn't load data from '{os.path.basename(args_dict['custom_settings_file'][i].name)}'. Make sure it's a valid JSON using a JSON validator", plaintext_to_html('')

        root.clipseg_model = None
        try:
            root.initial_clipskip = opts.data["CLIP_stop_at_last_layers"]
        except:
            root.initial_clipskip = 1
        root.basedirs = basedirs

        for basedir in basedirs:
            sys.path.extend([
                basedir + '/scripts/deforum_helpers/src',
                basedir + '/extensions/deforum/scripts/deforum_helpers/src',
                basedir + '/extensions/deforum-for-automatic1111-webui/scripts/deforum_helpers/src',
            ])
        
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
        finally:
            shared.total_tqdm = tqdm_backup
            opts.data["CLIP_stop_at_last_layers"] = root.initial_clipskip
        
        if video_args.store_frames_in_ram:
            dump_frames_cache(root)
        
        from base64 import b64encode
        
        real_audio_track = None
        if video_args.add_soundtrack != 'None':
            real_audio_track = anim_args.video_init_path if video_args.add_soundtrack == 'Init Video' else video_args.soundtrack_path
        
        # Delete folder with duplicated imgs from OS temp folder
        shutil.rmtree(root.tmp_deforum_run_duplicated_folder, ignore_errors=True)

        # Decide whether or not we need to try and frame interpolate laters
        need_to_frame_interpolate = False
        if video_args.frame_interpolation_engine != "None" and not video_args.skip_video_creation and not video_args.store_frames_in_ram:
            need_to_frame_interpolate = True
            
        if video_args.skip_video_creation:
            print("\nSkipping video creation, uncheck 'Skip video for run all' in 'Output' tab if you want to get a video too :)")
        else:
            import subprocess

            path_name_modifier = video_args.path_name_modifier
            if video_args.render_steps: # render steps from a single image
                fname = f"{path_name_modifier}_%09d.png"
                all_step_dirs = [os.path.join(args.outdir, d) for d in os.listdir(args.outdir) if os.path.isdir(os.path.join(args.outdir,d))]
                newest_dir = max(all_step_dirs, key=os.path.getmtime)
                image_path = os.path.join(newest_dir, fname)
                print(f"Reading images from {image_path}")
                mp4_path = os.path.join(newest_dir, f"{args.timestring}_{path_name_modifier}.mp4")
                max_video_frames = args.steps
            else: # render images for a video
                image_path = os.path.join(args.outdir, f"{args.timestring}_%09d.png")
                mp4_path = os.path.join(args.outdir, f"{args.timestring}.mp4")
                max_video_frames = anim_args.max_frames

            # Stitch video using ffmpeg!
            try:
                f_location, f_crf, f_preset = get_ffmpeg_params() # get params for ffmpeg exec
                ffmpeg_stitch_video(ffmpeg_location=f_location, fps=video_args.fps, outmp4_path=mp4_path, stitch_from_frame=0, stitch_to_frame=max_video_frames, imgs_path=image_path, add_soundtrack=video_args.add_soundtrack, audio_path=real_audio_track, crf=f_crf, preset=f_preset)
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
            import numpy as np
            a = np.random.rand(args.W, args.H, 3)*255
            root.first_frame = Image.fromarray(a.astype('uint8')).convert('RGB')
            root.initial_seed = 6934
        # FRAME INTERPOLATION TIME
        if need_to_frame_interpolate: 
            print(f"Got a request to *frame interpolate* using {video_args.frame_interpolation_engine}")
            process_video_interpolation(frame_interpolation_engine=video_args.frame_interpolation_engine, frame_interpolation_x_amount=video_args.frame_interpolation_x_amount,frame_interpolation_slow_mo_enabled=video_args.frame_interpolation_slow_mo_enabled, frame_interpolation_slow_mo_amount=video_args.frame_interpolation_slow_mo_amount, orig_vid_fps=video_args.fps, deforum_models_path=root.models_path, real_audio_track=real_audio_track, raw_output_imgs_path=args.outdir, img_batch_id=args.timestring, ffmpeg_location=f_location, ffmpeg_crf=f_crf, ffmpeg_preset=f_preset, keep_interp_imgs=video_args.frame_interpolation_keep_imgs, orig_vid_name=None, resolution=None)
        
        if video_args.make_gif and not video_args.skip_video_creation and not video_args.store_frames_in_ram:
            make_gifski_gif(imgs_raw_path = args.outdir, imgs_batch_id = args.timestring, fps = video_args.fps, models_folder = root.models_path, current_user_os = root.current_user_os)
        
        # Upscale video once generation is done:
        if video_args.r_upscale_video and not video_args.skip_video_creation and not video_args.store_frames_in_ram:
            
            # out mp4 path is defined in make_upscale func
            make_upscale_v2(upscale_factor = video_args.r_upscale_factor, upscale_model = video_args.r_upscale_model, keep_imgs = video_args.r_upscale_keep_imgs, imgs_raw_path = args.outdir, imgs_batch_id = args.timestring, fps = video_args.fps, deforum_models_path = root.models_path, current_user_os = root.current_user_os, ffmpeg_location=f_location, stitch_from_frame=0, stitch_to_frame=max_video_frames, ffmpeg_crf=f_crf, ffmpeg_preset=f_preset, add_soundtrack = video_args.add_soundtrack ,audio_path=real_audio_track)
            
        if video_args.delete_imgs and not video_args.skip_video_creation:
            handle_imgs_deletion(vid_path=mp4_path, imgs_folder_path=args.outdir, batch_id=args.timestring)
            
        root.initial_info += "\n The animation is stored in " + args.outdir
        reset_frames_cache(root) # cleanup the RAM in any case
        processed = Processed(p, [root.first_frame], root.initial_seed, root.initial_info)
        
        if processed is None:
            processed = process_images(p)

        shared.total_tqdm.clear()

        generation_info_js = processed.js()
        if opts.samples_log_stdout:
            print(generation_info_js)

        if opts.do_not_show_images:
            processed.images = []
            
        if opts.data.get("deforum_enable_persistent_settings"):
            persistent_sett_path = opts.data.get("deforum_persistent_settings_path")
            deforum_settings.save_settings_from_animation_run(args, anim_args, parseq_args, loop_args, controlnet_args, video_args, root, persistent_sett_path)

    return processed.images, args.timestring, generation_info_js, plaintext_to_html(processed.info), plaintext_to_html('')
