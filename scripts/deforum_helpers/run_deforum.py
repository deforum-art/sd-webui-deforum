import os
import shutil
import traceback
import gc
import torch
import modules.shared as shared
from modules.processing import Processed, StableDiffusionProcessingImg2Img
from .args import get_component_names, process_args
from .deforum_tqdm import DeforumTQDM
from .save_images import dump_frames_cache, reset_frames_cache
from .frame_interpolation import process_video_interpolation
from .general_utils import get_deforum_version
from .upscaling import make_upscale_v2
from .video_audio_utilities import ffmpeg_stitch_video, make_gifski_gif, handle_imgs_deletion, get_ffmpeg_params
from pathlib import Path
from .settings import save_settings_from_animation_run

# this global param will contain the latest generated video HTML-data-URL info (for preview inside the UI when needed)
last_vid_data = None

def run_deforum(*args):
    f_location, f_crf, f_preset = get_ffmpeg_params()  # get params for ffmpeg exec
    component_names = get_component_names()
    args_dict = {component_names[i]: args[i+2] for i in range(0, len(component_names))}
    p = StableDiffusionProcessingImg2Img(
        sd_model=shared.sd_model,
        outpath_samples = shared.opts.outdir_samples or shared.opts.outdir_img2img_samples
    )  # we'll set up the rest later

    times_to_run = 1
    # find how many times in total we need to run according to file count uploaded to Batch Mode upload box
    if args_dict['custom_settings_file'] is not None and len(args_dict['custom_settings_file']) > 1:
        times_to_run = len(args_dict['custom_settings_file'])

    for i in range(times_to_run): # run for as many times as we need
        print(f"\033[4;33mDeforum extension for auto1111 webui, v2.4b\033[0m")
        print(f"Git commit: {get_deforum_version()}")
        args_dict['self'] = None
        args_dict['p'] = p
        try:
            args_loaded_ok, root, args, anim_args, video_args, parseq_args, loop_args, controlnet_args = process_args(args_dict, i)
        except Exception as e:
            print("\n*START OF TRACEBACK*")
            traceback.print_exc()
            print("*END OF TRACEBACK*\nUser friendly error message:")
            print(f"Error: {e}. Check your prompts with a JSON validator please.")
            return None, None, None, f"Error: '{e}'. Check your prompts with a JSON validator please. Full error message is in your terminal/ cli."
        if args_loaded_ok is False:
            if times_to_run > 1:
                print(f"\033[31mWARNING:\033[0m skipped running from the following setting file, as it contains an invalid JSON: {os.path.basename(args_dict['custom_settings_file'][i].name)}")
                continue
            else:
                print(f"\033[31mERROR!\033[0m Couldn't load data from '{os.path.basename(args_dict['custom_settings_file'][i].name)}'. Make sure it's a valid JSON using a JSON validator")
                return None, None, None, f"Couldn't load data from '{os.path.basename(args_dict['custom_settings_file'][i].name)}'. Make sure it's a valid JSON using a JSON validator"

        root.initial_clipskip = shared.opts.data.get("CLIP_stop_at_last_layers", 1)
        root.initial_img2img_fix_steps = shared.opts.data.get("img2img_fix_steps", False)
        root.initial_noise_multiplier = shared.opts.data.get("initial_noise_multiplier", 1.0)
        root.initial_ddim_eta = shared.opts.data.get("eta_ddim", 0.0)
        root.initial_ancestral_eta = shared.opts.data.get("eta_ancestral", 1.0)
       
        # clean up unused memory
        reset_frames_cache(root)
        gc.collect()
        torch.cuda.empty_cache()
        
        # Import them *here* or we add 3 seconds to initial webui launch-time. user doesn't feel it when we import inside the func:
        from .render import render_animation
        from .render_modes import render_input_video, render_animation_with_video_mask, render_interpolation

        tqdm_backup = shared.total_tqdm
        shared.total_tqdm = DeforumTQDM(args, anim_args, parseq_args, video_args)
        try:  # dispatch to appropriate renderer
            if anim_args.animation_mode == '2D' or anim_args.animation_mode == '3D':
                if anim_args.use_mask_video: 
                    render_animation_with_video_mask(args, anim_args, video_args, parseq_args, loop_args, controlnet_args, root)  # allow mask video without an input video
                else:    
                    render_animation(args, anim_args, video_args, parseq_args, loop_args, controlnet_args, root)
            elif anim_args.animation_mode == 'Video Input':
                render_input_video(args, anim_args, video_args, parseq_args, loop_args, controlnet_args, root)#TODO: prettify code
            elif anim_args.animation_mode == 'Interpolation':
                render_interpolation(args, anim_args, video_args, parseq_args, loop_args, controlnet_args, root)
            else:
                print('Other modes are not available yet!')
        except Exception as e:
            print("\n*START OF TRACEBACK*")
            traceback.print_exc()
            print("*END OF TRACEBACK*\n")
            print("User friendly error message:")
            print(f"Error: {e}. Check your schedules/ init values please. Also make sure you don't have a backwards slash in any of your PATHs - use / instead of \\.")
            return None, None, None, f"Error: '{e}'. Check your schedules/ init values please. Also make sure you don't have a backwards slash in any of your PATHs - use / instead of \\. Full error message is in your terminal/ cli."
        finally:
            shared.total_tqdm = tqdm_backup
            # reset shared.opts.data vals to what they were before we started the animation. Else they will stick to the last value - it actually updates webui settings (config.json)
            shared.opts.data["CLIP_stop_at_last_layers"] = root.initial_clipskip
            shared.opts.data["img2img_fix_steps"] = root.initial_img2img_fix_steps
            shared.opts.data["initial_noise_multiplier"] = root.initial_noise_multiplier
            shared.opts.data["eta_ddim"] = root.initial_ddim_eta
            shared.opts.data["eta_ancestral"] = root.initial_ancestral_eta
        
        if video_args.store_frames_in_ram:
            dump_frames_cache(root)
        
        from base64 import b64encode
        
        real_audio_track = None
        if video_args.add_soundtrack != 'None':
            real_audio_track = anim_args.video_init_path if video_args.add_soundtrack == 'Init Video' else video_args.soundtrack_path
        
        # Establish path of subtitle file
        if shared.opts.data.get("deforum_save_gen_info_as_srt", False) and shared.opts.data.get("deforum_embed_srt", False):
            srt_path = os.path.join(args.outdir, f"{root.timestring}.srt")
        else:
            srt_path = None

        # Delete folder with duplicated imgs from OS temp folder
        shutil.rmtree(root.tmp_deforum_run_duplicated_folder, ignore_errors=True)

        # Decide whether we need to try and frame interpolate later
        need_to_frame_interpolate = False
        if video_args.frame_interpolation_engine != "None" and not video_args.skip_video_creation and not video_args.store_frames_in_ram:
            need_to_frame_interpolate = True
            
        if video_args.skip_video_creation:
            print("\nSkipping video creation, uncheck 'Skip video creation' in 'Output' tab if you want to get a video too :)")
        else:
            image_path = os.path.join(args.outdir, f"{root.timestring}_%09d.png")
            mp4_path = os.path.join(args.outdir, f"{root.timestring}.mp4")
            max_video_frames = anim_args.max_frames

            # Stitch video using ffmpeg!
            try:
                f_location, f_crf, f_preset = get_ffmpeg_params() # get params for ffmpeg exec
                ffmpeg_stitch_video(ffmpeg_location=f_location, fps=video_args.fps, outmp4_path=mp4_path, stitch_from_frame=0, stitch_to_frame=max_video_frames, imgs_path=image_path, add_soundtrack=video_args.add_soundtrack, audio_path=real_audio_track, crf=f_crf, preset=f_preset, srt_path=srt_path)
                mp4 = open(mp4_path, 'rb').read()
                data_url = f"data:video/mp4;base64, {b64encode(mp4).decode()}"
                global last_vid_data
                last_vid_data = f'<p style=\"font-weight:bold;margin-bottom:0em\">Deforum extension for auto1111 — version 2.4b </p><video controls loop><source src="{data_url}" type="video/mp4"></video>'
            except Exception as e:
                if need_to_frame_interpolate:
                    print(f"FFMPEG DID NOT STITCH ANY VIDEO. However, you requested to frame interpolate  - so we will continue to frame interpolation, but you'll be left only with the interpolated frames and not a video, since ffmpeg couldn't run. Original ffmpeg error: {e}")
                else:
                    print(f"** FFMPEG DID NOT STITCH ANY VIDEO ** Error: {e}")
                pass
              
        if video_args.make_gif and not video_args.skip_video_creation and not video_args.store_frames_in_ram:
            make_gifski_gif(imgs_raw_path = args.outdir, imgs_batch_id = root.timestring, fps = video_args.fps, models_folder = root.models_path, current_user_os = root.current_user_os)
        
        # Upscale video once generation is done:
        if video_args.r_upscale_video and not video_args.skip_video_creation and not video_args.store_frames_in_ram:
            # out mp4 path is defined in make_upscale func
            make_upscale_v2(upscale_factor = video_args.r_upscale_factor, upscale_model = video_args.r_upscale_model, keep_imgs = video_args.r_upscale_keep_imgs, imgs_raw_path = args.outdir, imgs_batch_id = root.timestring, fps = video_args.fps, deforum_models_path = root.models_path, current_user_os = root.current_user_os, ffmpeg_location=f_location, stitch_from_frame=0, stitch_to_frame=max_video_frames, ffmpeg_crf=f_crf, ffmpeg_preset=f_preset, add_soundtrack = video_args.add_soundtrack ,audio_path=real_audio_track, srt_path=srt_path)

        # FRAME INTERPOLATION TIME
        if need_to_frame_interpolate: 
            print(f"Got a request to *frame interpolate* using {video_args.frame_interpolation_engine}")
            path_to_interpolate = args.outdir
            
            upscaled_folder_path = os.path.join(args.outdir, f"{root.timestring}_upscaled")
            use_upscaled_images = video_args.frame_interpolation_use_upscaled and os.path.exists(upscaled_folder_path) and len(os.listdir(upscaled_folder_path)) > 1
            if use_upscaled_images:
                print(f"Using upscaled images for frame interpolation.")
                path_to_interpolate = upscaled_folder_path
            
            ouput_vid_path = process_video_interpolation(frame_interpolation_engine=video_args.frame_interpolation_engine, frame_interpolation_x_amount=video_args.frame_interpolation_x_amount,frame_interpolation_slow_mo_enabled=video_args.frame_interpolation_slow_mo_enabled, frame_interpolation_slow_mo_amount=video_args.frame_interpolation_slow_mo_amount, orig_vid_fps=video_args.fps, deforum_models_path=root.models_path, real_audio_track=real_audio_track, raw_output_imgs_path=path_to_interpolate, img_batch_id=root.timestring, ffmpeg_location=f_location, ffmpeg_crf=f_crf, ffmpeg_preset=f_preset, keep_interp_imgs=video_args.frame_interpolation_keep_imgs, orig_vid_name=None, resolution=None, srt_path=srt_path)

            # If the interpolated video was stitched from the upscaled frames, the video needs to be moved
            # out of the upscale directory.
            if use_upscaled_images and ouput_vid_path and os.path.exists(ouput_vid_path):
                ouput_vid_path_final = os.path.join(args.outdir, Path(ouput_vid_path).stem + "_upscaled.mp4")
                print(f"Moving upscaled, interpolated vid from {ouput_vid_path} to {ouput_vid_path_final}")
                shutil.move(ouput_vid_path, ouput_vid_path_final)

        if video_args.delete_imgs and not video_args.skip_video_creation:
            handle_imgs_deletion(vid_path=mp4_path, imgs_folder_path=args.outdir, batch_id=root.timestring)
            
        root.initial_info += f"\n The animation is stored in {args.outdir}"
        reset_frames_cache(root)  # cleanup the RAM in any case
        processed = Processed(p, [root.first_frame], 0, root.initial_info)

        shared.total_tqdm.clear()

        generation_info_js = processed.js()

        if shared.opts.data.get("deforum_enable_persistent_settings", False):
            persistent_sett_path = shared.opts.data.get("deforum_persistent_settings_path")
            save_settings_from_animation_run(args, anim_args, parseq_args, loop_args, controlnet_args, video_args, root, persistent_sett_path)

    return processed.images, root.timestring, generation_info_js, processed.info
