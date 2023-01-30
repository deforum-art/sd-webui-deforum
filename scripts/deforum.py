# Detach 'deforum_helpers' from 'scripts' to prevent "No module named 'scripts.deforum_helpers'"  error 
# causing Deforum's tab not show up in some cases when you've might've broken the environment with webui packages updates
import sys, os

basedirs = [os.getcwd()]
if 'google.colab' in sys.modules:
    basedirs.append('/content/gdrive/MyDrive/sd/stable-diffusion-webui') #hardcode as TheLastBen's colab seems to be the primal source

for basedir in basedirs:
    deforum_paths_to_ensure = [basedir + '/extensions/deforum-for-automatic1111-webui/scripts', basedir + '/extensions/deforum/scripts', basedir + '/scripts/deforum_helpers/src', basedir + '/extensions/deforum/scripts/deforum_helpers/src', basedir +'/extensions/deforum-for-automatic1111-webui/scripts/deforum_helpers/src',basedir]# + '/scripts/deforum_helpers/src/rife']

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
import gc
import torch
from webui import wrap_gradio_gpu_call
import modules.shared as shared
from modules.shared import opts, cmd_opts, state
from modules.ui import create_output_panel, plaintext_to_html, wrap_gradio_call
from types import SimpleNamespace

def run_deforum(*args, **kwargs):
    args_dict = {deforum_args.component_names[i]: args[i+2] for i in range(0, len(deforum_args.component_names))}
    p = StableDiffusionProcessingImg2Img(
        sd_model=shared.sd_model,
        outpath_samples = opts.outdir_samples or opts.outdir_img2img_samples,
        outpath_grids = opts.outdir_grids or opts.outdir_img2img_grids,
        #we'll setup the rest later
    )
    
    print('Deforum script for 2D, pseudo-2D and 3D animations')
    print('v0.5-webui-beta')
    
    args_dict['self'] = None
    args_dict['p'] = p
    
    root, args, anim_args, video_args, parseq_args, loop_args = deforum_args.process_args(args_dict)
    root.clipseg_model = None
    root.basedirs = basedirs

    # Install numexpr as it's the thing most people are having problems with
    from launch import is_installed, run_pip
    if not is_installed("numexpr"):
        run_pip("install numexpr", "numexpr")
    
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
    shared.total_tqdm = deforum_settings.DeforumTQDM(args, anim_args, parseq_args)
    try:
        # dispatch to appropriate renderer
        if anim_args.animation_mode == '2D' or anim_args.animation_mode == '3D':
            if anim_args.use_mask_video: 
                render_animation_with_video_mask(args, anim_args, video_args, parseq_args, loop_args, root.animation_prompts, root) # allow mask video without an input video
            else:    
                render_animation(args, anim_args, video_args, parseq_args, loop_args, root.animation_prompts, root)
        elif anim_args.animation_mode == 'Video Input':
            render_input_video(args, anim_args, video_args, parseq_args, loop_args, root.animation_prompts, root)#TODO: prettify code
        elif anim_args.animation_mode == 'Interpolation':
            render_interpolation(args, anim_args, video_args, parseq_args, loop_args, root.animation_prompts, root)
        else:
            print('Other modes are not available yet!')
    finally:
        shared.total_tqdm = tqdm_backup
    
    if video_args.store_frames_in_ram:
        dump_frames_cache(root)
    
    from base64 import b64encode
    
    real_audio_track = None
    if video_args.add_soundtrack != 'None':
        real_audio_track = anim_args.video_init_path if video_args.add_soundtrack == 'Init Video' else video_args.soundtrack_path
    
    if video_args.skip_video_for_run_all:
        print('Skipping video creation, uncheck skip_video_for_run_all if you want to run it')
    elif video_args.output_format == 'FFMPEG mp4':
        import subprocess

        if video_args.use_manual_settings:
            max_video_frames = video_args.max_video_frames #@param {type:"string"}
            image_path = video_args.image_path
            mp4_path = video_args.mp4_path
        else:
            path_name_modifier = video_args.path_name_modifier
            if video_args.render_steps: # render steps from a single image
                fname = f"{path_name_modifier}_%05d.png"
                all_step_dirs = [os.path.join(args.outdir, d) for d in os.listdir(args.outdir) if os.path.isdir(os.path.join(args.outdir,d))]
                newest_dir = max(all_step_dirs, key=os.path.getmtime)
                image_path = os.path.join(newest_dir, fname)
                print(f"Reading images from {image_path}")
                mp4_path = os.path.join(newest_dir, f"{args.timestring}_{path_name_modifier}.mp4")
                max_video_frames = args.steps
            else: # render images for a video
                image_path = os.path.join(args.outdir, f"{args.timestring}_%05d.png")
                mp4_path = os.path.join(args.outdir, f"{args.timestring}.mp4")
                max_video_frames = anim_args.max_frames

        print(f"{image_path} -> {mp4_path}")

        #save settings for the video
        video_settings_filename = os.path.join(args.outdir, f"{args.timestring}_video-settings.txt")
        with open(video_settings_filename, "w+", encoding="utf-8") as f:
            s = {**dict(video_args.__dict__)}
            json.dump(s, f, ensure_ascii=False, indent=4)
        # make video
        cmd = [
            video_args.ffmpeg_location,
            '-y',
            '-vcodec', 'png',
            '-r', str(int(video_args.fps)),
            '-start_number', str(0),
            '-i', image_path,
            '-frames:v', str(max_video_frames),
            '-c:v', 'libx264',
            '-vf',
            f'fps={int(video_args.fps)}',
            '-pix_fmt', 'yuv420p',
            '-crf', str(video_args.ffmpeg_crf),
            '-preset', video_args.ffmpeg_preset,
            '-pattern_type', 'sequence',
            mp4_path
        ]
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = process.communicate()
        if process.returncode != 0:
            print(stderr)
            raise RuntimeError(stderr)
        
        if video_args.add_soundtrack != 'None':
            cmd = [
                video_args.ffmpeg_location,
                '-i',
                mp4_path,
                '-i',
                real_audio_track,
                '-map', '0:v',
                '-map', '1:a',
                '-c:v', 'copy',
                '-shortest',
                mp4_path+'.temp.mp4'
            ]
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            stdout, stderr = process.communicate()
            if process.returncode != 0:
                print(stderr)
                raise RuntimeError(stderr)
            os.replace(mp4_path+'.temp.mp4', mp4_path)

        mp4 = open(mp4_path,'rb').read()
        data_url = "data:video/mp4;base64," + b64encode(mp4).decode()
        
        deforum_args.i1_store = f'<p style=\"font-weight:bold;margin-bottom:0.75em\">Deforum v0.5-webui-beta</p><video controls loop><source src="{data_url}" type="video/mp4"></video>'

    else: # *GIF* TIME!
        # TODO: add support for custom frame interpolation vid location?
        if video_args.use_manual_settings:
            max_video_frames = video_args.max_video_frames #@param {type:"string"}
            image_path = video_args.image_path
            mp4_path = video_args.mp4_path
        else:
            path_name_modifier = video_args.path_name_modifier
            if video_args.render_steps: # render steps from a single image
                fname = f"{path_name_modifier}_%05d.png"
                all_step_dirs = [os.path.join(args.outdir, d) for d in os.listdir(args.outdir) if os.path.isdir(os.path.join(args.outdir,d))]
                newest_dir = max(all_step_dirs, key=os.path.getmtime)
                image_path = os.path.join(newest_dir, fname)
                print(f"Reading images from {image_path}")
                mp4_path = os.path.join(newest_dir, f"{args.timestring}_{path_name_modifier}.gif")
                max_video_frames = args.steps
            else: # render images for a video
                image_path = os.path.join(args.outdir, f"{args.timestring}_%05d.png")
                mp4_path = os.path.join(args.outdir, f"{args.timestring}.gif")
                max_video_frames = anim_args.max_frames

        print(f"{image_path} -> {mp4_path}")

        #save settings for the video
        video_settings_filename = os.path.join(args.outdir, f"{args.timestring}_video-settings.txt")
        with open(video_settings_filename, "w+", encoding="utf-8") as f:
            s = {**dict(video_args.__dict__)}
            json.dump(s, f, ensure_ascii=False, indent=4)
        
        imagelist = [Image.open(os.path.join(args.outdir, image_path%d)) for d in range(max_video_frames) if os.path.exists(os.path.join(args.outdir, image_path%d))]
        
        imagelist[0].save(
            mp4_path,#gif here
            save_all=True,
            append_images=imagelist[1:],
            optimize=True,
            duration=1000/video_args.fps, #????
            loop=0
        )
        
        mp4 = open(mp4_path,'rb').read()
        data_url = "data:image/gif;base64," + b64encode(mp4).decode()
        
        deforum_args.i1_store = f'<p style=\"font-weight:bold;margin-bottom:0.75em\">Deforum v0.5-webui-beta</p><img src="{data_url}" type="image/gif"></img>'

    if root.initial_info is None:
        root.initial_info = "An error has occured and nothing has been generated!"
        root.initial_info += "\nPlease, report the bug to https://github.com/deforum-art/deforum-for-automatic1111-webui/issues"
        import numpy as np
        a = np.random.rand(args.W, args.H, 3)*255
        root.first_frame = Image.fromarray(a.astype('uint8')).convert('RGB')
        root.initial_seed = 6934
    # FRMAE INTERPOLATION TIME
    if video_args.frame_interpolation_x_amount != "Disabled" and not video_args.skip_video_for_run_all and not video_args.store_frames_in_ram:
        print(f"Got a request to *frame interpolate* using {video_args.frame_interpolation_engine}")
        # todo: make it 1=2,3=4 instead of 1,2,3,4
        process_video_interpolation(video_args.frame_interpolation_engine, video_args.frame_interpolation_x_amount, video_args.frame_interpolation_slow_mo_amount, video_args.fps, root.models_path, real_audio_track, args.outdir, args.timestring, video_args.ffmpeg_location, video_args.ffmpeg_crf, video_args.ffmpeg_preset, video_args.frame_interpolation_keep_imgs, None)
        
    root.initial_info += "\n The animation is stored in " + args.outdir + '\n'
    root.initial_info += "Only the first frame is shown in webui not to clutter the memory"
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

    return processed.images, generation_info_js, plaintext_to_html(processed.info), plaintext_to_html('#TODO')

def on_ui_tabs():
    with gr.Blocks(analytics_enabled=False) as deforum_interface:
        components = {}
        dummy_component = gr.Label(visible=False)
        with gr.Row(elem_id='deforum_progress_row').style(equal_height=False):
            with gr.Column(scale=1, variant='panel'):
                components = deforum_args.setup_deforum_setting_dictionary(None, True, True)
        
            with gr.Column(scale=1):
                with gr.Row():
                    btn = gr.Button("Click here after the generation to show the video")
                    components['btn'] = btn
                    close_btn = gr.Button("Close the video", visible=False)
                with gr.Row():
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
                with gr.Row(elem_id=f"{id_part}_generate_box"):
                    skip = gr.Button('Skip', elem_id=f"{id_part}_skip", visible=True)
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

                
                gr.HTML("<p>* Paths can be relative to webui folder OR full - absolute </p>")
                with gr.Row():
                    settings_path = gr.Textbox("deforum_settings.txt", elem_id='deforum_settings_path', label="General Settings File")
                    #reuse_latest_settings_btn = gr.Button('Reuse Latest', elem_id='deforum_reuse_latest_settings_btn')#TODO
                with gr.Row():
                    save_settings_btn = gr.Button('Save Settings', elem_id='deforum_save_settings_btn')
                    load_settings_btn = gr.Button('Load Settings', elem_id='deforum_load_settings_btn')
                with gr.Row():
                    video_settings_path = gr.Textbox("deforum_video-settings.txt", elem_id='deforum_video_settings_path', label="Video Settings File")
                    #reuse_latest_video_settings_btn = gr.Button('Reuse Latest', elem_id='deforum_reuse_latest_video_settings_btn')#TODO
                with gr.Row():
                    save_video_settings_btn = gr.Button('Save Video Settings', elem_id='deforum_save_video_settings_btn')
                    load_video_settings_btn = gr.Button('Load Video Settings', elem_id='deforum_load_video_settings_btn')

                # components['prompts'].visible = False#hide prompts for the time being
                #TODO clean up the code
                components['save_sample_per_step'].visible = False
                components['show_sample_per_step'].visible = False
                components['display_samples'].visible = False

               
        component_list = [components[name] for name in deforum_args.component_names]

        submit.click(
                    fn=wrap_gradio_gpu_call(run_deforum, extra_outputs=[None, '', '']),
                    _js="submit_deforum",
                    inputs=[dummy_component, dummy_component] + component_list,
                    outputs=[
                         deforum_gallery,
                         generation_info,
                         html_info,
                         html_log,
                    ],
                )
        
        settings_component_list = [components[name] for name in deforum_args.settings_component_names]
        video_settings_component_list = [components[name] for name in deforum_args.video_args_names]
        stuff = gr.HTML("") # wrap gradio call garbage
        stuff.visible = False
        
        save_settings_btn.click(
                    fn=wrap_gradio_call(deforum_settings.save_settings),
                    inputs=[settings_path] + settings_component_list,
                    outputs=[stuff],
                )
        
        load_settings_btn.click(
                    fn=wrap_gradio_call(deforum_settings.load_settings),
                    inputs=[settings_path]+ settings_component_list,
                    outputs=settings_component_list + [stuff],
                )
        
        save_video_settings_btn.click(
                    fn=wrap_gradio_call(deforum_settings.save_video_settings),
                    inputs=[video_settings_path] + video_settings_component_list,
                    outputs=[stuff],
                )
        
        load_video_settings_btn.click(
                    fn=wrap_gradio_call(deforum_settings.load_video_settings),
                    inputs=[video_settings_path] + video_settings_component_list,
                    outputs=video_settings_component_list + [stuff],
                )


    return [(deforum_interface, "Deforum", "deforum_interface")]

script_callbacks.on_ui_tabs(on_ui_tabs)
