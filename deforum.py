import scripts.deforum.args as deforum_args

import modules.scripts as scripts
import gradio as gr

from modules.processing import Processed
import sys
from modules.shared import opts, cmd_opts, state
from types import SimpleNamespace

class Script(scripts.Script):

    def title(self):
        return "Deforum v0.5-webui-beta"

    def ui(self, is_img2img):
        return deforum_args.setup_deforum_setting_ui(is_img2img)
    
    def deforum_main(self, p, args, anim_args, video_args):
        # clean up unused memory
        gc.collect()
        torch.cuda.empty_cache()
        
        sys.path.extend([
            'scripts/deforum/src',
        ])
        
        from scripts.deforum.render import render_animation, render_input_video, render_image_batch, render_interpolation
    

        # dispatch to appropriate renderer
        if anim_args.animation_mode == '2D' or anim_args.animation_mode == '3D':
            render_animation(args, anim_args, animation_prompts, root)
        elif anim_args.animation_mode == 'Video Input':
            render_input_video(args, anim_args, animation_prompts, root)
        elif anim_args.animation_mode == 'Interpolation':
            render_interpolation(args, anim_args, animation_prompts, root)
        else:
            render_image_batch(args, prompts, root)
        
        if video_args.skip_video_for_run_all:
            print('Skipping video creation, uncheck skip_video_for_run_all if you want to run it')
        else:
            import os
            import subprocess
            from base64 import b64encode

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
            
            # make video
            cmd = [
                'ffmpeg',
                '-y',
                '-vcodec', 'png',
                '-r', str(fps),
                '-start_number', str(0),
                '-i', image_path,
                '-frames:v', str(max_video_frames),
                '-c:v', 'libx264',
                '-vf',
                f'fps={fps}',
                '-pix_fmt', 'yuv420p',
                '-crf', '17',
                '-preset', 'veryfast',
                '-pattern_type', 'sequence',
                mp4_path
            ]
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            stdout, stderr = process.communicate()
            if process.returncode != 0:
                print(stderr)
                raise RuntimeError(stderr)

            mp4 = open(mp4_path,'rb').read()
            data_url = "data:video/mp4;base64," + b64encode(mp4).decode()
            display.display( display.HTML(f'<video controls loop><source src="{data_url}" type="video/mp4"></video>') )
    
        return Processed(p)

    def run(self, p, override_settings_with_file, custom_settings_file, animation_mode, max_frames, border, angle, zoom, translation_x, translation_y, translation_z, rotation_3d_x, rotation_3d_y, rotation_3d_z, flip_2d_perspective, perspective_flip_theta, perspective_flip_phi, perspective_flip_gamma, perspective_flip_fv, noise_schedule, strength_schedule, contrast_schedule, color_coherence, diffusion_cadence, use_depth_warping, midas_weight, near_plane, far_plane, fov, padding_mode, sampling_mode, save_depth_maps, video_init_path, extract_nth_frame, overwrite_extracted_frames, use_mask_video, video_mask_path, interpolate_key_frames, interpolate_x_frames, resume_from_timestring, resume_timestring, prompts, animation_prompts, override_webui_with_these, batch_name, filename_format, seed_behavior, use_init, from_img2img_instead_of_link, strength_0_no_init, strength, init_image, use_mask, use_alpha_as_mask, invert_mask, overlay_mask, mask_file, mask_brightness_adjust, mask_overlay_blur, skip_video_for_run_all, fps, use_manual_settings, render_steps, max_video_frames, path_name_modifier, image_path, mp4_path, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, i16, i17, i18, i19, i20, i21, i22, i23, i24, i25, i26, i27, i28, i29, i30, i31, i32, i33, i34):
        print('Hello, deforum!')
        
        args, anim_args, video_args = deforum_args.process_args(self, p, override_settings_with_file, custom_settings_file, animation_mode, max_frames, border, angle, zoom, translation_x, translation_y, translation_z, rotation_3d_x, rotation_3d_y, rotation_3d_z, flip_2d_perspective, perspective_flip_theta, perspective_flip_phi, perspective_flip_gamma, perspective_flip_fv, noise_schedule, strength_schedule, contrast_schedule, color_coherence, diffusion_cadence, use_depth_warping, midas_weight, near_plane, far_plane, fov, padding_mode, sampling_mode, save_depth_maps, video_init_path, extract_nth_frame, overwrite_extracted_frames, use_mask_video, video_mask_path, interpolate_key_frames, interpolate_x_frames, resume_from_timestring, resume_timestring, prompts, animation_prompts, override_webui_with_these, batch_name, filename_format, seed_behavior, use_init, from_img2img_instead_of_link, strength_0_no_init, strength, init_image, use_mask, use_alpha_as_mask, invert_mask, overlay_mask, mask_file, mask_brightness_adjust, mask_overlay_blur, skip_video_for_run_all, fps, use_manual_settings, render_steps, max_video_frames, path_name_modifier, image_path, mp4_path, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, i16, i17, i18, i19, i20, i21, i22, i23, i24, i25, i26, i27, i28, i29, i30, i31, i32, i33, i34)
        
        #return deforum_main(self, p, args, anim_args, video_args)
        return Processed(p)
    
    