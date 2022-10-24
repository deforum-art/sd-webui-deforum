import os
import json

def load_args(args_dict,anim_args_dict, custom_settings_file, root):
    print(f"reading custom settings from {custom_settings_file}")
    if not os.path.isfile(custom_settings_file):
        print('The custom settings file does not exist. The in-notebook settings will be used instead')
    else:
        with open(custom_settings_file, "r") as f:
            jdata = json.loads(f.read())
            root.animation_prompts = jdata["prompts"]
            for i, k in enumerate(args_dict):
                if k in jdata:
                    args_dict[k] = jdata[k]
                else:
                    print(f"key {k} doesn't exist in the custom settings data! using the default value of {args_dict[k]}")
            for i, k in enumerate(anim_args_dict):
                if k in jdata:
                    anim_args_dict[k] = jdata[k]
                else:
                    print(f"key {k} doesn't exist in the custom settings data! using the default value of {anim_args_dict[k]}")
            print(args_dict)
            print(anim_args_dict)

import gradio as gr

# In gradio gui settings save/load
def save_settings(settings_path, override_settings_with_file, custom_settings_file, animation_mode, max_frames, border, angle, zoom, translation_x, translation_y, translation_z, rotation_3d_x, rotation_3d_y, rotation_3d_z, flip_2d_perspective, perspective_flip_theta, perspective_flip_phi, perspective_flip_gamma, perspective_flip_fv, noise_schedule, strength_schedule, contrast_schedule, cfg_scale_schedule, fov_schedule, near_schedule, far_schedule, seed_schedule, color_coherence, diffusion_cadence, use_depth_warping, midas_weight, near_plane, far_plane, fov, padding_mode, sampling_mode, save_depth_maps, video_init_path, extract_nth_frame, overwrite_extracted_frames, use_mask_video, video_mask_path, interpolate_key_frames, interpolate_x_frames, resume_from_timestring, resume_timestring, prompts, animation_prompts, W, H, restore_faces, tiling, enable_hr, firstphase_width, firstphase_height, seed, sampler, seed_enable_extras, subseed_strength, seed_resize_from_w, seed_resize_from_h, steps, ddim_eta, n_batch, make_grid, grid_rows, save_settings, save_samples, display_samples, save_sample_per_step, show_sample_per_step, override_these_with_webui, batch_name, filename_format, seed_behavior, use_init, from_img2img_instead_of_link, strength_0_no_init, strength, init_image, use_mask, use_alpha_as_mask, invert_mask, overlay_mask, mask_file, mask_brightness_adjust, mask_overlay_blur):
    from scripts.deforum.args import pack_args, pack_anim_args
    args_dict = pack_args(W, H, restore_faces, tiling, enable_hr, firstphase_width, firstphase_height, seed, sampler, seed_enable_extras, subseed_strength, seed_resize_from_w, seed_resize_from_h, steps, ddim_eta, n_batch, make_grid, grid_rows, save_settings, save_samples, display_samples, save_sample_per_step, show_sample_per_step, override_these_with_webui, batch_name, filename_format, seed_behavior, use_init, from_img2img_instead_of_link, strength_0_no_init, strength, init_image, use_mask, use_alpha_as_mask, invert_mask, overlay_mask, mask_file, mask_brightness_adjust, mask_overlay_blur)
    anim_args_dict = pack_anim_args(animation_mode, max_frames, border, angle, zoom, translation_x, translation_y, translation_z, rotation_3d_x, rotation_3d_y, rotation_3d_z, flip_2d_perspective, perspective_flip_theta, perspective_flip_phi, perspective_flip_gamma, perspective_flip_fv, noise_schedule, strength_schedule, contrast_schedule, cfg_scale_schedule, fov_schedule, near_schedule, far_schedule, seed_schedule, color_coherence, diffusion_cadence, use_depth_warping, midas_weight, near_plane, far_plane, fov, padding_mode, sampling_mode, save_depth_maps, video_init_path, extract_nth_frame, overwrite_extracted_frames, use_mask_video, video_mask_path, interpolate_key_frames, interpolate_x_frames, resume_from_timestring, resume_timestring)
    arg_dict.prompts = json.loads(animation_prompts)
    print(f"saving custom settings to {settings_path}")
    with open(settings_path, "w") as f:
        f.write(json.dumps({**args_dict, **anim_args_dict}))

def save_video_settings(video_settings_path, skip_video_for_run_all, fps, output_format, ffmpeg_location, add_soundtrack, soundtrack_path, use_manual_settings, render_steps, max_video_frames, path_name_modifier, image_path, mp4_path):
    from scripts.deforum.args import pack_video_args
    video_args_dict = pack_video_args(skip_video_for_run_all, fps, output_format, ffmpeg_location, add_soundtrack, soundtrack_path, use_manual_settings, render_steps, max_video_frames, path_name_modifier, image_path, mp4_path)
    print(f"saving video settings to {video_settings_path}")
    with open(video_settings_path, "w") as f:
        f.write(json.dumps(video_args_dict))

def load_settings(settings_path):
    print(f"reading custom settings from {settings_path}")
    jdata = {}
    if not os.path.isfile(settings_path):
        print('The custom settings file does not exist. The values will be unchanged.')
        #return {override_settings_with_file, custom_settings_file, animation_mode, max_frames, border, angle, zoom, translation_x, translation_y, translation_z, rotation_3d_x, rotation_3d_y, rotation_3d_z, flip_2d_perspective, perspective_flip_theta, perspective_flip_phi, perspective_flip_gamma, perspective_flip_fv, noise_schedule, strength_schedule, contrast_schedule, cfg_scale_schedule, fov_schedule, near_schedule, far_schedule, seed_schedule, color_coherence, diffusion_cadence, use_depth_warping, midas_weight, near_plane, far_plane, fov, padding_mode, sampling_mode, save_depth_maps, video_init_path, extract_nth_frame, overwrite_extracted_frames, use_mask_video, video_mask_path, interpolate_key_frames, interpolate_x_frames, resume_from_timestring, resume_timestring, prompts, animation_prompts, W, H, restore_faces, tiling, enable_hr, firstphase_width, firstphase_height, seed, sampler, seed_enable_extras, subseed_strength, seed_resize_from_w, seed_resize_from_h, steps, ddim_eta, n_batch, make_grid, grid_rows, save_settings, save_samples, display_samples, save_sample_per_step, show_sample_per_step, override_these_with_webui, batch_name, filename_format, seed_behavior, use_init, from_img2img_instead_of_link, strength_0_no_init, strength, init_image, use_mask, use_alpha_as_mask, invert_mask, overlay_mask, mask_file, mask_brightness_adjust, mask_overlay_blur}
        return {}
    else:
        with open(settings_path, "r") as f:
            jdata = json.loads(f.read())
    ret = {}

    if 'animation_mode' in jdata:
        ret[animation_mode] = gr.update(value=jdata['animation_mode'])
    if 'max_frames' in jdata:
        ret[max_frames] = gr.update(value=jdata['max_frames'])
    if 'border' in jdata:
        ret[border] = gr.update(value=jdata['border'])
    if 'angle' in jdata:
        ret[angle] = gr.update(value=jdata['angle'])
    if 'zoom' in jdata:
        ret[zoom] = gr.update(value=jdata['zoom'])
    if 'translation_x' in jdata:
        ret[translation_x] = gr.update(value=jdata['translation_x'])
    if 'translation_y' in jdata:
        ret[translation_y] = gr.update(value=jdata['translation_y'])
    if 'translation_z' in jdata:
        ret[translation_z] = gr.update(value=jdata['translation_z'])
    if 'rotation_3d_x' in jdata:
        ret[rotation_3d_x] = gr.update(value=jdata['rotation_3d_x'])
    if 'rotation_3d_y' in jdata:
        ret[rotation_3d_y] = gr.update(value=jdata['rotation_3d_y'])
    if 'rotation_3d_z' in jdata:
        ret[rotation_3d_z] = gr.update(value=jdata['rotation_3d_z'])
    if 'flip_2d_perspective' in jdata:
        ret[flip_2d_perspective] = gr.update(value=jdata['flip_2d_perspective'])
    if 'perspective_flip_theta' in jdata:
        ret[perspective_flip_theta] = gr.update(value=jdata['perspective_flip_theta'])
    if 'perspective_flip_phi' in jdata:
        ret[perspective_flip_phi] = gr.update(value=jdata['perspective_flip_phi'])
    if 'perspective_flip_gamma' in jdata:
        ret[perspective_flip_gamma] = gr.update(value=jdata['perspective_flip_gamma'])
    if 'perspective_flip_fv' in jdata:
        ret[perspective_flip_fv] = gr.update(value=jdata['perspective_flip_fv'])
    if 'noise_schedule' in jdata:
        ret[noise_schedule] = gr.update(value=jdata['noise_schedule'])
    if 'strength_schedule' in jdata:
        ret[strength_schedule] = gr.update(value=jdata['strength_schedule'])
    if 'contrast_schedule' in jdata:
        ret[contrast_schedule] = gr.update(value=jdata['contrast_schedule'])
    if 'cfg_scale_schedule' in jdata:
        ret[cfg_scale_schedule] = gr.update(value=jdata['cfg_scale_schedule'])
    if 'fov_schedule' in jdata:
        ret[fov_schedule] = gr.update(value=jdata['fov_schedule'])
    if 'near_schedule' in jdata:
        ret[near_schedule] = gr.update(value=jdata['near_schedule'])
    if 'far_schedule' in jdata:
        ret[far_schedule] = gr.update(value=jdata['far_schedule'])
    if 'seed_schedule' in jdata:
        ret[seed_schedule] = gr.update(value=jdata['seed_schedule'])
    if 'color_coherence' in jdata:
        ret[color_coherence] = gr.update(value=jdata['color_coherence'])
    if 'diffusion_cadence' in jdata:
        ret[diffusion_cadence] = gr.update(value=jdata['diffusion_cadence'])
    if 'use_depth_warping' in jdata:
        ret[use_depth_warping] = gr.update(value=jdata['use_depth_warping'])
    if 'midas_weight' in jdata:
        ret[midas_weight] = gr.update(value=jdata['midas_weight'])
    #near_plane, far_plane, fov, padding_mode, sampling_mode, save_depth_maps, video_init_path, extract_nth_frame, overwrite_extracted_frames, use_mask_video, video_mask_path, interpolate_key_frames, interpolate_x_frames, resume_from_timestring, resume_timestring, prompts, animation_prompts, W, H, restore_faces, tiling, enable_hr, firstphase_width, firstphase_height, seed, sampler, seed_enable_extras, subseed_strength, seed_resize_from_w, seed_resize_from_h, steps, ddim_eta, n_batch
    if 'near_plane' in jdata:
        ret[near_plane] = gr.update(value=jdata['near_plane'])
    if 'far_plane' in jdata:
        ret[far_plane] = gr.update(value=jdata['far_plane'])
    if 'fov' in jdata:
        ret[fov] = gr.update(value=jdata['fov'])
    if 'padding_mode' in jdata:
        ret[padding_mode] = gr.update(value=jdata['padding_mode'])
    if 'sampling_mode' in jdata:
        ret[sampling_mode] = gr.update(value=jdata['sampling_mode'])
    if 'save_depth_maps' in jdata:
        ret[save_depth_maps] = gr.update(value=jdata['save_depth_maps'])
    if 'video_init_path' in jdata:
        ret[video_init_path] = gr.update(value=jdata['video_init_path'])
    if 'extract_nth_frame' in jdata:
        ret[extract_nth_frame] = gr.update(value=jdata['extract_nth_frame'])
    if 'overwrite_extracted_frames' in jdata:
        ret[overwrite_extracted_frames] = gr.update(value=jdata['overwrite_extracted_frames'])
    if 'use_mask_video' in jdata:
        ret[use_mask_video] = gr.update(value=jdata['use_mask_video'])
    if 'video_mask_path' in jdata:
        ret[video_mask_path] = gr.update(value=jdata['video_mask_path'])
    if 'interpolate_key_frames' in jdata:
        ret[interpolate_key_frames] = gr.update(value=jdata['interpolate_key_frames'])
    if 'interpolate_x_frames' in jdata:
        ret[interpolate_x_frames] = gr.update(value=jdata['interpolate_x_frames'])
    if 'resume_from_timestring' in jdata:
        ret[resume_from_timestring] = gr.update(value=jdata['resume_from_timestring'])
    if 'resume_timestring' in jdata:
        ret[resume_timestring] = gr.update(value=jdata['resume_timestring'])
    if 'prompts' in jdata:
        ret[prompts] = gr.update(value=jdata['prompts'])
        ret[animation_prompts] = gr.update(value=jdata['animation_prompts'])#compatibility with old versions
    if 'animation_prompts' in jdata:
        ret[animation_prompts] = gr.update(value=jdata['animation_prompts'])
    if 'W' in jdata:
        ret[W] = gr.update(value=jdata['W'])
    if 'H' in jdata:
        ret[H] = gr.update(value=jdata['H'])
    if 'restore_faces' in jdata:
        ret[restore_faces] = gr.update(value=jdata['restore_faces'])
    if 'tiling' in jdata:
        ret[tiling] = gr.update(value=jdata['tiling'])
    if 'enable_hr' in jdata:
        ret[enable_hr] = gr.update(value=jdata['enable_hr'])
    if 'firstphase_width' in jdata:
        ret[firstphase_width] = gr.update(value=jdata['firstphase_width'])
    if 'firstphase_height' in jdata:
        ret[firstphase_height] = gr.update(value=jdata['firstphase_height'])
    if 'seed' in jdata:
        ret[seed] = gr.update(value=jdata['seed'])
    if 'sampler' in jdata:
        ret[sampler] = gr.update(value=jdata['sampler'])
    if 'seed_enable_extras' in jdata:
        ret[seed_enable_extras] = gr.update(value=jdata['seed_enable_extras'])
    if 'subseed_strength' in jdata:
        ret[subseed_strength] = gr.update(value=jdata['subseed_strength'])
    if 'seed_resize_from_w' in jdata:
        ret[seed_resize_from_w] = gr.update(value=jdata['seed_resize_from_w'])
    if 'seed_resize_from_h' in jdata:
        ret[seed_resize_from_h] = gr.update(value=jdata['seed_resize_from_h'])
    if 'steps' in jdata:
        ret[steps] = gr.update(value=jdata['steps'])
    if 'ddim_eta' in jdata:
        ret[ddim_eta] = gr.update(value=jdata['ddim_eta'])
    if 'n_batch' in jdata:
        ret[n_batch] = gr.update(value=jdata['n_batch'])
    if 'make_grid' in jdata:
        ret[make_grid] = gr.update(value=jdata['make_grid'])    
    #make_grid, grid_rows, save_settings, save_samples, display_samples, save_sample_per_step, show_sample_per_step, override_these_with_webui, batch_name, filename_format, seed_behavior, use_init, from_img2img_instead_of_link, strength_0_no_init, strength, init_image, use_mask, use_alpha_as_mask, invert_mask, overlay_mask, mask_file, mask_brightness_adjust, mask_overlay_blur
    if 'grid_rows' in jdata:
        ret[grid_rows] = gr.update(value=jdata['grid_rows'])
    if 'save_settings' in jdata:
        ret[save_settings] = gr.update(value=jdata['save_settings'])
    if 'save_samples' in jdata:
        ret[save_samples] = gr.update(value=jdata['save_samples'])
    if 'display_samples' in jdata:
        ret[display_samples] = gr.update(value=jdata['display_samples'])
    if 'save_sample_per_step' in jdata:
        ret[save_sample_per_step] = gr.update(value=jdata['save_sample_per_step'])
    if 'show_sample_per_step' in jdata:
        ret[show_sample_per_step] = gr.update(value=jdata['show_sample_per_step'])
    if 'override_these_with_webui' in jdata:
        ret[override_these_with_webui] = gr.update(value=jdata['override_these_with_webui'])
    if 'batch_name' in jdata:
        ret[batch_name] = gr.update(value=jdata['batch_name'])
    if 'filename_format' in jdata:
        ret[filename_format] = gr.update(value=jdata['filename_format'])
    if 'seed_behavior' in jdata:
        ret[seed_behavior] = gr.update(value=jdata['seed_behavior'])
    if 'use_init' in jdata:
        ret[use_init] = gr.update(value=jdata['use_init'])
    if 'from_img2img_instead_of_link' in jdata:
        ret[from_img2img_instead_of_link] = gr.update(value=jdata['from_img2img_instead_of_link'])
    if 'strength_0_no_init' in jdata:
        ret[strength_0_no_init] = gr.update(value=jdata['strength_0_no_init'])
    if 'strength' in jdata:
        ret[strength] = gr.update(value=jdata['strength'])
    if 'init_image' in jdata:
        ret[init_image] = gr.update(value=jdata['init_image'])
    if 'use_mask' in jdata:
        ret[use_mask] = gr.update(value=jdata['use_mask'])
    if 'use_alpha_as_mask' in jdata:
        ret[use_alpha_as_mask] = gr.update(value=jdata['use_alpha_as_mask'])
    if 'invert_mask' in jdata:
        ret[invert_mask] = gr.update(value=jdata['invert_mask'])
    if 'overlay_mask' in jdata:
        ret[overlay_mask] = gr.update(value=jdata['overlay_mask'])
    if 'mask_file' in jdata:
        ret[mask_file] = gr.update(value=jdata['mask_file'])
    if 'mask_brightness_adjust' in jdata:
        ret[mask_brightness_adjust] = gr.update(value=jdata['mask_brightness_adjust'])
    if 'mask_overlay_blur' in jdata:
        ret[mask_overlay_blur] = gr.update(value=jdata['mask_overlay_blur'])
    
    return ret

def load_video_settings(video_settings_path):  
    print(f"reading custom video settings from {video_settings_path}")
    jdata = {}
    if not os.path.isfile(video_settings_path):
        print('The custom video settings file does not exist. The values will be unchanged.')
        #return {skip_video_for_run_all, fps, output_format, ffmpeg_location, add_soundtrack, soundtrack_path, use_manual_settings, render_steps, max_video_frames, path_name_modifier, image_path, mp4_path}
        return {}
    else:
        with open(video_settings_path, "r") as f:
            jdata = json.loads(f.read())
    ret = {}

    if 'skip_video_for_run_all' in jdata:
        ret[skip_video_for_run_all] = gr.update(value=jdata['skip_video_for_run_all'])
    if 'fps' in jdata:
        ret[fps] = gr.update(value=jdata['fps'])
    if 'output_format' in jdata:
        ret[output_format] = gr.update(value=jdata['output_format'])
    if 'ffmpeg_location' in jdata:
        ret[ffmpeg_location] = gr.update(value=jdata['ffmpeg_location'])
    if 'add_soundtrack' in jdata:
        ret[add_soundtrack] = gr.update(value=jdata['add_soundtrack'])
    if 'soundtrack_path' in jdata:
        ret[soundtrack_path] = gr.update(value=jdata['soundtrack_path'])
    if 'use_manual_settings' in jdata:
        ret[use_manual_settings] = gr.update(value=jdata['use_manual_settings'])
    if 'render_steps' in jdata:
        ret[render_steps] = gr.update(value=jdata['render_steps'])
    if 'max_video_frames' in jdata:
        ret[max_video_frames] = gr.update(value=jdata['max_video_frames'])
    if 'path_name_modifier' in jdata:
        ret[path_name_modifier] = gr.update(value=jdata['path_name_modifier'])
    if 'image_path' in jdata:
        ret[image_path] = gr.update(value=jdata['image_path'])
    if 'mp4_path' in jdata:
        ret[mp4_path] = gr.update(value=jdata['mp4_path'])
    
    return ret