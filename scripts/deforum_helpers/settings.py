from math import ceil
import os
import json
from .args import mask_fill_choices, DeforumArgs, DeforumAnimArgs
import logging

def load_args(args_dict,anim_args_dict, parseq_args_dict, custom_settings_file, root):
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
            for i, k in enumerate(parseq_args_dict):
                if k in jdata:
                    parseq_args_dict[k] = jdata[k]
                else:
                    print(f"key {k} doesn't exist in the custom settings data! using the default value of {parseq_args_dict[k]}")                    
            print(args_dict)
            print(anim_args_dict)
            print(parseq_args_dict)

import gradio as gr
 
# In gradio gui settings save/load
def save_settings(settings_path, override_settings_with_file, custom_settings_file, animation_mode, max_frames, border, angle, zoom, translation_x, translation_y, translation_z, rotation_3d_x, rotation_3d_y, rotation_3d_z, flip_2d_perspective, perspective_flip_theta, perspective_flip_phi, perspective_flip_gamma, perspective_flip_fv, noise_schedule, strength_schedule, contrast_schedule, cfg_scale_schedule, enable_steps_scheduling, steps_schedule, fov_schedule, near_schedule, far_schedule, seed_schedule, enable_sampler_scheduling, sampler_schedule, mask_schedule, use_noise_mask, noise_mask_schedule, kernel_schedule, sigma_schedule, amount_schedule, threshold_schedule, histogram_matching, color_coherence, color_coherence_video_every_N_frames, diffusion_cadence, noise_type, perlin_w, perlin_h, perlin_octaves, perlin_persistence, use_depth_warping, midas_weight, near_plane, far_plane, fov, padding_mode, sampling_mode, save_depth_maps, video_init_path, extract_nth_frame, extract_from_frame, extract_to_frame, overwrite_extracted_frames, use_mask_video, video_mask_path, interpolate_key_frames, interpolate_x_frames, resume_from_timestring, resume_timestring, prompts, animation_prompts, W, H, restore_faces, tiling, enable_hr, firstphase_width, firstphase_height, seed, sampler, seed_enable_extras, subseed, subseed_strength, seed_resize_from_w, seed_resize_from_h, steps, ddim_eta, n_batch, make_grid, grid_rows, save_settings, save_samples, display_samples, save_sample_per_step, show_sample_per_step, override_these_with_webui, batch_name, filename_format, seed_behavior, seed_iter_N, use_init, from_img2img_instead_of_link, strength_0_no_init, strength, init_image, use_mask, use_alpha_as_mask, invert_mask, overlay_mask, mask_file, mask_contrast_adjust, mask_brightness_adjust, mask_overlay_blur, fill, full_res_mask, full_res_mask_padding, reroll_blank_frames, parseq_manifest, parseq_use_deltas, hybrid_generate_inputframes, hybrid_generate_human_masks, hybrid_use_first_frame_as_init_image, hybrid_motion, hybrid_flow_method, hybrid_composite, hybrid_comp_mask_type, hybrid_comp_mask_inverse, hybrid_comp_mask_equalize, hybrid_comp_mask_auto_contrast, hybrid_comp_save_extra_frames, hybrid_comp_alpha_schedule, hybrid_comp_mask_blend_alpha_schedule, hybrid_comp_mask_contrast_schedule, hybrid_comp_mask_auto_contrast_cutoff_high_schedule, hybrid_comp_mask_auto_contrast_cutoff_low_schedule):
    from deforum_helpers.args import pack_args, pack_anim_args, pack_parseq_args
    args_dict = pack_args(W, H, restore_faces, tiling, enable_hr, firstphase_width, firstphase_height, seed, sampler, seed_enable_extras, subseed, subseed_strength, seed_resize_from_w, seed_resize_from_h, steps, ddim_eta, n_batch, make_grid, grid_rows, save_settings, save_samples, display_samples, save_sample_per_step, show_sample_per_step, override_these_with_webui, batch_name, filename_format, seed_behavior, seed_iter_N, use_init, from_img2img_instead_of_link, strength_0_no_init, strength, init_image, use_mask, use_alpha_as_mask, invert_mask, overlay_mask, mask_file, mask_contrast_adjust, mask_brightness_adjust, mask_overlay_blur, fill, full_res_mask, full_res_mask_padding, reroll_blank_frames)
    anim_args_dict = pack_anim_args(animation_mode, max_frames, border, angle, zoom, translation_x, translation_y, translation_z, rotation_3d_x, rotation_3d_y, rotation_3d_z, flip_2d_perspective, perspective_flip_theta, perspective_flip_phi, perspective_flip_gamma, perspective_flip_fv, noise_schedule, strength_schedule, contrast_schedule, cfg_scale_schedule, enable_steps_scheduling, steps_schedule, fov_schedule, near_schedule, far_schedule, seed_schedule, enable_sampler_scheduling, sampler_schedule, mask_schedule, use_noise_mask, noise_mask_schedule, kernel_schedule, sigma_schedule, amount_schedule, threshold_schedule, histogram_matching, color_coherence, color_coherence_video_every_N_frames, diffusion_cadence, noise_type, perlin_w, perlin_h, perlin_octaves, perlin_persistence, use_depth_warping, midas_weight, near_plane, far_plane, fov, padding_mode, sampling_mode, save_depth_maps, video_init_path, extract_nth_frame, extract_from_frame, extract_to_frame, overwrite_extracted_frames, use_mask_video, video_mask_path, interpolate_key_frames, interpolate_x_frames, resume_from_timestring, resume_timestring, hybrid_generate_inputframes, hybrid_generate_human_masks, hybrid_use_first_frame_as_init_image, hybrid_motion, hybrid_flow_method, hybrid_composite, hybrid_comp_mask_type, hybrid_comp_mask_inverse, hybrid_comp_mask_equalize, hybrid_comp_mask_auto_contrast, hybrid_comp_save_extra_frames, hybrid_comp_alpha_schedule, hybrid_comp_mask_blend_alpha_schedule, hybrid_comp_mask_contrast_schedule, hybrid_comp_mask_auto_contrast_cutoff_high_schedule, hybrid_comp_mask_auto_contrast_cutoff_low_schedule)
    parseq_dict = pack_parseq_args(parseq_manifest, parseq_use_deltas)    
    args_dict["prompts"] = json.loads(animation_prompts)
    print(f"saving custom settings to {settings_path}")
    with open(settings_path, "w") as f:
        f.write(json.dumps({**args_dict, **anim_args_dict, **parseq_dict}, ensure_ascii=False, indent=4))
    return [""]

def save_video_settings(video_settings_path, skip_video_for_run_all, fps, output_format, ffmpeg_location, add_soundtrack, soundtrack_path, use_manual_settings, render_steps, max_video_frames, path_name_modifier, image_path, mp4_path, store_frames_in_ram):
    from deforum_helpers.args import pack_video_args
    video_args_dict = pack_video_args(skip_video_for_run_all, fps, output_format, ffmpeg_location, add_soundtrack, soundtrack_path, use_manual_settings, render_steps, max_video_frames, path_name_modifier, image_path, mp4_path, store_frames_in_ram)
    print(f"saving video settings to {video_settings_path}")
    with open(video_settings_path, "w") as f:
        f.write(json.dumps(video_args_dict, ensure_ascii=False, indent=4))
    return [""]

def load_settings(settings_path, override_settings_with_file, custom_settings_file, animation_mode, max_frames, border, angle, zoom, translation_x, translation_y, translation_z, rotation_3d_x, rotation_3d_y, rotation_3d_z, flip_2d_perspective, perspective_flip_theta, perspective_flip_phi, perspective_flip_gamma, perspective_flip_fv, noise_schedule, strength_schedule, contrast_schedule, cfg_scale_schedule, enable_steps_scheduling, steps_schedule, fov_schedule, near_schedule, far_schedule, seed_schedule, enable_sampler_scheduling, sampler_schedule, mask_schedule, use_noise_mask, noise_mask_schedule, kernel_schedule, sigma_schedule, amount_schedule, threshold_schedule, histogram_matching, color_coherence, color_coherence_video_every_N_frames, diffusion_cadence, noise_type, perlin_w, perlin_h, perlin_octaves, perlin_persistence, use_depth_warping, midas_weight, near_plane, far_plane, fov, padding_mode, sampling_mode, save_depth_maps, video_init_path, extract_nth_frame, extract_from_frame, extract_to_frame, overwrite_extracted_frames, use_mask_video, video_mask_path, interpolate_key_frames, interpolate_x_frames, resume_from_timestring, resume_timestring, prompts, animation_prompts, W, H, restore_faces, tiling, enable_hr, firstphase_width, firstphase_height, seed, sampler, seed_enable_extras, subseed, subseed_strength, seed_resize_from_w, seed_resize_from_h, steps, ddim_eta, n_batch, make_grid, grid_rows, save_settings, save_samples, display_samples, save_sample_per_step, show_sample_per_step, override_these_with_webui, batch_name, filename_format, seed_behavior, seed_iter_N, use_init, from_img2img_instead_of_link, strength_0_no_init, strength, init_image, use_mask, use_alpha_as_mask, invert_mask, overlay_mask, mask_file, mask_contrast_adjust, mask_brightness_adjust, mask_overlay_blur, fill, full_res_mask, full_res_mask_padding, reroll_blank_frames, parseq_manifest, parseq_use_deltas, hybrid_generate_inputframes, hybrid_generate_human_masks, hybrid_use_first_frame_as_init_image, hybrid_motion, hybrid_flow_method, hybrid_composite, hybrid_comp_mask_type, hybrid_comp_mask_inverse, hybrid_comp_mask_equalize, hybrid_comp_mask_auto_contrast, hybrid_comp_save_extra_frames, hybrid_comp_alpha_schedule, hybrid_comp_mask_blend_alpha_schedule, hybrid_comp_mask_contrast_schedule, hybrid_comp_mask_auto_contrast_cutoff_high_schedule, hybrid_comp_mask_auto_contrast_cutoff_low_schedule):
    print(f"reading custom settings from {settings_path}")
    data = locals()
    data.pop("settings_path")
    jdata = {}
    if not os.path.isfile(settings_path):
        print('The custom settings file does not exist. The values will be unchanged.')
        return [override_settings_with_file, custom_settings_file, animation_mode, max_frames, border, angle, zoom, translation_x, translation_y, translation_z, rotation_3d_x, rotation_3d_y, rotation_3d_z, flip_2d_perspective, perspective_flip_theta, perspective_flip_phi, perspective_flip_gamma, perspective_flip_fv, noise_schedule, strength_schedule, contrast_schedule, cfg_scale_schedule, enable_steps_scheduling, steps_schedule, fov_schedule, near_schedule, far_schedule, seed_schedule, enable_sampler_scheduling, sampler_schedule, mask_schedule, use_noise_mask, noise_mask_schedule, kernel_schedule, sigma_schedule, amount_schedule, threshold_schedule, histogram_matching, color_coherence, color_coherence_video_every_N_frames, diffusion_cadence, noise_type, perlin_w, perlin_h, perlin_octaves, perlin_persistence, use_depth_warping, midas_weight, near_plane, far_plane, fov, padding_mode, sampling_mode, save_depth_maps, video_init_path, extract_nth_frame, extract_from_frame, extract_to_frame, overwrite_extracted_frames, use_mask_video, video_mask_path, interpolate_key_frames, interpolate_x_frames, resume_from_timestring, resume_timestring, prompts, animation_prompts, W, H, restore_faces, tiling, enable_hr, firstphase_width, firstphase_height, seed, sampler, seed_enable_extras, subseed, subseed_strength, seed_resize_from_w, seed_resize_from_h, steps, ddim_eta, n_batch, make_grid, grid_rows, save_settings, save_samples, display_samples, save_sample_per_step, show_sample_per_step, override_these_with_webui, batch_name, filename_format, seed_behavior, seed_iter_N, use_init, from_img2img_instead_of_link, strength_0_no_init, strength, init_image, use_mask, use_alpha_as_mask, invert_mask, overlay_mask, mask_file, mask_contrast_adjust, mask_brightness_adjust, mask_overlay_blur, fill, full_res_mask, full_res_mask_padding, reroll_blank_frames, parseq_manifest, parseq_use_deltas, hybrid_generate_inputframes, hybrid_generate_human_masks, hybrid_use_first_frame_as_init_image, hybrid_motion, hybrid_flow_method, hybrid_composite, hybrid_comp_mask_type, hybrid_comp_mask_inverse, hybrid_comp_mask_equalize, hybrid_comp_mask_auto_contrast, hybrid_comp_save_extra_frames, hybrid_comp_alpha_schedule, hybrid_comp_mask_blend_alpha_schedule, hybrid_comp_mask_contrast_schedule, hybrid_comp_mask_auto_contrast_cutoff_high_schedule, hybrid_comp_mask_auto_contrast_cutoff_low_schedule, ""]
    else:
        with open(settings_path, "r") as f:
            jdata = json.loads(f.read())
    ret = []

    if 'animation_prompts' in jdata:
        jdata['prompts'] = jdata['animation_prompts']#compatibility with old versions

    for key in data:
        if key == 'sampler':
            sampler_val = jdata[key]
            if type(sampler_val) == int:
                from modules.sd_samplers import samplers_for_img2img
                ret.append(samplers_for_img2img[sampler_val].name)
            else:
                ret.append(sampler_val)
        
        elif key == 'fill':
            if key in jdata:
                fill_val = jdata[key]
                if type(fill_val) == int:                    
                    ret.append(mask_fill_choices[fill_val])
                else:
                    ret.append(fill_val)
            else:
                fill_default = DeforumArgs()['fill']
                logging.debug(f"Fill not found in load file, using default value: {fill_default}")
                ret.append(mask_fill_choices[fill_default])
        
        elif key == 'reroll_blank_frames':
            if key in jdata:
                reroll_blank_frames_val = jdata[key]
                ret.append(reroll_blank_frames_val)
            else:
                reroll_blank_frames_default = DeforumArgs()['reroll_blank_frames']
                logging.debug(f"Reroll blank frames not found in load file, using default value: {reroll_blank_frames_default}")
                ret.append(reroll_blank_frames_default)
        
        elif key == 'noise_type':
            if key in jdata:
                noise_type_val = jdata[key]
                ret.append(noise_type_val)
            else:
                noise_type_default = DeforumAnimArgs()['noise_type']
                logging.debug(f"Noise type not found in load file, using default value: {noise_type_default}")
                ret.append(noise_type_default)
            
        elif key in jdata:
            ret.append(jdata[key])
        else:
            if key == 'animation_prompts':
                ret.append(json.dumps(jdata['prompts'], ensure_ascii=False, indent=4))
            else:
                ret.append(data[key])

    #stuff
    ret.append("")

    return ret

def load_video_settings(video_settings_path, skip_video_for_run_all, fps, output_format, ffmpeg_location, add_soundtrack, soundtrack_path, use_manual_settings, render_steps, max_video_frames, path_name_modifier, image_path, mp4_path, store_frames_in_ram):  
    print(f"reading custom video settings from {video_settings_path}")
    data = locals()
    data.pop("video_settings_path")
    jdata = {}
    if not os.path.isfile(video_settings_path):
        print('The custom video settings file does not exist. The values will be unchanged.')
        return [skip_video_for_run_all, fps, output_format, ffmpeg_location, add_soundtrack, soundtrack_path, use_manual_settings, render_steps, max_video_frames, path_name_modifier, image_path, mp4_path, store_frames_in_ram, ""]
    else:
        with open(video_settings_path, "r") as f:
            jdata = json.loads(f.read())
    ret = []

    for key in data:
        if key == 'add_soundtrack':
            add_soundtrack_val = jdata[key]
            if type(add_soundtrack_val) == bool:
                ret.append('File' if add_soundtrack_val else 'None')
            else:
                ret.append(add_soundtrack_val)
        elif key in jdata:
            ret.append(jdata[key])
        else:
            ret.append(data[key])
    
    #stuff
    ret.append("")
    
    return ret

import tqdm
from modules.shared import state, progress_print_out, opts, cmd_opts
class DeforumTQDM:
    def __init__(self, args, anim_args, parseq_args):
        self._tqdm = None
        self._args = args
        self._anim_args = anim_args
        self._parseq_args = parseq_args

    def reset(self):
        from .animation_key_frames import DeformAnimKeys
        from .parseq_adapter import ParseqAnimKeys
        deforum_total = 0
        # FIXME: get only amount of steps
        use_parseq = self._parseq_args.parseq_manifest != None and self._parseq_args.parseq_manifest.strip()
        keys = DeformAnimKeys(self._anim_args) if not use_parseq else ParseqAnimKeys(self._parseq_args, self._anim_args)        
        
        start_frame = 0
        if self._anim_args.resume_from_timestring:
            for tmp in os.listdir(self._args.outdir):
                filename = tmp.split("_")
                # don't use saved depth maps to count number of frames
                if self._anim_args.resume_timestring in filename and "depth" not in filename:
                    start_frame += 1
            start_frame = start_frame - 1
        using_vid_init = self._anim_args.animation_mode == 'Video Input'
        turbo_steps = 1 if using_vid_init else int(self._anim_args.diffusion_cadence)
        if self._anim_args.resume_from_timestring:
            last_frame = start_frame-1
            if turbo_steps > 1:
                last_frame -= last_frame%turbo_steps
            if turbo_steps > 1:
                turbo_next_frame_idx = last_frame
                turbo_prev_frame_idx = turbo_next_frame_idx
                start_frame = last_frame+turbo_steps
        frame_idx = start_frame
        had_first = False
        while frame_idx < self._anim_args.max_frames:
            strength = keys.strength_schedule_series[frame_idx]
            #sample, image = generate(args, root, frame_idx, return_sample=True)
            if not had_first and self._args.use_init and self._args.init_image != None and self._args.init_image != '':
                deforum_total += int(ceil(self._args.steps * (1-strength)))
                had_first = True
            elif not had_first:
                deforum_total += self._args.steps
                had_first = True
            else:
                deforum_total += int(ceil(self._args.steps * (1-strength)))

            if turbo_steps > 1:
                frame_idx += turbo_steps
            else:
                frame_idx += 1
        
        self._tqdm = tqdm.tqdm(
            desc="Deforum progress",
            total=deforum_total,
            position=1,
            file=progress_print_out
        )

    def update(self):
        if not opts.multiple_tqdm or cmd_opts.disable_console_progressbars:
            return
        if self._tqdm is None:
            self.reset()
        self._tqdm.update()

    def updateTotal(self, new_total):
        if not opts.multiple_tqdm or cmd_opts.disable_console_progressbars:
            return
        if self._tqdm is None:
            self.reset()
        self._tqdm.total=new_total

    def clear(self):
        if self._tqdm is not None:
            self._tqdm.close()
            self._tqdm = None
