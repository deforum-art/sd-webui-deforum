from math import ceil
import os
import json
import deforum_helpers.args as deforum_args
from .args import mask_fill_choices, DeforumArgs, DeforumAnimArgs
from .deprecation_utils import handle_deprecated_settings
from .general_utils import get_deforum_version
from modules.shared import opts
import modules.shared as sh
import logging

DEBUG_MODE = opts.data.get("deforum_debug_mode_enabled", False)

def get_keys_to_exclude():
    return ["n_batch", "seed_enable_extras", "save_samples", "display_samples", "show_sample_per_step", "filename_format", "from_img2img_instead_of_link", "scale", "subseed", "subseed_strength", "C", "f", "init_latent", "init_sample", "init_c", "noise_mask", "seed_internal", "perlin_w", "perlin_h", "mp4_path", "image_path", "output_format","render_steps","path_name_modifier", 'cn_1_input_video_chosen_file', 'cn_1_input_video_mask_chosen_file', 'cn_2_input_video_chosen_file', 'cn_2_input_video_mask_chosen_file', 'cn_3_input_video_chosen_file', 'cn_3_input_video_mask_chosen_file','cn_4_input_video_chosen_file', 'cn_4_input_video_mask_chosen_file']
       
def load_args(args_dict_main, args_dict, anim_args_dict, parseq_args_dict, loop_args_dict, controlnet_args_dict, video_args_dict, custom_settings_file, root, run_id):
    custom_settings_file = custom_settings_file[run_id]
    print(f"reading custom settings from {custom_settings_file.name}")
    if not os.path.isfile(custom_settings_file.name):
        print('Custom settings file does not exist. Using in-notebook settings.')
        return
    with open(custom_settings_file.name, "r") as f:
        try:
            jdata = json.loads(f.read())
        except:
            return False
        handle_deprecated_settings(jdata)
        root.animation_prompts = jdata.get("prompts", root.animation_prompts)
        if "animation_prompts_positive" in jdata:
            args_dict_main['animation_prompts_positive'] = jdata["animation_prompts_positive"] # Update the args_dict_main
        if "animation_prompts_negative" in jdata:
            args_dict_main['animation_prompts_negative'] = jdata["animation_prompts_negative"] # Update the args_dict_main
        keys_to_exclude = get_keys_to_exclude()
        for dicts in [args_dict, anim_args_dict, parseq_args_dict, loop_args_dict, video_args_dict]:
            for k, v in dicts.items():
                # Check if the key is not in the keys_to_exclude list before processing
                if k not in keys_to_exclude:
                    if k in jdata:
                        dicts[k] = jdata[k]
                    else:
                        print(f"Key {k} doesn't exist in the custom settings data! Using default value of {v}")
        print(args_dict, anim_args_dict, parseq_args_dict, loop_args_dict)
        return True

def save_settings_from_animation_run(args, anim_args, parseq_args, loop_args, controlnet_args, video_args, root, full_out_file_path = None):
    if full_out_file_path:
        args.__dict__["seed"] = root.raw_seed
        args.__dict__["batch_name"] = root.raw_batch_name
    args.__dict__["prompts"] = root.animation_prompts
    args.__dict__["positive_prompts"] = root.positive_prompts
    args.__dict__["negative_prompts"] = root.negative_prompts
    exclude_keys = get_keys_to_exclude()
    settings_filename = full_out_file_path if full_out_file_path else os.path.join(args.outdir, f"{args.timestring}_settings.txt")
    with open(settings_filename, "w+", encoding="utf-8") as f:
        s = {}
        for d in (args.__dict__, anim_args.__dict__, parseq_args.__dict__, loop_args.__dict__, controlnet_args.__dict__, video_args.__dict__):
            s.update({k: v for k, v in d.items() if k not in exclude_keys})
        s["sd_model_name"] = sh.sd_model.sd_checkpoint_info.name
        s["sd_model_hash"] = sh.sd_model.sd_checkpoint_info.hash
        s["deforum_git_commit_id"] = get_deforum_version()
        json.dump(s, f, ensure_ascii=False, indent=4)

# In gradio gui settings save/ load funcs:
def save_settings(*args, **kwargs):
    from deforum_helpers.args import pack_args, pack_anim_args, pack_parseq_args, pack_loop_args, pack_controlnet_args, pack_video_args
    settings_path = args[0].strip()
    settings_component_names = deforum_args.get_settings_component_names()
    data = {settings_component_names[i]: args[i+1] for i in range(0, len(settings_component_names))}
    args_dict = pack_args(data)
    anim_args_dict = pack_anim_args(data)
    parseq_dict = pack_parseq_args(data)
    args_dict["prompts"] = json.loads(data['animation_prompts'])
    args_dict["animation_prompts_positive"] = data['animation_prompts_positive']
    args_dict["animation_prompts_negative"] = data['animation_prompts_negative']
    loop_dict = pack_loop_args(data)
    controlnet_dict = pack_controlnet_args(data)
    video_args_dict = pack_video_args(data)
    combined = {**args_dict, **anim_args_dict, **parseq_dict, **loop_dict, **controlnet_dict, **video_args_dict}
    exclude_keys = get_keys_to_exclude()
    filtered_combined = {k: v for k, v in combined.items() if k not in exclude_keys}
    filtered_combined["sd_model_name"] = sh.sd_model.sd_checkpoint_info.name
    filtered_combined["sd_model_hash"] = sh.sd_model.sd_checkpoint_info.hash
    filtered_combined["deforum_git_commit_id"] = get_deforum_version()
    print(f"saving custom settings to {settings_path}")
    with open(settings_path, "w", encoding='utf-8') as f:
        f.write(json.dumps(filtered_combined, ensure_ascii=False, indent=4))
    
    return [""]

def load_all_settings(*args, ui_launch=False, **kwargs):
    import gradio as gr
    settings_path = args[0].strip()
    settings_component_names = deforum_args.get_settings_component_names()
    data = {settings_component_names[i]: args[i+1] for i in range(len(settings_component_names))}
    print(f"reading custom settings from {settings_path}")

    if not os.path.isfile(settings_path):
        print('The custom settings file does not exist. The values will be unchanged.')
        if ui_launch:
            return ({key: gr.update(value=value) for key, value in data.items()},)
        else:
            return list(data.values()) + [""]

    with open(settings_path, "r", encoding='utf-8') as f:
        jdata = json.load(f)
        handle_deprecated_settings(jdata)
        if 'animation_prompts' in jdata:
            jdata['prompts'] = jdata['animation_prompts']

    result = {}
    for key, default_val in data.items():
        val = jdata.get(key, default_val)
        if key == 'sampler' and isinstance(val, int):
            from modules.sd_samplers import samplers_for_img2img
            val = samplers_for_img2img[val].name
        elif key == 'fill' and isinstance(val, int):
            val = mask_fill_choices[val]
        elif key in {'reroll_blank_frames', 'noise_type'} and key not in jdata:
            default_key_val = (DeforumArgs if key != 'noise_type' else DeforumAnimArgs)[key]
            logging.debug(f"{key} not found in load file, using default value: {default_key_val}")
            val = default_key_val
        elif key in {'animation_prompts_positive', 'animation_prompts_negative'}:
            val = jdata.get(key, default_val)
        elif key == 'animation_prompts':
            val = json.dumps(jdata['prompts'], ensure_ascii=False, indent=4)

        result[key] = val

    if ui_launch:
        return ({key: gr.update(value=value) for key, value in result.items()},)
    else:
        return list(result.values()) + [""]


def load_video_settings(*args, **kwargs):
    video_settings_path = args[0].strip()
    data = {deforum_args.video_args_names[i]: args[i+1] for i in range(0, len(deforum_args.video_args_names))}
    print(f"reading custom video settings from {video_settings_path}")
    jdata = {}
    if not os.path.isfile(video_settings_path):
        print('The custom video settings file does not exist. The values will be unchanged.')
        return [data[name] for name in deforum_args.video_args_names] + [""]
    else:
        with open(video_settings_path, "r") as f:
            jdata = json.loads(f.read())
            handle_deprecated_settings(jdata)
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
    def __init__(self, args, anim_args, parseq_args, video_args):
        self._tqdm = None
        self._args = args
        self._anim_args = anim_args
        self._parseq_args = parseq_args
        self._video_args = video_args

    def reset(self):
        from .animation_key_frames import DeformAnimKeys
        from .parseq_adapter import ParseqAnimKeys
        deforum_total = 0
        # FIXME: get only amount of steps
        use_parseq = self._parseq_args.parseq_manifest != None and self._parseq_args.parseq_manifest.strip()
        keys = DeformAnimKeys(self._anim_args) if not use_parseq else ParseqAnimKeys(self._parseq_args, self._anim_args, self._video_args, mute=True)
        
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
