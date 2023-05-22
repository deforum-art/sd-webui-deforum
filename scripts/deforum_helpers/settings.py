import os
import json
import modules.shared as sh
from .args import DeforumArgs, DeforumAnimArgs, DeforumOutputArgs, ParseqArgs, LoopArgs, get_settings_component_names, pack_args
from .deforum_controlnet import controlnet_component_names
from .defaults import mask_fill_choices
from .deprecation_utils import handle_deprecated_settings
from .general_utils import get_deforum_version, clean_gradio_path_strings

def get_keys_to_exclude():
    return ["init_sample", "perlin_w", "perlin_h", "image_path", "outdir"]
    # perlin params are used just not shown in ui for now, so not to be deleted
    # image_path and outdir are in use, not to be deleted

def load_args(args_dict_main, args, anim_args, parseq_args, loop_args, controlnet_args, video_args, custom_settings_file, root, run_id):
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
            args_dict_main['animation_prompts_positive'] = jdata["animation_prompts_positive"]
        if "animation_prompts_negative" in jdata:
            args_dict_main['animation_prompts_negative'] = jdata["animation_prompts_negative"]
        keys_to_exclude = get_keys_to_exclude()
        for args_namespace in [args, anim_args, parseq_args, loop_args, controlnet_args, video_args]:
            for k, v in vars(args_namespace).items():
                if k not in keys_to_exclude:
                    if k in jdata:
                        setattr(args_namespace, k, jdata[k])
                    else:
                        print(f"Key {k} doesn't exist in the custom settings data! Using default value of {v}")
        print(args, anim_args, parseq_args, loop_args)
        return True

# save settings function that get calls when run_deforum is being called
def save_settings_from_animation_run(args, anim_args, parseq_args, loop_args, controlnet_args, video_args, root, full_out_file_path = None):
    if full_out_file_path:
        args.__dict__["seed"] = root.raw_seed
        args.__dict__["batch_name"] = root.raw_batch_name
    args.__dict__["prompts"] = root.animation_prompts
    args.__dict__["positive_prompts"] = args.positive_prompts
    args.__dict__["negative_prompts"] = args.negative_prompts
    exclude_keys = get_keys_to_exclude()
    settings_filename = full_out_file_path if full_out_file_path else os.path.join(args.outdir, f"{root.timestring}_settings.txt")
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
    settings_path = args[0].strip()
    settings_path = clean_gradio_path_strings(settings_path)
    settings_component_names = get_settings_component_names()
    data = {settings_component_names[i]: args[i+1] for i in range(0, len(settings_component_names))}
    args_dict = pack_args(data, DeforumArgs)
    anim_args_dict = pack_args(data, DeforumAnimArgs)
    parseq_dict = pack_args(data, ParseqArgs)
    args_dict["prompts"] = json.loads(data['animation_prompts'])
    args_dict["animation_prompts_positive"] = data['animation_prompts_positive']
    args_dict["animation_prompts_negative"] = data['animation_prompts_negative']
    loop_dict = pack_args(data, LoopArgs)
    controlnet_dict = pack_args(data, controlnet_component_names)
    video_args_dict = pack_args(data, DeforumOutputArgs)
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
    settings_path = clean_gradio_path_strings(settings_path)
    settings_component_names = get_settings_component_names()
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
            print(f"{key} not found in load file, using default value: {default_key_val}")
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
    vid_args_names = list(DeforumOutputArgs().keys())
    data = {vid_args_names[i]: args[i+1] for i in range(0, len(vid_args_names))}
    print(f"reading custom video settings from {video_settings_path}")
    jdata = {}
    if not os.path.isfile(video_settings_path):
        print('The custom video settings file does not exist. The values will be unchanged.')
        return [data[name] for name in vid_args_names] + [""]
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
    
    return ret