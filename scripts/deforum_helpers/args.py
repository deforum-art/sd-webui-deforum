import json
import os
import tempfile
import time
from types import SimpleNamespace
import modules.paths as ph
import modules.shared as sh
from modules.processing import get_fixed_seed
from .defaults import get_guided_imgs_default_json, mask_fill_choices
from .deforum_controlnet import controlnet_component_names
from .general_utils import get_os, substitute_placeholders

def RootArgs():
    return {
        "device": sh.device,
        "models_path": ph.models_path + '/Deforum',
        "half_precision": not sh.cmd_opts.no_half,
        "mask_preset_names": ['everywhere', 'video_mask'],
        "frames_cache": [],
        "raw_batch_name": None,
        "raw_seed": None,
        "initial_info": None,
        "first_frame": None,
        "animation_prompts": None,
        "current_user_os": get_os(),
        "tmp_deforum_run_duplicated_folder": os.path.join(tempfile.gettempdir(), 'tmp_run_deforum')
    }

def CoreArgs():  # TODO: change or do something with this ugliness
    return {
        "subseed": -1,
        "subseed_strength": 0,
        "timestring": "",
        "init_sample": None,
        "noise_mask": None,
        "seed_internal": 0
    }

def DeforumAnimArgs():
    return {
        "animation_mode": '2D',  # ['None', '2D', '3D', 'Video Input', 'Interpolation']
        "max_frames": 120,
        "border": 'replicate',  # ['wrap', 'replicate']
        "angle": "0: (0)",
        "zoom": "0: (1.0025+0.002*sin(1.25*3.14*t/30))",
        "translation_x": "0: (0)",
        "translation_y": "0: (0)",
        "translation_z": "0: (1.75)",
        "transform_center_x": "0: (0.5)",
        "transform_center_y": "0: (0.5)",
        "rotation_3d_x": "0:(0)",
        "rotation_3d_y": "0:(0)",
        "rotation_3d_z": "0:(0)",
        "enable_perspective_flip": False,
        "perspective_flip_theta": "0:(0)",
        "perspective_flip_phi": "0: (0)",
        "perspective_flip_gamma": "0: (0)",
        "perspective_flip_fv": "0: (53)",
        "noise_schedule": "0: (0.065)",
        "strength_schedule": "0: (0.65)",
        "contrast_schedule": "0: (1.0)",
        "cfg_scale_schedule": "0: (7)",
        "enable_steps_scheduling": {
            "label": "Enable steps scheduling",
            "type": "checkbox",
            "value": False,
            "info": ""
        },
        "steps_schedule": {
            "label": "Steps schedule",
            "type": "textbox",
            "value": "0: (25)",
            "info": "mainly allows using more than 200 steps. otherwise, it's a mirror-like param of 'strength schedule'"
        },
        "fov_schedule": "0: (70)",
        "aspect_ratio_schedule": "0: (1)",
        "aspect_ratio_use_old_formula": False,
        "near_schedule": "0: (200)",
        "far_schedule": "0: (10000)",
        "seed_schedule": {
            "label": "Seed schedule",
            "type": "textbox",
            "value": '0:(s), 1:(-1), "max_f-2":(-1), "max_f-1":(s)',
            "info": ""
        },
        "pix2pix_img_cfg_scale_schedule": {
            "label": "Pix2Pix img CFG schedule",
            "type": "textbox",
            "value": "0:(1.5)",
            "info": "ONLY in use when working with a P2P ckpt!"
        },
        "enable_subseed_scheduling": {
            "label": "Enable Subseed scheduling",
            "type": "checkbox",
            "value": False,
            "info": ""
        },
        "subseed_schedule": {
            "label": "Subseed schedule",
            "type": "textbox",
            "value": "0: (1)",
            "info": ""
        },
        "subseed_strength_schedule": {
            "label": "Subseed strength schedule",
            "type": "textbox",
            "value": "0: (0)",
            "info": ""
        },
        "enable_sampler_scheduling": False,  # Sampler Scheduling
        "sampler_schedule": '0: ("Euler a")',
        "use_noise_mask": False,  # Composable mask scheduling
        "mask_schedule": '0: ("{video_mask}")',
        "noise_mask_schedule": '0: ("{video_mask}")',
        "enable_checkpoint_scheduling": False,  # Checkpoint Scheduling
        "checkpoint_schedule": '0: ("model1.ckpt"), 100: ("model2.safetensors")',
        "enable_clipskip_scheduling": False,  # CLIP skip Scheduling
        "clipskip_schedule": '0: (2)',
        "enable_noise_multiplier_scheduling": True,  # Noise Multiplier Scheduling
        "noise_multiplier_schedule": '0: (1.05)',
        "resume_from_timestring": {
            "label": "Resume from timestring",
            "type": "checkbox",
            "value": False,
            "info": ""
        },
        "resume_timestring": {
            "label": "Resume timestring",
            "type": "textbox",
            "value": "20230129210106",
            "info": ""
        },
        "enable_ddim_eta_scheduling": {
            "label": "Enable DDIM ETA scheduling",
            "type": "checkbox",
            "value": False,
            "visible": False,
            "info": ""
        },
        "ddim_eta_schedule": {
            "label": "DDIM ETA Schedule",
            "type": "textbox",
            "value": "0: (0)",
            "visible": False,
            "info": ""
        },
        "enable_ancestral_eta_scheduling": {
            "label": "Enable Ancestral ETA scheduling",
            "type": "checkbox",
            "value": False,
            "info": ""
        },
        "ancestral_eta_schedule": {
            "label": "Ancestral ETA Schedule",
            "type": "textbox",
            "value": "0: (1)",
            "visible": False,
            "info": ""
        },
        # Anti-blur
        "amount_schedule": "0: (0.1)",
        "kernel_schedule": "0: (5)",
        "sigma_schedule": "0: (1.0)",
        "threshold_schedule": "0: (0.0)",
        # Coherence
        "color_coherence": 'LAB',  # ['None', 'HSV', 'LAB', 'RGB', 'Video Input', 'Image']
        "color_coherence_image_path": "",
        "color_coherence_video_every_N_frames": 1,
        "color_force_grayscale": False,
        "legacy_colormatch": False,
        "diffusion_cadence": '2',  # ['1','2','3','4','5','6','7','8']
        "optical_flow_cadence": 'None',  # ['None', 'RAFT','DIS Medium', 'DIS Fine', 'Farneback']
        "cadence_flow_factor_schedule": "0: (1)",
        "optical_flow_redo_generation": 'None',  # ['None', 'RAFT', 'DIS Medium', 'DIS Fine', 'Farneback']
        "redo_flow_factor_schedule": "0: (1)",
        "diffusion_redo": '0',
        # **Noise settings:**
        "noise_type": 'perlin',  # ['uniform', 'perlin']
        # Perlin params
        "perlin_w": 8,
        "perlin_h": 8,
        "perlin_octaves": 4,
        "perlin_persistence": 0.5,
        # **3D Depth Warping:**
        "use_depth_warping": True,
        "depth_algorithm": 'Midas-3-Hybrid',
        # ['Midas+AdaBins (old)','Zoe+AdaBins (old)', 'Midas-3-Hybrid','Midas-3.1-BeitLarge', 'AdaBins', 'Zoe', 'Leres'] Midas-3.1-BeitLarge is temporarily removed 04-05-23 until fixed
        "midas_weight": 0.2,  # midas/ zoe weight - only relevant in old/ legacy depth_algorithm modes. see above ^
        "padding_mode": 'border',  # ['border', 'reflection', 'zeros']
        "sampling_mode": 'bicubic',  # ['bicubic', 'bilinear', 'nearest']
        "save_depth_maps": {
                "label": "Save 3D depth maps",
                "type": "checkbox",
                "value": False,
                "info": "save animation's depth maps as extra files"
            },
        # **Video Input:**
        "video_init_path": {
                "label": "Video init path/ URL",
                "type": "textbox",
                "value": 'https://deforum.github.io/a1/V1.mp4',
                "info": ""
            },
        "extract_nth_frame": {
                "label": "Extract nth frame",
                "type": "number",
                "precision": 0,
                "value": 1,
                "info": ""
            },
        "extract_from_frame": {
                "label": "Extract from frame",
                "type": "number",
                "precision": 0,
                "value": 0,
                "info": ""
        },
        "extract_to_frame": {
                "label": "Extract to frame",
                "type": "number",
                "precision": 0,
                "value": -1,
                "info": ""
        },
        "overwrite_extracted_frames": {
                "label": "Overwrite extracted frames",
                "type": "checkbox",
                "value": False,
                "info": ""
            },
        "use_mask_video": {
                "label": "Use mask video",
                "type": "checkbox",
                "value": False,
                "info": ""
            },
        "video_mask_path": {
                "label": "Video mask path",
                "type": "textbox",
                "value": 'https://deforum.github.io/a1/VM1.mp4',
                "info": ""
            },
        # **Hybrid Video for 2D/3D Animation Mode:**
        "hybrid_comp_alpha_schedule": "0:(0.5)",
        "hybrid_comp_mask_blend_alpha_schedule": "0:(0.5)",
        "hybrid_comp_mask_contrast_schedule": "0:(1)",
        "hybrid_comp_mask_auto_contrast_cutoff_high_schedule": "0:(100)",
        "hybrid_comp_mask_auto_contrast_cutoff_low_schedule": "0:(0)",
        "hybrid_flow_factor_schedule": "0:(1)",
        "hybrid_generate_inputframes": False,
        "hybrid_generate_human_masks": "None",  # ['None','PNGs','Video', 'Both']
        "hybrid_use_first_frame_as_init_image": True,
        "hybrid_motion": "None",  # ['None','Optical Flow','Perspective','Affine']
        "hybrid_motion_use_prev_img": False,
        "hybrid_flow_consistency": False,
        "hybrid_consistency_blur": 2,
        "hybrid_flow_method": "RAFT",  # ['RAFT', 'DIS Medium', 'DIS Fine', 'Farneback']
        "hybrid_composite": 'None',  # ['None', 'Normal', 'Before Motion', 'After Generation']
        "hybrid_use_init_image": False,
        "hybrid_comp_mask_type": "None",  # ['None', 'Depth', 'Video Depth', 'Blend', 'Difference']
        "hybrid_comp_mask_inverse": False,
        "hybrid_comp_mask_equalize": "None",  # ['None','Before','After','Both']
        "hybrid_comp_mask_auto_contrast": False,
        "hybrid_comp_save_extra_frames": False
    }

def DeforumArgs():
    from modules.sd_samplers import samplers_for_img2img
    return {
            "W": {
                "label": "Width",
                "type": "slider",
                "min": 8,
                "max": 2048,
                "step": 8,
                "value": 512,
            },
            "H": {
                "label": "Height",
                "type": "slider",
                "min": 64,
                "max": 2048,
                "step": 64,
                "value": 512,
            },
            "show_info_on_ui": True,
            "tiling": {
                "label": "Tiling",
                "type": "checkbox",
                "value": False,
                "info": "Enable for seamless-tiling of each generated image. Experimental"
            },
            "restore_faces": {
                "label": "Restore faces",
                "type": "checkbox",
                "value": False,
                "info": "enable to trigger webui's face restoration on each frame during the generation"
            },
            "seed_enable_extras": {
                "label": "Enable subseed controls",
                "type": "checkbox",
                "visible": False,
                "value": False,
                "info": ""
            },
            "seed_resize_from_w": {
                "label": "Resize seed from width",
                "type": "slider",
                "min": 0,
                "max": 2048,
                "step": 64,
                "value": 0,
            },
            "seed_resize_from_h": {
                "label": "Resize seed from height",
                "type": "slider",
                "min": 0,
                "max": 2048,
                "step": 64,
                "value": 0,
            },
            "seed": {
                "label": "Seed",
                "type": "number",
                "precision": 0,
                "value": -1,
                "info": "Starting seed for the animation. -1 for random"
            },
            "sampler": {
                "label": "Sampler",
                "type": "dropdown",
                "choices": [x.name for x in samplers_for_img2img],
                "value": samplers_for_img2img[0].name,
            },
            "steps": {
                "label": "step",
                "type": "slider",
                "min": 1,
                "max": 200,
                "step": 1,
                "value": 25,
            },
            "batch_name": {
                "label": "Batch name",
                "type": "textbox",
                "value": "Deforum_{timestring}",
                "info": "output images will be placed in a folder with this name ({timestring} token will be replaced) inside the img2img output folder. Supports params placeholders. e.g {seed}, {w}, {h}, {prompts}"
            },
            "seed_behavior": {
                "label": "Seed behavior",
                "type": "radio",
                "choices": ['iter', 'fixed', 'random', 'ladder', 'alternate', 'schedule'],
                "value": "iter",
                "info": "controls the seed behavior that is used for animation. hover on the options to see more info"
            },
            "seed_iter_N": {
                "label": "Seed iter N",
                "type": "number",
                "precision": 0,
                "value": 1,
                "info": "for how many frames the same seed should stick before iterating to the next one"
            },
            "use_init": {
                "label": "Use init",
                "type": "checkbox",
                "value": False,
                "info": ""
            },
            "strength": {
                "label": "strength",
                "type": "slider",
                "min": 0,
                "max": 1,
                "step": 0.01,
                "value": 0.8,
            },
            "strength_0_no_init": {
                "label": "Strength 0 no init",
                "type": "checkbox",
                "value": True,
                "info": ""
            },
            "init_image": {
                "label": "Init image",
                "type": "textbox",
                "value": "https://deforum.github.io/a1/I1.png",
                "info": ""
            },
            "use_mask": {
                "label": "Use mask",
                "type": "checkbox",
                "value": False,
                "info": ""
            },
            "use_alpha_as_mask": {
                "label": "Use alpha as mask",
                "type": "checkbox",
                "value": False,
                "info": ""
            },
            "mask_file": {
                "label": "Mask file",
                "type": "textbox",
                "value": "https://deforum.github.io/a1/M1.jpg",
                "info": ""
            },
            "invert_mask": {
                "label": "Invert mask",
                "type": "checkbox",
                "value": False,
                "info": ""
            },
            "mask_contrast_adjust": {
                "label": "Mask contrast adjust",
                "type": "number",
                "precision": None,
                "value": 1.0,
                "info": ""
            },
            "mask_brightness_adjust": {
                "label": "Mask brightness adjust",
                "type": "number",
                "precision": None,
                "value": 1.0,
                "info": ""
            },
            "overlay_mask": {
                "label": "Overlay mask",
                "type": "checkbox",
                "value": True,
                "info": ""
            },
            "mask_overlay_blur": {
                "label": "Mask overlay blur",
                "type": "slider",
                "min": 0,
                "max": 64,
                "step": 1,
                "value": 4,
            },
            "fill": {
                "label": "Mask fill",
                "type": "radio",
                "radio_type": "index",
                "choices": mask_fill_choices,
                "value": "fill",
                "info": ""
            },
            "full_res_mask": {
                "label": "Full res mask",
                "type": "checkbox",
                "value": True,
                "info": ""
            },
            "full_res_mask_padding": {
                "label": "Full res mask padding",
                "type": "slider",
                "min": 0,
                "max": 512,
                "step": 1,
                "value": 4,
            },
            "reroll_blank_frames": {
                "label": "Reroll blank frames",
                "type": "radio",
                "radio_type": "index",
                "choices": ['reroll', 'interrupt', 'ignore'],
                "value": "ignore",
                "info": ""
            },
            "reroll_patience": {
                "label": "Reroll patience",
                "type": "number",
                "precision": None,
                "value": 10,
                "info": ""
            },
        }

def LoopArgs():
    return {
        "use_looper": {
            "label": "Enable guided images mode",
            "type": "checkbox",
            "value": False,
        },
        "init_images": {
            "label": "Images to use for keyframe guidance",
            "type": "textbox",
            "lines": 9,
            "value": get_guided_imgs_default_json(),
        },
        "image_strength_schedule": {
            "label": "Image strength schedule",
            "type": "textbox",
            "value": "0:(0.75)",
        },
        "blendFactorMax": {
            "label": "Blend factor max",
            "type": "textbox",
            "value": "0:(0.35)",
        },
        "blendFactorSlope": {
            "label": "Blend factor slope",
            "type": "textbox",
            "value": "0:(0.25)",
        },
        "tweening_frames_schedule": {
            "label": "Tweening frames schedule",
            "type": "textbox",
            "value": "0:(20)",
        },
        "color_correction_factor": {
            "label": "Color correction factor",
            "type": "textbox",
            "value": "0:(0.075)",
        }
    }

def ParseqArgs():
    return {
        "parseq_manifest": {
            "label": "Parseq Manifest (JSON or URL)",
            "type": "textbox",
            "lines": 4,
            "value": None,
        },
        "parseq_use_deltas": {
            "label": "Use delta values for movement parameters",
            "type": "checkbox",
            "value": True,
        }
    }

def DeforumOutputArgs():
    return {
        "skip_video_creation": {
            "label": "Skip video creation",
            "type": "checkbox",
            "value": False,
            "info": "If enabled, only images will be saved"
        },
        "fps": {
                "label": "FPS",
                "type": "slider",
                "min": 1,
                "max": 240,
                "step": 1,
                "value": 15,
        },
        "make_gif": {
            "label": "Make GIF",
            "type": "checkbox",
            "value": False,
            "info": "make gif in addition to the video/s"
        },
        "delete_imgs": {
            "label": "Delete Imgs",
            "type": "checkbox",
            "value": False,
            "info": "auto-delete imgs when video is ready"
        },
            #False,  # True will delete all imgs after a successful mp4 creation
        "image_path": "C:/SD/20230124234916_%09d.png",
        "add_soundtrack": {
                "label": "Add soundtrack",
                "type": "radio",
                "choices": ['None', 'File', 'Init Video'],
                "value": "None",
                "info": "add audio to video from file/url or init video"
        },
        "soundtrack_path": {
            "label": "Soundtrack path",
            "type": "textbox",
            "value": "https://deforum.github.io/a1/A1.mp3",
            "info": "abs. path or url to audio file"
        },
        # End-Run upscaling
        "r_upscale_video": {
            "label": "Upscale",
            "type": "checkbox",
            "value": False,
            "info": "upscale output imgs when run is finished"
        },
        "r_upscale_factor": 'x2',  # ['2x', 'x3', 'x4'],
        "r_upscale_model": {
                "label": "Upscale model",
                "type": "dropdown",
                "choices": ['realesr-animevideov3', 'realesrgan-x4plus', 'realesrgan-x4plus-anime'],
                "value": 'realesr-animevideov3',
            },
        "r_upscale_keep_imgs": True,
        "store_frames_in_ram": {
            "label": "Store frames in ram",
            "type": "checkbox",
            "value": False,
            "info": "auto-delete imgs when video is ready",
            "visible": False
        },
        # **Interpolate Video Settings**
        "frame_interpolation_engine": "None",  # ["None", "RIFE v4.6", "FILM"]
        "frame_interpolation_x_amount": 2,  # [2 to 1000 depends on the engine]
        "frame_interpolation_slow_mo_enabled": False,
        "frame_interpolation_slow_mo_amount": 2,  # [2 to 10]
        "frame_interpolation_keep_imgs": False
    }

def get_component_names():
    return ['override_settings_with_file', 'custom_settings_file', *DeforumAnimArgs().keys(), 'animation_prompts', 'animation_prompts_positive', 'animation_prompts_negative',
            *DeforumArgs().keys(), *DeforumOutputArgs().keys(), *ParseqArgs().keys(), *LoopArgs().keys(), *controlnet_component_names()]

def get_settings_component_names():
    return [name for name in get_component_names()]

def pack_default_args(args_dict):
    args_dict = {name: args_dict[name] for name in DeforumArgs()}
    args_dict.update({name: CoreArgs()[name] for name in CoreArgs()})
    return args_dict

def pack_args(args_dict, arg_set):
    return {name: args_dict[name] for name in arg_set()}

def process_args(args_dict_main, run_id):
    from .settings import load_args
    override_settings_with_file = args_dict_main['override_settings_with_file']
    custom_settings_file = args_dict_main['custom_settings_file']
    args_dict = pack_default_args(args_dict_main)
    anim_args_dict = pack_args(args_dict_main, DeforumAnimArgs)
    video_args_dict = pack_args(args_dict_main, DeforumOutputArgs)
    parseq_args_dict = pack_args(args_dict_main, ParseqArgs)
    loop_args_dict = pack_args(args_dict_main, LoopArgs)
    controlnet_args_dict = pack_args(args_dict_main, controlnet_component_names)

    root = SimpleNamespace(**RootArgs())
    p = args_dict_main['p']
    root.animation_prompts = json.loads(args_dict_main['animation_prompts'])

    args_loaded_ok = True  # can use this later to error cleanly upon wrong gen param in ui
    if override_settings_with_file:
        args_loaded_ok = load_args(args_dict_main, args_dict, anim_args_dict, parseq_args_dict, loop_args_dict, controlnet_args_dict, video_args_dict, custom_settings_file, root, run_id)

    positive_prompts = args_dict_main['animation_prompts_positive']
    negative_prompts = args_dict_main['animation_prompts_negative']
    negative_prompts = negative_prompts.replace('--neg', '')  # remove --neg from negative_prompts if received by mistake
    root.animation_prompts = {key: f"{positive_prompts} {val} {'' if '--neg' in val else '--neg'} {negative_prompts}" for key, val in root.animation_prompts.items()}

    os.makedirs(root.models_path, exist_ok=True)  # TODO: this can and probably should be removed from here to the launch of the webui funcs

    args = SimpleNamespace(**args_dict)
    anim_args = SimpleNamespace(**anim_args_dict)
    video_args = SimpleNamespace(**video_args_dict)
    parseq_args = SimpleNamespace(**parseq_args_dict)
    loop_args = SimpleNamespace(**loop_args_dict)
    controlnet_args = SimpleNamespace(**controlnet_args_dict)

    if args.seed == -1:
        root.raw_seed = -1
    args.seed = get_fixed_seed(args.seed)
    if root.raw_seed != -1:
        root.raw_seed = args.seed
    args.timestring = time.strftime('%Y%m%d%H%M%S')
    args.strength = max(0.0, min(1.0, args.strength))
    args.prompts = json.loads(args_dict_main['animation_prompts'])
    args.positive_prompts = args_dict_main['animation_prompts_positive']
    args.negative_prompts = args_dict_main['animation_prompts_negative']

    if not args.use_init and not anim_args.hybrid_use_init_image:
        args.init_image = None

    if anim_args.animation_mode == 'None':
        anim_args.max_frames = 1
    elif anim_args.animation_mode == 'Video Input':
        args.use_init = True

    current_arg_list = [args, anim_args, video_args, parseq_args]
    full_base_folder_path = os.path.join(os.getcwd(), p.outpath_samples)
    root.raw_batch_name = args.batch_name
    args.batch_name = substitute_placeholders(args.batch_name, current_arg_list, full_base_folder_path)
    args.outdir = os.path.join(p.outpath_samples, str(args.batch_name))
    args.outdir = os.path.join(os.getcwd(), args.outdir)
    os.makedirs(args.outdir, exist_ok=True)

    return args_loaded_ok, root, args, anim_args, video_args, parseq_args, loop_args, controlnet_args
