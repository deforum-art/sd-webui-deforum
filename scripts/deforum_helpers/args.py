import json
import os
import tempfile
import time
from types import SimpleNamespace
import modules.paths as ph
import modules.shared as sh
from modules.sd_samplers import samplers_for_img2img
from modules.processing import get_fixed_seed
from .defaults import get_guided_imgs_default_json, mask_fill_choices
from .deforum_controlnet import controlnet_component_names
from .general_utils import get_os, substitute_placeholders

def RootArgs():
    return {
        "device": sh.device,
        "models_path": ph.models_path + '/Deforum',
        "half_precision": not sh.cmd_opts.no_half,
        "clipseg_model": None,
        "mask_preset_names": ['everywhere', 'video_mask'],
        "frames_cache": [],
        "raw_batch_name": None,
        "raw_seed": None,
        "timestring": "",
        "subseed": -1,
        "subseed_strength": 0,
        "seed_internal": 0,
        "init_sample": None,
        "noise_mask": None,
        "initial_info": None,
        "first_frame": None,
        "animation_prompts": None,
        "current_user_os": get_os(),
        "tmp_deforum_run_duplicated_folder": os.path.join(tempfile.gettempdir(), 'tmp_run_deforum')
    }

# 'Midas-3.1-BeitLarge' is temporarily removed until fixed. Can add it back anytime as it's supported in the back-end depth code
def DeforumAnimArgs():
    return {
        "animation_mode": {
            "label": "Animation mode",
            "type": "radio",
            "choices": ['2D', '3D', 'Video Input', 'Interpolation'],
            "value": "2D",
            "info": "control animation mode, will hide non relevant params upon change"
        },
        "max_frames": {
            "label": "Max frames",
            "type": "number",
            "precision": 0,
            "value": 120,
            "info": "end the animation at this frame number",
        },
        "border": {
            "label": "Border mode",
            "type": "radio",
            "choices": ['replicate', 'wrap'],
            "value": "replicate",
            "info": "controls pixel generation method for images smaller than the frame. hover on the options to see more info"
        },
        "angle": {
            "label": "Angle",
            "type": "textbox",
            "value": "0: (0)",
            "info": "rotate canvas clockwise/anticlockwise in degrees per frame"
        },

        "zoom": {
            "label": "Zoom",
            "type": "textbox",
            "value": "0: (1.0025+0.002*sin(1.25*3.14*t/30))",
            "info": "scale the canvas size, multiplicatively. [static = 1.0]"
        },

        "translation_x": {
            "label": "Translation X",
            "type": "textbox",
            "value": "0: (0)",
            "info": "move canvas left/right in pixels per frame"
        },

        "translation_y": {
            "label": "Translation Y",
            "type": "textbox",
            "value": "0: (0)",
            "info": "move canvas up/down in pixels per frame"
        },
        "translation_z": {
            "label": "Translation Z",
            "type": "textbox",
            "value": "0: (1.75)",
            "info": "move canvas towards/away from view [speed set by FOV]"
        },
        "transform_center_x": {
            "label": "Transform Center X",
            "type": "textbox",
            "value": "0: (0.5)",
            "info": "X center axis for 2D angle/zoom"
        },

        "transform_center_y": {
            "label": "Transform Center Y",
            "type": "textbox",
            "value": "0: (0.5)",
            "info": "Y center axis for 2D angle/zoom"
        },
        "rotation_3d_x": {
            "label": "Rotation 3D X",
            "type": "textbox",
            "value": "0: (0)",
            "info": "tilt canvas up/down in degrees per frame"
        },
        "rotation_3d_y": {
            "label": "Rotation 3D Y",
            "type": "textbox",
            "value": "0: (0)",
            "info": "pan canvas left/right in degrees per frame"
        },
        "rotation_3d_z": {
            "label": "Rotation 3D Z",
            "type": "textbox",
            "value": "0: (0)",
            "info": "roll canvas clockwise/anticlockwise"
        },
        "enable_perspective_flip": {
            "label": "Enable perspective flip",
            "type": "checkbox",
            "value": False,
            "info": ""
        },
        "perspective_flip_theta": {
            "label": "Perspective flip theta",
            "type": "textbox",
            "value": "0: (0)",
            "info": ""
        },
        "perspective_flip_phi": {
            "label": "Perspective flip phi",
            "type": "textbox",
            "value": "0: (0)",
            "info": ""
        },
        "perspective_flip_gamma": {
            "label": "Perspective flip gamma",
            "type": "textbox",
            "value": "0: (0)",
            "info": ""
        },
        "perspective_flip_fv": {
            "label": "Perspective flip tv",
            "type": "textbox",
            "value": "0: (53)",
            "info": "the 2D vanishing point of perspective (rec. range 30-160)"
        },
        "noise_schedule": {
            "label": "Noise schedule",
            "type": "textbox",
            "value": "0: (0.065)",
            "info": ""
        },
        "strength_schedule": {
            "label": "Strength schedule",
            "type": "textbox",
            "value": "0: (0.65)",
            "info": "amount of presence of previous frame to influence next frame, also controls steps in the following formula [steps - (strength_schedule * steps)]"
        },
        "contrast_schedule": "0: (1.0)",
        "cfg_scale_schedule": {
            "label": "CFG scale schedule",
            "type": "textbox",
            "value": "0: (7)",
            "info": "how closely the image should conform to the prompt. Lower values produce more creative results. (recommended range 5-15)`"
        },
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
        "fov_schedule": {
            "label": "FOV schedule",
            "type": "textbox",
            "value": "0: (70)",
            "info": "adjusts the scale at which the canvas is moved in 3D by the translation_z value. [maximum range -180 to +180, with 0 being undefined. Values closer to 180 will make the image have less depth, while values closer to 0 will allow more depth]"
        },
        "aspect_ratio_schedule": {
            "label": "Aspect Ratio schedule",
            "type": "textbox",
            "value": "0: (1)",
            "info": "adjusts the aspect ratio for the depth calculations"
        },
        "aspect_ratio_use_old_formula": {
            "label": "Use old aspect ratio formula",
            "type": "checkbox",
            "value": False,
            "info": "for backward compatibility. uses the formula: `width/height`"
        },
        "near_schedule": {
            "label": "Near schedule",
            "type": "textbox",
            "value": "0: (200)",
            "info": ""
        },
        "far_schedule": {
            "label": "Far schedule",
            "type": "textbox",
            "value": "0: (10000)",
            "info": ""
        },
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
        "enable_sampler_scheduling": {
            "label": "Enable sampler scheduling",
            "type": "checkbox",
            "value": False,
            "info": ""
        },
        "sampler_schedule": {
            "label": "Sampler schedule",
            "type": "textbox",
            "value": '0: ("Euler a")',
            "info": "allows keyframing different samplers. Use names as they appear in ui dropdown in 'run' tab"
        },
        "use_noise_mask": {
            "label": "Use noise mask",
            "type": "checkbox",
            "value": False,
            "info": ""
        },
        "mask_schedule": {
            "label": "Mask schedule",
            "type": "textbox",
            "value": '0: ("{video_mask}")',
            "info": ""
        },
        "noise_mask_schedule": {
            "label": "Noise mask schedule",
            "type": "textbox",
            "value": '0: ("{video_mask}")',
            "info": ""
        },
        "enable_checkpoint_scheduling": {
            "label": "Enable checkpoint scheduling",
            "type": "checkbox",
            "value": False,
            "info": ""
        },
        "checkpoint_schedule": {
            "label": "allows keyframing different sd models. use *full* name as appears in ui dropdown",
            "type": "textbox",
            "value": '0: ("model1.ckpt"), 100: ("model2.safetensors")',
            "info": "allows keyframing different sd models. use *full* name as appears in ui dropdown"
        },
        "enable_clipskip_scheduling": {
            "label": "Enable CLIP skip scheduling",
            "type": "checkbox",
            "value": False,
            "info": ""
        },
        "clipskip_schedule": {
            "label": "CLIP skip schedule",
            "type": "textbox",
            "value": "0: (2)",
            "info": ""
        },
        "enable_noise_multiplier_scheduling": {
            "label": "Enable noise multiplier scheduling",
            "type": "checkbox",
            "value": True,
            "info": ""
        },
        "noise_multiplier_schedule": {
            "label": "Noise multiplier schedule",
            "type": "textbox",
            "value": "0: (1.05)",
            "info": ""
        },
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
            "info": "noise multiplier; higher = more unpredictable results"
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
            "info": "noise multiplier; applies to Euler a and other samplers that have the letter 'a' in them"
        },
        "ancestral_eta_schedule": {
            "label": "Ancestral ETA Schedule",
            "type": "textbox",
            "value": "0: (1)",
            "visible": False,
            "info": ""
        },
        "amount_schedule": {
            "label": "Amount schedule",
            "type": "textbox",
            "value": "0: (0.1)",
            "info": ""
        },
        "kernel_schedule": {
            "label": "Kernel schedule",
            "type": "textbox",
            "value": "0: (5)",
            "info": ""
        },
        "sigma_schedule": {
            "label": "Sigma schedule",
            "type": "textbox",
            "value": "0: (1)",
            "info": ""
        },
        "threshold_schedule": {
            "label": "Threshold schedule",
            "type": "textbox",
            "value": "0: (0)",
            "info": ""
        },
        "color_coherence": {
            "label": "Color coherence",
            "type": "dropdown",
            "choices": ['None', 'HSV', 'LAB', 'RGB', 'Video Input', 'Image'],
            "value": "LAB",
            "info": "choose an algorithm/ method for keeping color coherence across the animation"
        },
        "color_coherence_image_path": {
            "label": "Color coherence image path",
            "type": "textbox",
            "value": "",
            "info": ""
        },
        "color_coherence_video_every_N_frames": {
            "label": "Color coherence video every N frames",
            "type": "number",
            "precision": 0,
            "value": 1,
            "info": "",
        },
        "color_force_grayscale": {
            "label": "Color force Grayscale",
            "type": "checkbox",
            "value": False,
            "info": "force all frames to be in grayscale"
        },
        "legacy_colormatch": {
            "label": "Legacy colormatch",
            "type": "checkbox",
            "value": False,
            "info": "apply colormatch before adding noise (use with CN's Tile)"
        },
        "diffusion_cadence": {
            "label": "Cadence",
            "type": "slider",
            "minimum": 1,
            "maximum": 50,
            "step": 1,
            "value": 2,
            "info": "# of in-between frames that will not be directly diffused"
        },
        "optical_flow_cadence": {
            "label": "Optical flow cadence",
            "type": "dropdown",
            "choices": ['None', 'RAFT', 'DIS Medium', 'DIS Fine', 'Farneback'],
            "value": "None",
            "info": "use optical flow estimation for your in-between (cadence) frames"
        },
        "cadence_flow_factor_schedule": {
            "label": "Cadence flow factor schedule",
            "type": "textbox",
            "value": "0: (1)",
            "info": ""
        },
        "optical_flow_redo_generation": {
            "label": "Optical flow generation",
            "type": "dropdown",
            "choices": ['None', 'RAFT', 'DIS Medium', 'DIS Fine', 'Farneback'],
            "value": "None",
            "info": "this option takes twice as long because it generates twice in order to capture the optical flow from the previous image to the first generation, then warps the previous image and redoes the generation"
        },
        "redo_flow_factor_schedule": {
            "label": "Generation flow factor schedule",
            "type": "textbox",
            "value": "0: (1)",
            "info": ""
        },
        "diffusion_redo": '0',
        "noise_type": {
            "label": "Noise type",
            "type": "radio",
            "choices": ['uniform', 'perlin'],
            "value": "perlin",
            "info": ""
        },
        "perlin_w": {
            "label": "Perlin W",
            "type": "slider",
            "minimum": 0.1,
            "maximum": 16,
            "step": 0.1,
            "value": 8,
            "visible": False
        },
        "perlin_h": {
            "label": "Perlin H",
            "type": "slider",
            "minimum": 0.1,
            "maximum": 16,
            "step": 0.1,
            "value": 8,
            "visible": False
        },
        "perlin_octaves": {
            "label": "Perlin octaves",
            "type": "slider",
            "minimum": 1,
            "maximum": 7,
            "step": 1,
            "value": 4
        },
        "perlin_persistence": {
            "label": "Perlin persistence",
            "type": "slider",
            "minimum": 0,
            "maximum": 1,
            "step": 0.02,
            "value": 0.5
        },
        "use_depth_warping": {
            "label": "Use depth warping",
            "type": "checkbox",
            "value": True,
            "info": ""
        },
        "depth_algorithm": {
            "label": "Depth Algorithm",
            "type": "dropdown",
            "choices": ['Midas+AdaBins (old)', 'Zoe+AdaBins (old)', 'Midas-3-Hybrid', 'AdaBins', 'Zoe', 'Leres'],
            "value": "Midas-3-Hybrid",
            "info": "choose an algorithm/ method for keeping color coherence across the animation"
        },
        "midas_weight": {
            "label": "MiDaS/Zoe weight",
            "type": "number",
            "precision": None,
            "value": 0.2,
            "info": "sets a midpoint at which a depth-map is to be drawn: range [-1 to +1]",
            "visible": False
        },
        "padding_mode": {
            "label": "Padding mode",
            "type": "radio",
            "choices": ['border', 'reflection', 'zeros'],
            "value": "border",
            "info": "controls the handling of pixels outside the field of view as they come into the scene"
        },
        "sampling_mode": {
            "label": "Padding mode",
            "type": "radio",
            "choices": ['bicubic', 'bilinear', 'nearest'],
            "value": "bicubic",
            "info": ""
        },
        "save_depth_maps": {
            "label": "Save 3D depth maps",
            "type": "checkbox",
            "value": False,
            "info": "save animation's depth maps as extra files"
        },
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
        "hybrid_comp_alpha_schedule": {
            "label": "Comp alpha schedule",
            "type": "textbox",
            "value": "0:(0.5)",
            "info": ""
        },
        "hybrid_comp_mask_blend_alpha_schedule": {
            "label": "Comp mask blend alpha schedule",
            "type": "textbox",
            "value": "0:(0.5)",
            "info": ""
        },
        "hybrid_comp_mask_contrast_schedule": {
            "label": "Comp mask contrast schedule",
            "type": "textbox",
            "value": "0:(1)",
            "info": ""
        },
        "hybrid_comp_mask_auto_contrast_cutoff_high_schedule": {
            "label": "Comp mask auto contrast cutoff high schedule",
            "type": "textbox",
            "value": "0:(100)",
            "info": ""
        },
        "hybrid_comp_mask_auto_contrast_cutoff_low_schedule": {
            "label": "Comp mask auto contrast cutoff low schedule",
            "type": "textbox",
            "value": "0:(0)",
            "info": ""
        },
        "hybrid_flow_factor_schedule": {
            "label": "Flow factor schedule",
            "type": "textbox",
            "value": "0:(1)",
            "info": ""
        },
        "hybrid_generate_inputframes": {
            "label": "Generate inputframes",
            "type": "checkbox",
            "value": False,
            "info": ""
        },
        "hybrid_generate_human_masks": {
            "label": "Generate human masks",
            "type": "radio",
            "choices": ['None', 'PNGs', 'Video', 'Both'],
            "value": "None",
            "info": ""
        },
        "hybrid_use_first_frame_as_init_image": {
            "label": "First frame as init image",
            "type": "checkbox",
            "value": True,
            "info": "",
            "visible": False
        },
        "hybrid_motion": {
            "label": "Hybrid motion",
            "type": "radio",
            "choices": ['None', 'Optical Flow', 'Perspective', 'Affine'],
            "value": "None",
            "info": ""
        },
        "hybrid_motion_use_prev_img": {
            "label": "Motion use prev img",
            "type": "checkbox",
            "value": False,
            "info": "",
            "visible": False
        },
        "hybrid_flow_consistency": {
            "label": "Flow consistency mask",
            "type": "checkbox",
            "value": False,
            "info": "",
            "visible": False
        },
        "hybrid_consistency_blur": {
            "label": "Consistency mask blur",
            "type": "slider",
            "minimum": 0,
            "maximum": 16,
            "step": 1,
            "value": 2,
            "visible": False
        },
        "hybrid_flow_method": {
            "label": "Flow method",
            "type": "radio",
            "choices": ['RAFT', 'DIS Medium', 'DIS Fine', 'Farneback'],
            "value": "RAFT",
            "info": "",
            "visible": False
        },
        "hybrid_composite": 'None',  # ['None', 'Normal', 'Before Motion', 'After Generation']
        "hybrid_use_init_image": {
            "label": "Use init image as video",
            "type": "checkbox",
            "value": False,
            "info": "",
        },
        "hybrid_comp_mask_type": {
            "label": "Comp mask type",
            "type": "radio",
            "choices": ['None', 'Depth', 'Video Depth', 'Blend', 'Difference'],
            "value": "None",
            "info": "",
            "visible": False
        },
        "hybrid_comp_mask_inverse": False,
        "hybrid_comp_mask_equalize": {
            "label": "Comp mask equalize",
            "type": "radio",
            "choices": ['None', 'Before', 'After', 'Both'],
            "value": "None",
            "info": "",
        },
        "hybrid_comp_mask_auto_contrast": False,
        "hybrid_comp_save_extra_frames": False
    }

def DeforumArgs():
    return {
        "W": {
            "label": "Width",
            "type": "slider",
            "minimum": 64,
            "maximum": 2048,
            "step": 64,
            "value": 512,
        },
        "H": {
            "label": "Height",
            "type": "slider",
            "minimum": 64,
            "maximum": 2048,
            "step": 64,
            "value": 512,
        },
        "show_info_on_ui": True,
        "tiling": {
            "label": "Tiling",
            "type": "checkbox",
            "value": False,
            "info": "enable for seamless-tiling of each generated image. Experimental"
        },
        "restore_faces": {
            "label": "Restore faces",
            "type": "checkbox",
            "value": False,
            "info": "enable to trigger webui's face restoration on each frame during the generation"
        },
        "seed_resize_from_w": {
            "label": "Resize seed from width",
            "type": "slider",
            "minimum": 0,
            "maximum": 2048,
            "step": 64,
            "value": 0,
        },
        "seed_resize_from_h": {
            "label": "Resize seed from height",
            "type": "slider",
            "minimum": 0,
            "maximum": 2048,
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
            "minimum": 1,
            "maximum": 200,
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
            "minimum": 0,
            "maximum": 1,
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
            "minimum": 0,
            "maximum": 64,
            "step": 1,
            "value": 4,
        },
        "fill": {
            "label": "Mask fill",
            "type": "radio",
            "radio_type": "index",
            "choices": ['fill', 'original', 'latent noise', 'latent nothing'],
            "value": 'original',
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
            "minimum": 0,
            "maximum": 512,
            "step": 1,
            "value": 4,
        },
        "reroll_blank_frames": {
            "label": "Reroll blank frames",
            "type": "radio",
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
            "minimum": 1,
            "maximum": 240,
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
        "image_path": {
            "label": "Image path",
            "type": "textbox",
            "value": "C:/SD/20230124234916_%09d.png",
        },
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
        "r_upscale_video": {
            "label": "Upscale",
            "type": "checkbox",
            "value": False,
            "info": "upscale output imgs when run is finished"
        },
        "r_upscale_factor": {
            "label": "Upscale factor",
            "type": "dropdown",
            "choices": ['x2', 'x3', 'x4'],
            "value": "x2",
        },
        "r_upscale_model": {
            "label": "Upscale model",
            "type": "dropdown",
            "choices": ['realesr-animevideov3', 'realesrgan-x4plus', 'realesrgan-x4plus-anime'],
            "value": 'realesr-animevideov3',
        },
        "r_upscale_keep_imgs": {
            "label": "Keep Imgs",
            "type": "checkbox",
            "value": True,
            "info": "don't delete upscaled imgs",
        },
        "store_frames_in_ram": {
            "label": "Store frames in ram",
            "type": "checkbox",
            "value": False,
            "info": "auto-delete imgs when video is ready",
            "visible": False
        },
        "frame_interpolation_engine": {
            "label": "Engine",
            "type": "radio",
            "choices": ['None', 'RIFE v4.6', 'FILM'],
            "value": "None",
            "info": "select the frame interpolation engine. hover on the options for more info"
        },
        "frame_interpolation_x_amount": {
            "label": "Interp X",
            "type": "slider",
            "minimum": 2,
            "maximum": 10,
            "step": 1,
            "value": 2,
        },
        "frame_interpolation_slow_mo_enabled": {
            "label": "Slow Mo",
            "type": "checkbox",
            "value": False,
            "visible": False,
            "info": "Slow-Mo the interpolated video, audio will not be used if enabled",
        },
        "frame_interpolation_slow_mo_amount": {
            "label": "Slow-Mo X",
            "type": "slider",
            "minimum": 2,
            "maximum": 10,
            "step": 1,
            "value": 2,
        },
        "frame_interpolation_keep_imgs": {
            "label": "Keep Imgs",
            "type": "checkbox",
            "value": False,
            "info": "Keep interpolated images on disk",
            "visible": False
        },
        "frame_interpolation_use_upscaled": {
            "label": "Use Upscaled",
            "type": "checkbox",
            "value": False,
            "info": "Interpolate upscaled images, if available",
            "visible": False
        },        

    }

def get_component_names():
    return ['override_settings_with_file', 'custom_settings_file', *DeforumAnimArgs().keys(), 'animation_prompts', 'animation_prompts_positive', 'animation_prompts_negative',
            *DeforumArgs().keys(), *DeforumOutputArgs().keys(), *ParseqArgs().keys(), *LoopArgs().keys(), *controlnet_component_names()]

def get_settings_component_names():
    return [name for name in get_component_names()]

def pack_args(args_dict, keys_function):
    return {name: args_dict[name] for name in keys_function()}

def process_args(args_dict_main, run_id):
    from .settings import load_args
    override_settings_with_file = args_dict_main['override_settings_with_file']
    custom_settings_file = args_dict_main['custom_settings_file']
    p = args_dict_main['p']

    root = SimpleNamespace(**RootArgs())
    args = SimpleNamespace(**{name: args_dict_main[name] for name in DeforumArgs()})
    anim_args = SimpleNamespace(**{name: args_dict_main[name] for name in DeforumAnimArgs()})
    video_args = SimpleNamespace(**{name: args_dict_main[name] for name in DeforumOutputArgs()})
    parseq_args = SimpleNamespace(**{name: args_dict_main[name] for name in ParseqArgs()})
    loop_args = SimpleNamespace(**{name: args_dict_main[name] for name in LoopArgs()})
    controlnet_args = SimpleNamespace(**{name: args_dict_main[name] for name in controlnet_component_names()})

    root.animation_prompts = json.loads(args_dict_main['animation_prompts'])

    args_loaded_ok = True
    if override_settings_with_file:
        args_loaded_ok = load_args(args_dict_main, args, anim_args, parseq_args, loop_args, controlnet_args, video_args, custom_settings_file, root, run_id)

    positive_prompts = args_dict_main['animation_prompts_positive']
    negative_prompts = args_dict_main['animation_prompts_negative']
    negative_prompts = negative_prompts.replace('--neg', '')  # remove --neg from negative_prompts if received by mistake
    root.animation_prompts = {key: f"{positive_prompts} {val} {'' if '--neg' in val else '--neg'} {negative_prompts}" for key, val in root.animation_prompts.items()}

    if args.seed == -1:
        root.raw_seed = -1
    args.seed = get_fixed_seed(args.seed)
    if root.raw_seed != -1:
        root.raw_seed = args.seed
    root.timestring = time.strftime('%Y%m%d%H%M%S')
    args.strength = max(0.0, min(1.0, args.strength))
    args.prompts = json.loads(args_dict_main['animation_prompts'])
    args.positive_prompts = args_dict_main['animation_prompts_positive']
    args.negative_prompts = args_dict_main['animation_prompts_negative']

    if not args.use_init and not anim_args.hybrid_use_init_image:
        args.init_image = None

    elif anim_args.animation_mode == 'Video Input':
        args.use_init = True

    current_arg_list = [args, anim_args, video_args, parseq_args, root]
    full_base_folder_path = os.path.join(os.getcwd(), p.outpath_samples)
    root.raw_batch_name = args.batch_name
    args.batch_name = substitute_placeholders(args.batch_name, current_arg_list, full_base_folder_path)
    args.outdir = os.path.join(p.outpath_samples, str(args.batch_name))
    args.outdir = os.path.join(os.getcwd(), args.outdir)
    os.makedirs(args.outdir, exist_ok=True)

    return args_loaded_ok, root, args, anim_args, video_args, parseq_args, loop_args, controlnet_args
