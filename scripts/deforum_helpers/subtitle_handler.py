from decimal import Decimal, getcontext
from modules.shared import opts

param_dict = {
    "angle": {"backend": "angle_series", "user": "Angle", "print": "Angle"},
    "transform_center_x": {"backend": "transform_center_x_series", "user": "Trans Center X", "print": "Tr.C.X"},
    "transform_center_y": {"backend": "transform_center_y_series", "user": "Trans Center Y", "print": "Tr.C.Y"},
    "zoom": {"backend": "zoom_series", "user": "Zoom", "print": "Zoom"},
    "translation_x": {"backend": "translation_x_series", "user": "Trans X", "print": "TrX"},
    "translation_y": {"backend": "translation_y_series", "user": "Trans Y", "print": "TrY"},
    "translation_z": {"backend": "translation_z_series", "user": "Trans Z", "print": "TrZ"},
    "rotation_3d_x": {"backend": "rotation_3d_x_series", "user": "Rot 3D X", "print": "RotX"},
    "rotation_3d_y": {"backend": "rotation_3d_y_series", "user": "Rot 3D Y", "print": "RotY"},
    "rotation_3d_z": {"backend": "rotation_3d_z_series", "user": "Rot 3D Z", "print": "RotZ"},
    "perspective_flip_theta": {"backend": "perspective_flip_theta_series", "user": "Per Fl Theta", "print": "PerFlT"},
    "perspective_flip_phi": {"backend": "perspective_flip_phi_series", "user": "Per Fl Phi", "print": "PerFlP"},
    "perspective_flip_gamma": {"backend": "perspective_flip_gamma_series", "user": "Per Fl Gamma", "print": "PerFlG"},
    "perspective_flip_fv": {"backend": "perspective_flip_fv_series", "user": "Per Fl FV", "print": "PerFlFV"},
    "noise_schedule": {"backend": "noise_schedule_series", "user": "Noise Sch", "print": "Noise"},
    "strength_schedule": {"backend": "strength_schedule_series", "user": "Str Sch", "print": "StrSch"},
    "contrast_schedule": {"backend": "contrast_schedule_series", "user": "Contrast Sch", "print": "CtrstSch"},
    "cfg_scale_schedule": {"backend": "cfg_scale_schedule_series", "user": "CFG Sch", "print": "CFGSch"},
    "pix2pix_img_cfg_scale_schedule": {"backend": "pix2pix_img_cfg_scale_series", "user": "P2P Img CFG Sch", "print": "P2PCfgSch"},
    "subseed_schedule": {"backend": "subseed_schedule_series", "user": "Subseed Sch", "print": "SubSSch"},
    "subseed_strength_schedule": {"backend": "subseed_strength_schedule_series", "user": "Subseed Str Sch", "print": "SubSStrSch"},
    "checkpoint_schedule": {"backend": "checkpoint_schedule_series", "user": "Ckpt Sch", "print": "CkptSch"},
    "steps_schedule": {"backend": "steps_schedule_series", "user": "Steps Sch", "print": "StepsSch"},
    "seed_schedule": {"backend": "seed_schedule_series", "user": "Seed Sch", "print": "SeedSch"},
    "sampler_schedule": {"backend": "sampler_schedule_series", "user": "Sampler Sch", "print": "SamplerSchedule"},
    "clipskip_schedule": {"backend": "clipskip_schedule_series", "user": "Clipskip Sch", "print": "ClipskipSchedule"},
    "noise_multiplier_schedule": {"backend": "noise_multiplier_schedule_series", "user": "Noise Multp Sch", "print": "NoiseMultiplierSchedule"},
    "mask_schedule": {"backend": "mask_schedule_series", "user": "Mask Sch", "print": "MaskSchedule"},
    "noise_mask_schedule": {"backend": "noise_mask_schedule_series", "user": "Noise Mask Sch", "print": "NoiseMaskSchedule"},
    "amount_schedule": {"backend": "amount_schedule_series", "user": "Ant.Blr Amount Sch", "print": "AmountSchedule"},
    "kernel_schedule": {"backend": "kernel_schedule_series", "user": "Ant.Blr Kernel Sch", "print": "KernelSchedule"},
    "sigma_schedule": {"backend": "sigma_schedule_series", "user": "Ant.Blr Sigma Sch", "print": "SigmaSchedule"},
    "threshold_schedule": {"backend": "threshold_schedule_series", "user": "Ant.Blr Threshold Sch", "print": "ThresholdSchedule"},
    "aspect_ratio_schedule": {"backend": "aspect_ratio_series", "user": "Aspect Ratio Sch", "print": "AspectRatioSchedule"},
    "fov_schedule": {"backend": "fov_series", "user": "FOV Sch", "print": "FieldOfViewSchedule"},
    "near_schedule": {"backend": "near_series", "user": "Near Sch", "print": "NearSchedule"},
    "cadence_flow_factor_schedule": {"backend": "cadence_flow_factor_schedule_series", "user": "Cadence Flow Factor Sch", "print": "CadenceFlowFactorSchedule"},
    "redo_flow_factor_schedule": {"backend": "redo_flow_factor_schedule_series", "user": "Redo Flow Factor Sch", "print": "RedoFlowFactorSchedule"},
    "far_schedule": {"backend": "far_series", "user": "Far Sch", "print": "FarSchedule"},
    "hybrid_comp_alpha_schedule": {"backend": "hybrid_comp_alpha_schedule_series", "user": "Hyb Comp Alpha Sch", "print": "HybridCompAlphaSchedule"},
    "hybrid_comp_mask_blend_alpha_schedule": {"backend": "hybrid_comp_mask_blend_alpha_schedule_series", "user": "Hyb Comp Mask Blend Alpha Sch", "print": "HybridCompMaskBlendAlphaSchedule"},
    "hybrid_comp_mask_contrast_schedule": {"backend": "hybrid_comp_mask_contrast_schedule_series", "user": "Hyb Comp Mask Ctrst Sch", "print": "HybridCompMaskContrastSchedule"},
    "hybrid_comp_mask_auto_contrast_cutoff_high_schedule": {"backend": "hybrid_comp_mask_auto_contrast_cutoff_high_schedule_series", "user": "Hyb Comp Mask Auto Contrast Cutoff High Sch", "print": "HybridCompMaskAutoContrastCutoffHighSchedule"},
    "hybrid_comp_mask_auto_contrast_cutoff_low_schedule": {"backend": "hybrid_comp_mask_auto_contrast_cutoff_low_schedule_series", "user": "Hyb Comp Mask Auto Ctrst Cut Low Sch", "print": "HybridCompMaskAutoContrastCutoffLowSchedule"},
    "hybrid_flow_factor_schedule": {"backend": "hybrid_flow_factor_schedule_series", "user": "Hybrid Flow Factor Sch", "print": "HybridFlowFactorSchedule"},
}

def time_to_srt_format(seconds):
    hours, remainder = divmod(seconds, 3600)
    minutes, remainder = divmod(remainder, 60)
    seconds, milliseconds = divmod(remainder, 1)
    return f"{hours:02}:{minutes:02}:{int(seconds):02},{int(milliseconds * 1000):03}"

def init_srt_file(filename, fps, precision=20):
    with open(filename, "w") as f:
        pass
    getcontext().prec = precision
    frame_duration = Decimal(1) / Decimal(fps)
    return frame_duration

def write_frame_subtitle(filename, frame_number, frame_duration, text):
    frame_start_time = Decimal(frame_number) * frame_duration
    frame_end_time = (Decimal(frame_number) + Decimal(1)) * frame_duration

    with open(filename, "a") as f:
        f.write(f"{frame_number + 1}\n")
        f.write(f"{time_to_srt_format(frame_start_time)} --> {time_to_srt_format(frame_end_time)}\n")
        f.write(f"{text}\n\n")

def format_animation_params(keys, prompt_series, frame_idx):
    params_to_print = opts.data.get("deforum_save_gen_info_as_srt_params", ['Seed'])
    params_string = ""
    for key, value in param_dict.items():
        if value['user'] in params_to_print:
            backend_key = value['backend']
            print_key = value['print']
            param_value = getattr(keys, backend_key)[frame_idx]
            if isinstance(param_value, float) and param_value == int(param_value):
                formatted_value = str(int(param_value))
            elif isinstance(param_value, float) and not param_value.is_integer():
                formatted_value = f"{param_value:.3f}"
            else:
                formatted_value = f"{param_value}"
            params_string += f"{print_key}: {formatted_value}; "

    if "Prompt" in params_to_print:
        params_string += f"Prompt: {prompt_series[frame_idx]}; "        

    params_string = params_string.rstrip("; ")  # Remove trailing semicolon and whitespace
    return params_string
    
def get_user_values():
    items = [v["user"] for v in param_dict.values()]
    items.append("Prompt")
    return items