from .easing import easing_linear_interp

def get_cadence_keys(keys, tween_frame_idx):
    diffusion_cadence_easing = keys.diffusion_cadence_easing_schedule_series[tween_frame_idx]
    cadence_flow_easing = keys.cadence_flow_easing_schedule_series[tween_frame_idx]
    hybrid_flow_factor = keys.hybrid_flow_factor_schedule_series[tween_frame_idx]
    cadence_flow_factor = keys.cadence_flow_factor_schedule_series[tween_frame_idx]
    return diffusion_cadence_easing, cadence_flow_easing, hybrid_flow_factor, cadence_flow_factor

def get_cadence_tweens(tween_frame_idx, tween_frame_start_idx, frame_idx, diffusion_cadence_easing, cadence_flow_easing):
    tween = float(tween_frame_idx - tween_frame_start_idx + 1) / float(frame_idx - tween_frame_start_idx)
    # eased tweens locked with master tween
    tween_images = easing_linear_interp(tween, diffusion_cadence_easing)
    tween_flow = easing_linear_interp(tween, cadence_flow_easing)
    return tween, tween_images, tween_flow