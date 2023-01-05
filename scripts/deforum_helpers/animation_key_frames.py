import re
import numpy as np
import numexpr
import pandas as pd
from .prompt import check_is_number

class DeformAnimKeys():
    def __init__(self, anim_args):
        self.angle_series = get_inbetweens(parse_key_frames(anim_args.angle), anim_args.max_frames)
        self.zoom_series = get_inbetweens(parse_key_frames(anim_args.zoom), anim_args.max_frames)
        self.translation_x_series = get_inbetweens(parse_key_frames(anim_args.translation_x), anim_args.max_frames)
        self.translation_y_series = get_inbetweens(parse_key_frames(anim_args.translation_y), anim_args.max_frames)
        self.translation_z_series = get_inbetweens(parse_key_frames(anim_args.translation_z), anim_args.max_frames)
        self.rotation_3d_x_series = get_inbetweens(parse_key_frames(anim_args.rotation_3d_x), anim_args.max_frames)
        self.rotation_3d_y_series = get_inbetweens(parse_key_frames(anim_args.rotation_3d_y), anim_args.max_frames)
        self.rotation_3d_z_series = get_inbetweens(parse_key_frames(anim_args.rotation_3d_z), anim_args.max_frames)
        self.perspective_flip_theta_series = get_inbetweens(parse_key_frames(anim_args.perspective_flip_theta), anim_args.max_frames)
        self.perspective_flip_phi_series = get_inbetweens(parse_key_frames(anim_args.perspective_flip_phi), anim_args.max_frames)
        self.perspective_flip_gamma_series = get_inbetweens(parse_key_frames(anim_args.perspective_flip_gamma), anim_args.max_frames)
        self.perspective_flip_fv_series = get_inbetweens(parse_key_frames(anim_args.perspective_flip_fv), anim_args.max_frames)
        self.noise_schedule_series = get_inbetweens(parse_key_frames(anim_args.noise_schedule), anim_args.max_frames)
        self.strength_schedule_series = get_inbetweens(parse_key_frames(anim_args.strength_schedule), anim_args.max_frames)
        self.contrast_schedule_series = get_inbetweens(parse_key_frames(anim_args.contrast_schedule), anim_args.max_frames)
        self.cfg_scale_schedule_series = get_inbetweens(parse_key_frames(anim_args.cfg_scale_schedule), anim_args.max_frames)
        self.seed_schedule_series = get_inbetweens(parse_key_frames(anim_args.seed_schedule), anim_args.max_frames)
        self.sampler_schedule_series = get_inbetweens(parse_key_frames(anim_args.sampler_schedule), anim_args.max_frames, is_single_string = True)
        self.kernel_schedule_series = get_inbetweens(parse_key_frames(anim_args.kernel_schedule), anim_args.max_frames)
        self.sigma_schedule_series = get_inbetweens(parse_key_frames(anim_args.sigma_schedule), anim_args.max_frames)
        self.amount_schedule_series = get_inbetweens(parse_key_frames(anim_args.amount_schedule), anim_args.max_frames)
        self.threshold_schedule_series = get_inbetweens(parse_key_frames(anim_args.threshold_schedule), anim_args.max_frames)
        self.fov_series = get_inbetweens(parse_key_frames(anim_args.fov_schedule), anim_args.max_frames)
        self.near_series = get_inbetweens(parse_key_frames(anim_args.near_schedule), anim_args.max_frames)
        self.far_series = get_inbetweens(parse_key_frames(anim_args.far_schedule), anim_args.max_frames)
        self.hybrid_comp_alpha_schedule_series = get_inbetweens(parse_key_frames(anim_args.hybrid_comp_alpha_schedule), anim_args.max_frames)
        self.hybrid_comp_mask_blend_alpha_schedule_series = get_inbetweens(parse_key_frames(anim_args.hybrid_comp_mask_blend_alpha_schedule), anim_args.max_frames)
        self.hybrid_comp_mask_contrast_schedule_series = get_inbetweens(parse_key_frames(anim_args.hybrid_comp_mask_contrast_schedule), anim_args.max_frames)
        self.hybrid_comp_mask_auto_contrast_cutoff_high_schedule_series = get_inbetweens(parse_key_frames(anim_args.hybrid_comp_mask_auto_contrast_cutoff_high_schedule), anim_args.max_frames)
        self.hybrid_comp_mask_auto_contrast_cutoff_low_schedule_series = get_inbetweens(parse_key_frames(anim_args.hybrid_comp_mask_auto_contrast_cutoff_low_schedule), anim_args.max_frames)

def get_inbetweens(key_frames, max_frames, integer=False, interp_method='Linear', is_single_string = False):
    key_frame_series = pd.Series([np.nan for a in range(max_frames)])
    for i in range(0, max_frames):
        if i in key_frames:
            value = key_frames[i]
            value_is_number = check_is_number(value)
            # if it's only a number, leave the rest for the default interpolation
            if value_is_number:
                t = i
                key_frame_series[i] = value
        if not value_is_number:
            t = i
            if is_single_string:
                if value.find("'") > -1:
                    value = value.replace("'","")
                if value.find('"') > -1:
                    value = value.replace('"',"")
            key_frame_series[i] = numexpr.evaluate(value) if not is_single_string else value # workaround for values formatted like 0:("I am test") //used for sampler schedules
    key_frame_series = key_frame_series.astype(float) if not is_single_string else key_frame_series # as string
    
    if interp_method == 'Cubic' and len(key_frames.items()) <= 3:
        interp_method = 'Quadratic'    
    if interp_method == 'Quadratic' and len(key_frames.items()) <= 2:
        interp_method = 'Linear'
          
    key_frame_series[0] = key_frame_series[key_frame_series.first_valid_index()]
    key_frame_series[max_frames-1] = key_frame_series[key_frame_series.last_valid_index()]
    key_frame_series = key_frame_series.interpolate(method=interp_method.lower(), limit_direction='both')
    if integer:
        return key_frame_series.astype(int)
    return key_frame_series

def parse_key_frames(string, prompt_parser=None):
    # because math functions (i.e. sin(t)) can utilize brackets 
    # it extracts the value in form of some stuff
    # which has previously been enclosed with brackets and
    # with a comma or end of line existing after the closing one
    pattern = r'((?P<frame>[0-9]+):[\s]*\((?P<param>[\S\s]*?)\)([,][\s]?|[\s]?$))'
    frames = dict()
    for match_object in re.finditer(pattern, string):
        frame = int(match_object.groupdict()['frame'])
        param = match_object.groupdict()['param']
        if prompt_parser:
            frames[frame] = prompt_parser(param)
        else:
            frames[frame] = param
    if frames == {} and len(string) != 0:
        raise RuntimeError('Key Frame string not correctly formatted')
    return frames
