import re
import numpy as np
import numexpr
import pandas as pd
from .prompt import check_is_number

class DeformAnimKeys():
    def __init__(self, anim_args, seed=-1):
        self.fi = FrameInterpolater(anim_args.max_frames, seed)
        self.angle_series = self.fi.get_inbetweens(self.fi.parse_key_frames(anim_args.angle))
        self.transform_center_x_series = self.fi.get_inbetweens(self.fi.parse_key_frames(anim_args.transform_center_x))
        self.transform_center_y_series = self.fi.get_inbetweens(self.fi.parse_key_frames(anim_args.transform_center_y))
        self.zoom_series = self.fi.get_inbetweens(self.fi.parse_key_frames(anim_args.zoom))
        self.translation_x_series = self.fi.get_inbetweens(self.fi.parse_key_frames(anim_args.translation_x))
        self.translation_y_series = self.fi.get_inbetweens(self.fi.parse_key_frames(anim_args.translation_y))
        self.translation_z_series = self.fi.get_inbetweens(self.fi.parse_key_frames(anim_args.translation_z))
        self.rotation_3d_x_series = self.fi.get_inbetweens(self.fi.parse_key_frames(anim_args.rotation_3d_x))
        self.rotation_3d_y_series = self.fi.get_inbetweens(self.fi.parse_key_frames(anim_args.rotation_3d_y))
        self.rotation_3d_z_series = self.fi.get_inbetweens(self.fi.parse_key_frames(anim_args.rotation_3d_z))
        self.perspective_flip_theta_series = self.fi.get_inbetweens(self.fi.parse_key_frames(anim_args.perspective_flip_theta))
        self.perspective_flip_phi_series = self.fi.get_inbetweens(self.fi.parse_key_frames(anim_args.perspective_flip_phi))
        self.perspective_flip_gamma_series = self.fi.get_inbetweens(self.fi.parse_key_frames(anim_args.perspective_flip_gamma))
        self.perspective_flip_fv_series = self.fi.get_inbetweens(self.fi.parse_key_frames(anim_args.perspective_flip_fv))
        self.noise_schedule_series = self.fi.get_inbetweens(self.fi.parse_key_frames(anim_args.noise_schedule))
        self.strength_schedule_series = self.fi.get_inbetweens(self.fi.parse_key_frames(anim_args.strength_schedule))
        self.contrast_schedule_series = self.fi.get_inbetweens(self.fi.parse_key_frames(anim_args.contrast_schedule))
        self.cfg_scale_schedule_series = self.fi.get_inbetweens(self.fi.parse_key_frames(anim_args.cfg_scale_schedule))
        self.ddim_eta_schedule_series = self.fi.get_inbetweens(self.fi.parse_key_frames(anim_args.ddim_eta_schedule))
        self.ancestral_eta_schedule_series = self.fi.get_inbetweens(self.fi.parse_key_frames(anim_args.ancestral_eta_schedule))
        self.pix2pix_img_cfg_scale_series = self.fi.get_inbetweens(self.fi.parse_key_frames(anim_args.pix2pix_img_cfg_scale_schedule))
        self.subseed_schedule_series = self.fi.get_inbetweens(self.fi.parse_key_frames(anim_args.subseed_schedule))
        self.subseed_strength_schedule_series = self.fi.get_inbetweens(self.fi.parse_key_frames(anim_args.subseed_strength_schedule))
        self.checkpoint_schedule_series = self.fi.get_inbetweens(self.fi.parse_key_frames(anim_args.checkpoint_schedule), is_single_string = True)
        self.steps_schedule_series = self.fi.get_inbetweens(self.fi.parse_key_frames(anim_args.steps_schedule))
        self.seed_schedule_series = self.fi.get_inbetweens(self.fi.parse_key_frames(anim_args.seed_schedule))
        self.sampler_schedule_series = self.fi.get_inbetweens(self.fi.parse_key_frames(anim_args.sampler_schedule), is_single_string = True)
        self.clipskip_schedule_series = self.fi.get_inbetweens(self.fi.parse_key_frames(anim_args.clipskip_schedule))
        self.noise_multiplier_schedule_series = self.fi.get_inbetweens(self.fi.parse_key_frames(anim_args.noise_multiplier_schedule))
        self.mask_schedule_series = self.fi.get_inbetweens(self.fi.parse_key_frames(anim_args.mask_schedule), is_single_string = True)
        self.noise_mask_schedule_series = self.fi.get_inbetweens(self.fi.parse_key_frames(anim_args.noise_mask_schedule), is_single_string = True)
        self.kernel_schedule_series = self.fi.get_inbetweens(self.fi.parse_key_frames(anim_args.kernel_schedule))
        self.sigma_schedule_series = self.fi.get_inbetweens(self.fi.parse_key_frames(anim_args.sigma_schedule))
        self.amount_schedule_series = self.fi.get_inbetweens(self.fi.parse_key_frames(anim_args.amount_schedule))
        self.threshold_schedule_series = self.fi.get_inbetweens(self.fi.parse_key_frames(anim_args.threshold_schedule))
        self.aspect_ratio_series = self.fi.get_inbetweens(self.fi.parse_key_frames(anim_args.aspect_ratio_schedule))
        self.fov_series = self.fi.get_inbetweens(self.fi.parse_key_frames(anim_args.fov_schedule))
        self.near_series = self.fi.get_inbetweens(self.fi.parse_key_frames(anim_args.near_schedule))
        self.cadence_flow_factor_schedule_series = self.fi.get_inbetweens(self.fi.parse_key_frames(anim_args.cadence_flow_factor_schedule))
        self.redo_flow_factor_schedule_series = self.fi.get_inbetweens(self.fi.parse_key_frames(anim_args.redo_flow_factor_schedule))
        self.far_series = self.fi.get_inbetweens(self.fi.parse_key_frames(anim_args.far_schedule))
        self.hybrid_comp_alpha_schedule_series = self.fi.get_inbetweens(self.fi.parse_key_frames(anim_args.hybrid_comp_alpha_schedule))
        self.hybrid_comp_mask_blend_alpha_schedule_series = self.fi.get_inbetweens(self.fi.parse_key_frames(anim_args.hybrid_comp_mask_blend_alpha_schedule))
        self.hybrid_comp_mask_contrast_schedule_series = self.fi.get_inbetweens(self.fi.parse_key_frames(anim_args.hybrid_comp_mask_contrast_schedule))
        self.hybrid_comp_mask_auto_contrast_cutoff_high_schedule_series = self.fi.get_inbetweens(self.fi.parse_key_frames(anim_args.hybrid_comp_mask_auto_contrast_cutoff_high_schedule))
        self.hybrid_comp_mask_auto_contrast_cutoff_low_schedule_series = self.fi.get_inbetweens(self.fi.parse_key_frames(anim_args.hybrid_comp_mask_auto_contrast_cutoff_low_schedule))
        self.hybrid_flow_factor_schedule_series = self.fi.get_inbetweens(self.fi.parse_key_frames(anim_args.hybrid_flow_factor_schedule))

class ControlNetKeys():
    def __init__(self, anim_args, controlnet_args):
        self.fi = FrameInterpolater(max_frames=anim_args.max_frames)
        self.schedules = {}
        for i in range(1, 6): # 5 CN models in total
            for suffix in ['weight', 'guidance_start', 'guidance_end']:
                prefix = f"cn_{i}"
                key = f"{prefix}_{suffix}_schedule_series"
                self.schedules[key] = self.fi.get_inbetweens(self.fi.parse_key_frames(getattr(controlnet_args, f"{prefix}_{suffix}")))
                setattr(self, key, self.schedules[key])

class LooperAnimKeys():
    def __init__(self, loop_args, anim_args, seed):
        self.fi = FrameInterpolater(anim_args.max_frames, seed)
        self.use_looper = loop_args.use_looper
        self.imagesToKeyframe = loop_args.init_images
        self.image_strength_schedule_series = self.fi.get_inbetweens(self.fi.parse_key_frames(loop_args.image_strength_schedule))
        self.blendFactorMax_series = self.fi.get_inbetweens(self.fi.parse_key_frames(loop_args.blendFactorMax))
        self.blendFactorSlope_series = self.fi.get_inbetweens(self.fi.parse_key_frames(loop_args.blendFactorSlope))
        self.tweening_frames_schedule_series = self.fi.get_inbetweens(self.fi.parse_key_frames(loop_args.tweening_frames_schedule))
        self.color_correction_factor_series = self.fi.get_inbetweens(self.fi.parse_key_frames(loop_args.color_correction_factor))

class FrameInterpolater():
    def __init__(self, max_frames=0, seed=-1) -> None:
        self.max_frames = max_frames
        self.seed = seed

    def sanitize_value(self, value):
        return value.replace("'","").replace('"',"").replace('(',"").replace(')',"")

    def get_inbetweens(self, key_frames, integer=False, interp_method='Linear', is_single_string = False):
        key_frame_series = pd.Series([np.nan for a in range(self.max_frames)])
        # get our ui variables set for numexpr.evaluate
        max_f = self.max_frames -1
        s = self.seed
        for i in range(0, self.max_frames):
            if i in key_frames:
                value = key_frames[i]
                value_is_number = check_is_number(self.sanitize_value(value))
                if value_is_number: # if it's only a number, leave the rest for the default interpolation
                    key_frame_series[i] = self.sanitize_value(value)
            if not value_is_number:
                t = i
                # workaround for values formatted like 0:("I am test") //used for sampler schedules
                key_frame_series[i] = numexpr.evaluate(value) if not is_single_string else self.sanitize_value(value)
            elif is_single_string:# take previous string value and replicate it
                key_frame_series[i] = key_frame_series[i-1]
        key_frame_series = key_frame_series.astype(float) if not is_single_string else key_frame_series # as string

        if interp_method == 'Cubic' and len(key_frames.items()) <= 3:
            interp_method = 'Quadratic'
        if interp_method == 'Quadratic' and len(key_frames.items()) <= 2:
            interp_method = 'Linear'

        key_frame_series[0] = key_frame_series[key_frame_series.first_valid_index()]
        key_frame_series[self.max_frames-1] = key_frame_series[key_frame_series.last_valid_index()]
        key_frame_series = key_frame_series.interpolate(method=interp_method.lower(), limit_direction='both')
        if integer:
            return key_frame_series.astype(int)
        return key_frame_series

    def parse_key_frames(self, string):
        # because math functions (i.e. sin(t)) can utilize brackets 
        # it extracts the value in form of some stuff
        # which has previously been enclosed with brackets and
        # with a comma or end of line existing after the closing one
        frames = dict()
        for match_object in string.split(","):
            frameParam = match_object.split(":")
            max_f = self.max_frames -1
            s = self.seed
            frame = int(self.sanitize_value(frameParam[0])) if check_is_number(self.sanitize_value(frameParam[0].strip())) else int(numexpr.evaluate(frameParam[0].strip().replace("'","",1).replace('"',"",1)[::-1].replace("'","",1).replace('"',"",1)[::-1]))
            frames[frame] = frameParam[1].strip()
        if frames == {} and len(string) != 0:
            raise RuntimeError('Key Frame string not correctly formatted')
        return frames