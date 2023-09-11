# Copyright (C) 2023 Deforum LLC
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, version 3 of the License.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

# Contact the authors: https://deforum.github.io/

import re
import numpy as np
import numexpr
import pandas as pd
from .prompt import check_is_number
from modules import scripts, shared

class DeformAnimKeys():
    def __init__(self, anim_args, seed=-1):
        self.fi = FrameInterpolater(anim_args.max_frames, seed)
        self.angle_series = self.fi.parse_inbetweens(anim_args.angle, 'angle')
        self.transform_center_x_series = self.fi.parse_inbetweens(anim_args.transform_center_x, 'transform_center_x')
        self.transform_center_y_series = self.fi.parse_inbetweens(anim_args.transform_center_y, 'transform_center_y')
        self.zoom_series = self.fi.parse_inbetweens(anim_args.zoom, 'zoom')
        self.translation_x_series = self.fi.parse_inbetweens(anim_args.translation_x, 'translation_x')
        self.translation_y_series = self.fi.parse_inbetweens(anim_args.translation_y, 'translation_y')
        self.translation_z_series = self.fi.parse_inbetweens(anim_args.translation_z, 'translation_z')
        self.rotation_3d_x_series = self.fi.parse_inbetweens(anim_args.rotation_3d_x, 'rotation_3d_x')
        self.rotation_3d_y_series = self.fi.parse_inbetweens(anim_args.rotation_3d_y, 'rotation_3d_y')
        self.rotation_3d_z_series = self.fi.parse_inbetweens(anim_args.rotation_3d_z, 'rotation_3d_z')
        self.perspective_flip_theta_series = self.fi.parse_inbetweens(anim_args.perspective_flip_theta, 'perspective_flip_theta')
        self.perspective_flip_phi_series = self.fi.parse_inbetweens(anim_args.perspective_flip_phi, 'perspective_flip_phi')
        self.perspective_flip_gamma_series = self.fi.parse_inbetweens(anim_args.perspective_flip_gamma, 'perspective_flip_gamma')
        self.perspective_flip_fv_series = self.fi.parse_inbetweens(anim_args.perspective_flip_fv, 'perspective_flip_fv')
        self.noise_schedule_series = self.fi.parse_inbetweens(anim_args.noise_schedule, 'noise_schedule')
        self.strength_schedule_series = self.fi.parse_inbetweens(anim_args.strength_schedule, 'strength_schedule')
        self.contrast_schedule_series = self.fi.parse_inbetweens(anim_args.contrast_schedule, 'contrast_schedule')
        self.cfg_scale_schedule_series = self.fi.parse_inbetweens(anim_args.cfg_scale_schedule, 'cfg_scale_schedule')
        self.ddim_eta_schedule_series = self.fi.parse_inbetweens(anim_args.ddim_eta_schedule, 'ddim_eta_schedule')
        self.ancestral_eta_schedule_series = self.fi.parse_inbetweens(anim_args.ancestral_eta_schedule, 'ancestral_eta_schedule')
        self.pix2pix_img_cfg_scale_series = self.fi.parse_inbetweens(anim_args.pix2pix_img_cfg_scale_schedule, 'pix2pix_img_cfg_scale_schedule')
        self.subseed_schedule_series = self.fi.parse_inbetweens(anim_args.subseed_schedule, 'subseed_schedule')
        self.subseed_strength_schedule_series = self.fi.parse_inbetweens(anim_args.subseed_strength_schedule, 'subseed_strength_schedule')
        self.checkpoint_schedule_series = self.fi.parse_inbetweens(anim_args.checkpoint_schedule, 'checkpoint_schedule', is_single_string = True)
        self.steps_schedule_series = self.fi.parse_inbetweens(anim_args.steps_schedule, 'steps_schedule')
        self.seed_schedule_series = self.fi.parse_inbetweens(anim_args.seed_schedule, 'seed_schedule')
        self.sampler_schedule_series = self.fi.parse_inbetweens(anim_args.sampler_schedule, 'sampler_schedule', is_single_string = True)
        self.clipskip_schedule_series = self.fi.parse_inbetweens(anim_args.clipskip_schedule, 'clipskip_schedule')
        self.noise_multiplier_schedule_series = self.fi.parse_inbetweens(anim_args.noise_multiplier_schedule, 'noise_multiplier_schedule')
        self.mask_schedule_series = self.fi.parse_inbetweens(anim_args.mask_schedule, 'mask_schedule', is_single_string = True)
        self.noise_mask_schedule_series = self.fi.parse_inbetweens(anim_args.noise_mask_schedule, 'noise_mask_schedule', is_single_string = True)
        self.kernel_schedule_series = self.fi.parse_inbetweens(anim_args.kernel_schedule, 'kernel_schedule')
        self.sigma_schedule_series = self.fi.parse_inbetweens(anim_args.sigma_schedule, 'sigma_schedule')
        self.amount_schedule_series = self.fi.parse_inbetweens(anim_args.amount_schedule, 'amount_schedule')
        self.threshold_schedule_series = self.fi.parse_inbetweens(anim_args.threshold_schedule, 'threshold_schedule')
        self.aspect_ratio_series = self.fi.parse_inbetweens(anim_args.aspect_ratio_schedule, 'aspect_ratio_schedule')
        self.fov_series = self.fi.parse_inbetweens(anim_args.fov_schedule, 'fov_schedule')
        self.near_series = self.fi.parse_inbetweens(anim_args.near_schedule, 'near_schedule')
        self.cadence_flow_factor_schedule_series = self.fi.parse_inbetweens(anim_args.cadence_flow_factor_schedule, 'cadence_flow_factor_schedule')
        self.redo_flow_factor_schedule_series = self.fi.parse_inbetweens(anim_args.redo_flow_factor_schedule, 'redo_flow_factor_schedule')
        self.far_series = self.fi.parse_inbetweens(anim_args.far_schedule, 'far_schedule')
        self.hybrid_comp_alpha_schedule_series = self.fi.parse_inbetweens(anim_args.hybrid_comp_alpha_schedule, 'hybrid_comp_alpha_schedule')
        self.hybrid_comp_mask_blend_alpha_schedule_series = self.fi.parse_inbetweens(anim_args.hybrid_comp_mask_blend_alpha_schedule, 'hybrid_comp_mask_blend_alpha_schedule')
        self.hybrid_comp_mask_contrast_schedule_series = self.fi.parse_inbetweens(anim_args.hybrid_comp_mask_contrast_schedule, 'hybrid_comp_mask_contrast_schedule')
        self.hybrid_comp_mask_auto_contrast_cutoff_high_schedule_series = self.fi.parse_inbetweens(anim_args.hybrid_comp_mask_auto_contrast_cutoff_high_schedule, 'hybrid_comp_mask_auto_contrast_cutoff_high_schedule')
        self.hybrid_comp_mask_auto_contrast_cutoff_low_schedule_series = self.fi.parse_inbetweens(anim_args.hybrid_comp_mask_auto_contrast_cutoff_low_schedule, 'hybrid_comp_mask_auto_contrast_cutoff_low_schedule')
        self.hybrid_flow_factor_schedule_series = self.fi.parse_inbetweens(anim_args.hybrid_flow_factor_schedule, 'hybrid_flow_factor_schedule')

class ControlNetKeys():
    def __init__(self, anim_args, controlnet_args):
        self.fi = FrameInterpolater(max_frames=anim_args.max_frames)
        self.schedules = {}
        max_models = shared.opts.data.get("control_net_unit_count", shared.opts.data.get("control_net_max_models_num", 5))
        num_of_models = 5
        num_of_models = num_of_models if max_models <= 5 else max_models
        for i in range(1, num_of_models + 1):
            for suffix in ['weight', 'guidance_start', 'guidance_end']:
                prefix = f"cn_{i}"
                input_key = f"{prefix}_{suffix}"
                output_key = f"{input_key}_schedule_series"
                self.schedules[output_key] = self.fi.parse_inbetweens(getattr(controlnet_args, input_key), input_key)
                setattr(self, output_key, self.schedules[output_key])

class LooperAnimKeys():
    def __init__(self, loop_args, anim_args, seed):
        self.fi = FrameInterpolater(anim_args.max_frames, seed)
        self.use_looper = loop_args.use_looper
        self.imagesToKeyframe = loop_args.init_images
        self.image_strength_schedule_series = self.fi.parse_inbetweens(loop_args.image_strength_schedule, 'image_strength_schedule')
        self.blendFactorMax_series = self.fi.parse_inbetweens(loop_args.blendFactorMax, 'blendFactorMax')
        self.blendFactorSlope_series = self.fi.parse_inbetweens(loop_args.blendFactorSlope, 'blendFactorSlope')
        self.tweening_frames_schedule_series = self.fi.parse_inbetweens(loop_args.tweening_frames_schedule, 'tweening_frames_schedule')
        self.color_correction_factor_series = self.fi.parse_inbetweens(loop_args.color_correction_factor, 'color_correction_factor')

class FrameInterpolater():
    def __init__(self, max_frames=0, seed=-1) -> None:
        self.max_frames = max_frames
        self.seed = seed

    def parse_inbetweens(self, value, filename = 'unknown', is_single_string = False):
        return self.get_inbetweens(self.parse_key_frames(value, filename = filename), filename = filename, is_single_string = is_single_string)

    def sanitize_value(self, value):
        return value.replace("'","").replace('"',"").replace('(',"").replace(')',"")

    def get_inbetweens(self, key_frames, integer=False, interp_method='Linear', is_single_string = False, filename = 'unknown'):
        key_frame_series = pd.Series([np.nan for a in range(self.max_frames)])
        # get our ui variables set for numexpr.evaluate
        max_f = self.max_frames -1
        s = self.seed
        for i in range(0, self.max_frames):
            if i in key_frames:
                value = key_frames[i]
                sanitized_value = self.sanitize_value(value)
                value_is_number = check_is_number(sanitized_value)
                if value_is_number: # if it's only a number, leave the rest for the default interpolation
                    key_frame_series[i] = sanitized_value
            if not value_is_number:
                t = i
                # workaround for values formatted like 0:("I am test") //used for sampler schedules
                try:
                    key_frame_series[i] = numexpr.evaluate(value) if not is_single_string else sanitized_value
                except SyntaxError as e:
                    e.filename = f"{filename}@frame#{i}"
                    raise e
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

    def parse_key_frames(self, string, filename='unknown'):
        # because math functions (i.e. sin(t)) can utilize brackets 
        # it extracts the value in form of some stuff
        # which has previously been enclosed with brackets and
        # with a comma or end of line existing after the closing one
        frames = dict()
        for match_object in string.split(","):
            frameParam = match_object.split(":")
            max_f = self.max_frames -1
            s = self.seed
            try:
                frame = int(self.sanitize_value(frameParam[0])) if check_is_number(self.sanitize_value(frameParam[0].strip())) else int(numexpr.evaluate(frameParam[0].strip().replace("'","",1).replace('"',"",1)[::-1].replace("'","",1).replace('"',"",1)[::-1]))
                frames[frame] = frameParam[1].strip()
            except SyntaxError as e:
                e.filename = filename
                raise e
        if frames == {} and len(string) != 0:
            raise RuntimeError('Key Frame string not correctly formatted')
        return frames