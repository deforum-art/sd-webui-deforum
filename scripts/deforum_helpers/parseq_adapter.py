from operator import itemgetter
import json
import logging
import pandas as pd
import numpy as np
import operator

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)


class ParseqAnimKeys():
    def __init__(self, parseq_args, anim_args):

        self.parseq_json = json.loads(parseq_args.parseq_manifest)
        self.rendered_frames = self.parseq_json['rendered_frames']
        
        self.max_frame = self.get_max('frame')
        count_defined_frames = len(self.rendered_frames)        
        expected_defined_frames = self.max_frame+1 # frames are 0-indexed

        self.required_frames = anim_args.max_frames

        if (expected_defined_frames != count_defined_frames): 
            logging.warning(f"There may be duplicated or missing frame data in the Parseq input: expected {expected_defined_frames} frames including frame 0 because the highest frame number is {self.max_frame}, but there are {count_defined_frames} frames defined.")

        if (anim_args.max_frames > count_defined_frames):
            logging.info(f"Parseq data defines {count_defined_frames} frames, but the requested animation is {anim_args.max_frames} frames. The last Parseq frame definition will be duplicated to match the expected frame count.")
        if (anim_args.max_frames < count_defined_frames):
            logging.info(f"Parseq data defines {count_defined_frames} frames, but the requested animation is {anim_args.max_frames} frames. The last Parseq frame definitions will be ignored.")
        else:
            logging.info(f"Parseq data defines {count_defined_frames} frames.")

        # Parseq treats input values as absolute values. So if you want to 
        # progressively rotate 180 degrees over 4 frames, you specify: 45, 90, 135, 180.
        # However, many animation parameters are relative to the previous frame if there is enough
        # loopback strength. So if you want to rotate 180 degrees over 5 frames, the animation engine expects:
        # 45, 45, 45, 45. Therefore, for such parameter, we use the fact that Parseq supplies delta values.
        optional_delta = '_delta' if parseq_args.parseq_use_deltas else ''
        self.angle_series = self.parseq_to_anim_series('angle' + optional_delta)
        self.zoom_series = self.parseq_to_anim_series('zoom' + optional_delta)        
        self.translation_x_series = self.parseq_to_anim_series('translation_x' + optional_delta)
        self.translation_y_series = self.parseq_to_anim_series('translation_y' + optional_delta)
        self.translation_z_series = self.parseq_to_anim_series('translation_z' + optional_delta)
        self.rotation_3d_x_series = self.parseq_to_anim_series('rotation_3d_x' + optional_delta)
        self.rotation_3d_y_series = self.parseq_to_anim_series('rotation_3d_y' + optional_delta)
        self.rotation_3d_z_series = self.parseq_to_anim_series('rotation_3d_z' + optional_delta)
        self.perspective_flip_theta_series = self.parseq_to_anim_series('perspective_flip_theta' + optional_delta)
        self.perspective_flip_phi_series = self.parseq_to_anim_series('perspective_flip_phi' + optional_delta)
        self.perspective_flip_gamma_series = self.parseq_to_anim_series('perspective_flip_gamma' + optional_delta)
 
        # Non-motion animation args
        self.perspective_flip_fv_series = self.parseq_to_anim_series('perspective_flip_fv')
        self.noise_schedule_series = self.parseq_to_anim_series('noise')
        self.strength_schedule_series = self.parseq_to_anim_series('strength')
        self.contrast_schedule_series = self.parseq_to_anim_series('contrast')
        self.cfg_scale_schedule_series = self.parseq_to_anim_series('scale')
        self.seed_schedule_series = self.parseq_to_anim_series('seed')
        self.fov_series = self.parseq_to_anim_series('fov')
        self.near_series = self.parseq_to_anim_series('near')
        self.far_series = self.parseq_to_anim_series('far')
        self.prompts = self.parseq_to_anim_series('deforum_prompt') # formatted as "{positive} --neg {negative}"
        self.subseed_series = self.parseq_to_anim_series('subseed')
        self.subseed_strength_series = self.parseq_to_anim_series('subseed_strength')

        # Config:
        # TODO this is currently ignored. User must ensure the output FPS set in parseq
        # matches the one set in Deforum to avoid unexpected results.
        self.config_output_fps = self.parseq_json['options']['output_fps']

    def get_max(self, seriesName):
        return max(self.rendered_frames, key=itemgetter(seriesName))[seriesName]

    def parseq_to_anim_series(self, seriesName):
        key_frame_series = pd.Series([np.nan for a in range(self.required_frames)])
        
        for frame in self.rendered_frames:
            frame_idx = frame['frame']
            if frame_idx < self.required_frames:                
                if not np.isnan(key_frame_series[frame_idx]):
                    logging.warning(f"Duplicate frame definition {frame_idx} detected for data {seriesName}. Latest wins.")        
                key_frame_series[frame_idx] = frame[seriesName]

        # If the animation will have more frames than Parseq defines,
        # duplicate final value to match the required frame count.
        while (frame_idx < self.required_frames):
            key_frame_series[frame_idx] = operator.itemgetter(-1)(self.rendered_frames)[seriesName]
            frame_idx += 1

        return key_frame_series
