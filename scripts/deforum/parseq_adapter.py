from operator import itemgetter
import json
import logging
import pandas as pd
import numpy as np

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.DEBUG)


class ParseqAnimKeys():
    def __init__(self, parseq_json_string, deforum_anim=None):

        self.parseq_json = json.loads(parseq_json_string)
        self.rendered_frames = self.parseq_json['rendered_frames']
        
        self.max_frame = self.get_max('frame')
        count_defined_frames = len(self.rendered_frames)        
        expected_defined_frames = self.max_frame+1 # frames are 0-indexed
        if (expected_defined_frames != count_defined_frames): 
            logging.warning(f"There may be duplicated or missing frame data in the Parseq input: expected {expected_defined_frames} framesincluding frame 0 because the highest frame number is {self.max_frame}, but there are {count_defined_frames} frames defined.")
        else:
            logging.info(f"Parseq data defines {count_defined_frames} frames.")

        # Existing deforum animation series
        self.angle_series = self.parseq_to_anim_series('angle')
        self.zoom_series = self.parseq_to_anim_series('zoom')
        self.translation_x_series = self.parseq_to_anim_series('translation_x')
        self.translation_y_series = self.parseq_to_anim_series('translation_y')
        self.translation_z_series = self.parseq_to_anim_series('translation_z')
        self.rotation_3d_x_series = self.parseq_to_anim_series('rotation_3d_x')
        self.rotation_3d_y_series = self.parseq_to_anim_series('rotation_3d_y')
        self.rotation_3d_z_series = self.parseq_to_anim_series('rotation_3d_z')
        self.perspective_flip_theta_series = self.parseq_to_anim_series('perspective_flip_theta')
        self.perspective_flip_phi_series = self.parseq_to_anim_series('perspective_flip_phi')
        self.perspective_flip_gamma_series = self.parseq_to_anim_series('perspective_flip_gamma')
        self.perspective_flip_fv_series = self.parseq_to_anim_series('perspective_flip_fv')
        self.noise_schedule_series = self.parseq_to_anim_series('noise')
        self.strength_schedule_series = self.parseq_to_anim_series('strength')
        self.contrast_schedule_series = self.parseq_to_anim_series('contrast')
        self.cfg_scale_schedule_series = self.parseq_to_anim_series('scale')
        self.seed_schedule_series = self.parseq_to_anim_series('seed')
        self.fov_series = self.parseq_to_anim_series('fov')
        self.near_series = self.parseq_to_anim_series('near')
        self.far_series = self.parseq_to_anim_series('far')

        # Debug
        # self.angle_series = deforum_anim.angle_series
        # self.zoom_series = deforum_anim.zoom_series
        # self.translation_x_series = deforum_anim.translation_x_series
        # self.translation_y_series = deforum_anim.translation_y_series
        # self.translation_z_series = deforum_anim.translation_z_series
        # self.rotation_3d_x_series = deforum_anim.rotation_3d_x_series
        # self.rotation_3d_y_series = deforum_anim.rotation_3d_y_series
        # self.rotation_3d_z_series = deforum_anim.rotation_3d_z_series
        # self.perspective_flip_theta_series = deforum_anim.perspective_flip_theta_series
        # self.perspective_flip_phi_series = deforum_anim.perspective_flip_phi_series
        # self.perspective_flip_gamma_series = deforum_anim.perspective_flip_gamma_series
        # self.perspective_flip_fv_series = deforum_anim.perspective_flip_fv_series
        # self.noise_schedule_series = deforum_anim.noise_schedule_series
        # self.strength_schedule_series = deforum_anim.strength_schedule_series
        # self.contrast_schedule_series = deforum_anim.contrast_schedule_series
        # self.cfg_scale_schedule_series = deforum_anim.cfg_scale_schedule_series
        # self.seed_schedule_series = deforum_anim.seed_schedule_series
        # self.fov_series = deforum_anim.fov_series
        # self.near_series = deforum_anim.near_series
        # self.far_series = deforum_anim.far_series

        # Additional animation series
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
        key_frame_series = pd.Series([np.nan for a in range(self.max_frame+1)])
        
        for frame in self.rendered_frames:
            frame_idx = frame['frame']
            if not np.isnan(key_frame_series[frame_idx]):
                logging.warning(f"Duplicate frame definition {frame_idx} detected for data {seriesName}. Latest wins.")        
            key_frame_series[frame_idx] = frame[seriesName]

        return key_frame_series



# json_file = open("./test_data.json", "r")
# parseq_json_string = json_file.read()
# keys = ParseqAnimKeys(parseq_json_string)
# logging.info(keys.seed_schedule_series)
# logging.info(keys.subseed_series)
# logging.info(keys.subseed_strength_series)
