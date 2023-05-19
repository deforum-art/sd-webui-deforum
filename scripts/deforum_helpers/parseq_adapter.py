import copy
import json
import logging
import operator
from operator import itemgetter
import numpy as np
import pandas as pd
import requests
from .animation_key_frames import DeformAnimKeys
from .rich import console

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)

class ParseqAnimKeys():
    def __init__(self, parseq_args, anim_args, video_args, mute=False):

        # Resolve manifest either directly from supplied value
        # or via supplied URL
        manifestOrUrl = parseq_args.parseq_manifest.strip()
        if (manifestOrUrl.startswith('http')):
            logging.info(f"Loading Parseq manifest from URL: {manifestOrUrl}")
            try:
                body = requests.get(manifestOrUrl).text
                logging.debug(f"Loaded remote manifest: {body}")
                self.parseq_json = json.loads(body)

                # Add the parseq manifest without the detailed frame data to parseq_args.
                # This ensures it will be saved in the settings file, so that you can always
                # see exactly what parseq prompts and keyframes were used, even if what the URL
                # points to changes.
                parseq_args.fetched_parseq_manifest_summary = copy.deepcopy(self.parseq_json)
                if parseq_args.fetched_parseq_manifest_summary['rendered_frames']:
                    del parseq_args.fetched_parseq_manifest_summary['rendered_frames']
                if parseq_args.fetched_parseq_manifest_summary['rendered_frames_meta']:
                    del parseq_args.fetched_parseq_manifest_summary['rendered_frames_meta']

            except Exception as e:
                logging.error(f"Unable to load Parseq manifest from URL: {manifestOrUrl}")
                raise e
        else:
            self.parseq_json = json.loads(manifestOrUrl)

        self.default_anim_keys = DeformAnimKeys(anim_args)
        self.rendered_frames = self.parseq_json['rendered_frames']       
        self.max_frame = self.get_max('frame')
        self.required_frames = anim_args.max_frames
        # TODO these values are currently only used to emit a subtle warning. User must ensure the output FPS set in parseq
        # matches the one set in Deforum to avoid unexpected results.
        # In the future we may wish to override video_args.fps value with the one from parseq.
        self.required_fps = video_args.fps
        self.config_output_fps = self.parseq_json['options']['output_fps']

        if not mute:
            self.print_parseq_table()

        count_defined_frames = len(self.rendered_frames)        
        expected_defined_frames = self.max_frame+1 # frames are 0-indexed
        if (expected_defined_frames != count_defined_frames): 
            logging.warning(f"There may be duplicated or missing frame data in the Parseq input: expected {expected_defined_frames} frames including frame 0 because the highest frame number is {self.max_frame}, but there are {count_defined_frames} frames defined.")

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
        self.sampler_schedule_series = self.parseq_to_anim_series('sampler_schedule')
        self.contrast_schedule_series = self.parseq_to_anim_series('contrast')
        self.cfg_scale_schedule_series = self.parseq_to_anim_series('scale')
        self.steps_schedule_series = self.parseq_to_anim_series("steps_schedule")
        self.seed_schedule_series = self.parseq_to_anim_series('seed')
        self.fov_series = self.parseq_to_anim_series('fov')
        self.near_series = self.parseq_to_anim_series('near')
        self.far_series = self.parseq_to_anim_series('far')
        self.prompts = self.parseq_to_anim_series('deforum_prompt') # formatted as "{positive} --neg {negative}"
        self.subseed_schedule_series = self.parseq_to_anim_series('subseed')
        self.subseed_strength_schedule_series = self.parseq_to_anim_series('subseed_strength')
        self.kernel_schedule_series = self.parseq_to_anim_series('antiblur_kernel')
        self.sigma_schedule_series = self.parseq_to_anim_series('antiblur_sigma')
        self.amount_schedule_series = self.parseq_to_anim_series('antiblur_amount')
        self.threshold_schedule_series = self.parseq_to_anim_series('antiblur_threshold')

    def print_parseq_table(self):
        from rich.table import Table
        from rich import box
        table = Table(padding=0, box=box.ROUNDED, show_lines=True)
        table.add_column("", style="white bold")
        table.add_column("Parseq", style="cyan")
        table.add_column("Deforum", style="green")

        table.add_row("Fields", '\n'.join(self.managed_fields()), '\n'.join(self.unmanaged_fields()))
        table.add_row("Prompts", "✅" if self.manages_prompts() else "❌", "✅" if not self.manages_prompts() else "❌")
        table.add_row("Frames", str(len(self.rendered_frames)), str(self.required_frames) + (" ⚠️" if self.required_frames != len(self.rendered_frames) else ""))
        table.add_row("FPS", str(self.config_output_fps), str(self.required_fps) + (" ⚠️" if self.required_fps != self.config_output_fps else ""))

        console.print("\nUse this table to validate your Parseq & Deforum setup:")
        console.print(table)

    def manages_prompts(self):
        return 'deforum_prompt' in self.rendered_frames[0].keys()
    
    def managed_fields(self):
        return [field for field in self.rendered_frames[0].keys()
                            if (field not in ['frame', 'deforum_prompt']
                                    and not field.endswith('_delta')
                                    and not field.endswith('_pc'))]
    
    def unmanaged_fields(self):
        managed_fields = self.managed_fields()
        all_fields = [self.strip_suffixes(property) for property, _ in vars(self.default_anim_keys).items() if property not in ['fi'] and not property.startswith('_')]
        return [field for field in all_fields if field not in managed_fields]


    def get_max(self, seriesName):
        return max(self.rendered_frames, key=itemgetter(seriesName))[seriesName]

    def parseq_to_anim_series(self, seriesName):
        
        # Check if valus is present in first frame of JSON data. If not, assume it's undefined.
        # The Parseq contract is that the first frame (at least) must define values for all fields.
        try:
            if self.rendered_frames[0][seriesName] is not None:
                logging.debug(f"Found {seriesName} in first frame of Parseq data. Assuming it's defined.")
        except KeyError:
            return None

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

    # fallback to anim_args if the series is not defined in the Parseq data
    def __getattribute__(inst, name):
        try:
            definedField = super(ParseqAnimKeys, inst).__getattribute__(name)
        except AttributeError:
            # No field with this name has been explicitly extracted from the JSON data.
            # It must be a new parameter. Let's see if it's in the raw JSON.

            parseqName = inst.strip_suffixes(name)
            
            # returns None if not defined in Parseq JSON data
            definedField = inst.parseq_to_anim_series(parseqName)
            if (definedField is not None):
                # add the field to the instance so we don't compute it again.
                setattr(inst, name, definedField)

        if (definedField is not None):
            return definedField
        else:
            logging.debug(f"Data for {name} not defined in Parseq data. Falling back to standard Deforum values.")
            return getattr(inst.default_anim_keys, name)

    
    # parseq doesn't use _series, _schedule or _schedule_series suffixes in the
    # JSON data - remove them.        
    def strip_suffixes(self, name):
        strippableSuffixes = ['_series', '_schedule']
        parseqName = name
        while any(parseqName.endswith(suffix) for suffix in strippableSuffixes):
            for suffix in strippableSuffixes:
                if parseqName.endswith(suffix):
                    parseqName = parseqName[:-len(suffix)]
        return parseqName

