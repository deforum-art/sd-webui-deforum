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

import copy
import json
import logging
import operator
from operator import itemgetter
import numpy as np
import pandas as pd
import requests
from .animation_key_frames import DeformAnimKeys, ControlNetKeys, LooperAnimKeys
from .rich import console

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)

IGNORED_FIELDS = ['fi', 'use_looper', 'imagesToKeyframe', 'schedules']

class ParseqAdapter():
    def __init__(self, parseq_args, anim_args, video_args, controlnet_args, loop_args, mute=False):

        # Basic data extraction
        self.use_parseq = parseq_args.parseq_manifest and parseq_args.parseq_manifest.strip()
        self.use_deltas = parseq_args.parseq_use_deltas

        self.parseq_json = self.load_manifest(parseq_args) if self.use_parseq else json.loads('{ "rendered_frames": [{"frame": 0}] }')
        self.rendered_frames = self.parseq_json['rendered_frames']       
        self.max_frame = self.get_max('frame')
        self.required_frames = anim_args.max_frames        

        # Wrap the original schedules with Parseq decorators, so that Parseq values will override the original values IFF appropriate.
        self.anim_keys = ParseqAnimKeysDecorator(self, DeformAnimKeys(anim_args))
        self.cn_keys = ParseqControlNetKeysDecorator(self, ControlNetKeys(anim_args, controlnet_args)) if controlnet_args else None
        # -1 because seed seems to be unused in LooperAnimKeys
        self.looper_keys = ParseqLooperKeysDecorator(self, LooperAnimKeys(loop_args, anim_args, -1)) if loop_args else None

        # Validation
        if (self.use_parseq):
            self.required_fps = video_args.fps
            self.config_output_fps = self.parseq_json['options']['output_fps']
            count_defined_frames = len(self.rendered_frames)
            expected_defined_frames = self.max_frame+1 # frames are 0-indexed
            if (expected_defined_frames != count_defined_frames): 
                logging.warning(f"There may be duplicated or missing frame data in the Parseq input: expected {expected_defined_frames} frames including frame 0 because the highest frame number is {self.max_frame}, but there are {count_defined_frames} frames defined.")
            if not mute:
                self.print_parseq_table()
    
    # Resolve manifest either directly from supplied value or via supplied URL
    def load_manifest(self, parseq_args):
        manifestOrUrl = parseq_args.parseq_manifest.strip()
        if (manifestOrUrl.startswith('http')):
            logging.info(f"Loading Parseq manifest from URL: {manifestOrUrl}")
            try:
                body = requests.get(manifestOrUrl).text
                logging.debug(f"Loaded remote manifest: {body}")
                parseq_json = json.loads(body)
                if not parseq_json or not 'rendered_frames' in parseq_json:
                    raise Exception(f"The JSON data does not look like a Parseq manifest (missing field 'rendered_frames').")

                # SIDE EFFECT!
                # Add the parseq manifest without the detailed frame data to parseq_args.
                # This ensures it will be saved in the settings file, so that you can always
                # see exactly what parseq prompts and keyframes were used, even if what the URL
                # points to changes.
                parseq_args.fetched_parseq_manifest_summary = copy.deepcopy(parseq_json)
                if parseq_args.fetched_parseq_manifest_summary['rendered_frames']:
                    del parseq_args.fetched_parseq_manifest_summary['rendered_frames']
                if parseq_args.fetched_parseq_manifest_summary['rendered_frames_meta']:
                    del parseq_args.fetched_parseq_manifest_summary['rendered_frames_meta']

                return parseq_json

            except Exception as e:
                logging.error(f"Unable to load Parseq manifest from URL: {manifestOrUrl}")
                raise e
        else:
            return json.loads(manifestOrUrl)        

    def print_parseq_table(self):
        from rich.table import Table
        from rich import box
        
        table = Table(padding=0, box=box.ROUNDED, show_lines=True)
        table.add_column("", style="white bold")
        table.add_column("Parseq", style="cyan")
        table.add_column("Deforum", style="green")

        table.add_row("Animation", '\n'.join(self.anim_keys.managed_fields()), '\n'.join(self.anim_keys.unmanaged_fields()))
        if self.cn_keys:
            table.add_row("ControlNet", '\n'.join(self.cn_keys.managed_fields()), '\n'.join(self.cn_keys.unmanaged_fields()))
        if self.looper_keys:
            table.add_row("Guided Images", '\n'.join(self.looper_keys.managed_fields()), '\n'.join(self.looper_keys.unmanaged_fields()))            
        table.add_row("Prompts", "✅" if self.manages_prompts() else "❌", "✅" if not self.manages_prompts() else "❌")
        table.add_row("Frames", str(len(self.rendered_frames)), str(self.required_frames) + (" ⚠️" if str(self.required_frames) != str(len(self.rendered_frames))+"" else ""))
        table.add_row("FPS", str(self.config_output_fps), str(self.required_fps) + (" ⚠️" if str(self.required_fps) != str(self.config_output_fps) else ""))

        console.print("\nUse this table to validate your Parseq & Deforum setup:")
        console.print(table)

    def manages_prompts(self):
        return self.use_parseq and 'deforum_prompt' in self.rendered_frames[0].keys()

    def manages_seed(self):
        return self.use_parseq and 'seed' in self.rendered_frames[0].keys()    
    
    def get_max(self, seriesName):
        return max(self.rendered_frames, key=itemgetter(seriesName))[seriesName]


class ParseqAbstractDecorator():  

    def __init__(self, adapter: ParseqAdapter, fallback_keys):
        self.adapter = adapter
        self.fallback_keys = fallback_keys

    def parseq_to_series(self, seriesName):
        
        # Check if value is present in first frame of JSON data. If not, assume it's undefined.
        # The Parseq contract is that the first frame (at least) must define values for all fields.
        try:
            if self.adapter.rendered_frames[0][seriesName] is not None:
                logging.debug(f"Found {seriesName} in first frame of Parseq data. Assuming it's defined.")
        except KeyError:
            return None

        key_frame_series = pd.Series([np.nan for a in range(self.adapter.required_frames)])
        
        for frame in self.adapter.rendered_frames:
            frame_idx = frame['frame']
            if frame_idx < self.adapter.required_frames:                
                if not np.isnan(key_frame_series[frame_idx]):
                    logging.warning(f"Duplicate frame definition {frame_idx} detected for data {seriesName}. Latest wins.")        
                key_frame_series[frame_idx] = frame[seriesName]

        # If the animation will have more frames than Parseq defines,
        # duplicate final value to match the required frame count.
        while (frame_idx < self.adapter.required_frames):
            key_frame_series[frame_idx] = operator.itemgetter(-1)(self.adapter.rendered_frames)[seriesName]
            frame_idx += 1

        return key_frame_series

    # fallback to anim_args if the series is not defined in the Parseq data
    def __getattribute__(inst, name):
        try:
            definedField = super(ParseqAbstractDecorator, inst).__getattribute__(name)
        except AttributeError:
            # No field with this name has been explicitly extracted from the JSON data.
            # It must be a new parameter. Let's see if it's in the raw JSON.

            parseqName = inst.strip_suffixes(name)
            
            # returns None if not defined in Parseq JSON data
            definedField = inst.parseq_to_series(parseqName)
            if (definedField is not None):
                # add the field to the instance so we don't compute it again.
                setattr(inst, name, definedField)

        if (definedField is not None):
            return definedField
        else:
            logging.debug(f"Data for {name} not defined in Parseq data. Falling back to standard Deforum values.")
            return getattr(inst.fallback_keys, name)

    
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
    
    # parseq prefixes some field names for clarity. These prefixes are not present in the original Deforum names.
    def strip_parseq_prefixes(self, name):
        strippablePrefixes = ['guided_']
        parseqName = name
        while any(parseqName.startswith(prefix) for prefix in strippablePrefixes):
            for prefix in strippablePrefixes:
                if parseqName.startswith(prefix):
                    parseqName = parseqName[len(prefix):]
        return parseqName    
    
    def all_parseq_fields(self):
        return [self.strip_parseq_prefixes(field) for field in self.adapter.rendered_frames[0].keys() if (not field.endswith('_delta') and not field.endswith('_pc'))]

    def managed_fields(self):
        all_parseq_fields = self.all_parseq_fields()
        deforum_fields = [self.strip_suffixes(property) for property, _ in vars(self.fallback_keys).items() if property not in IGNORED_FIELDS and not property.startswith('_')]
        return [field for field in deforum_fields if field in all_parseq_fields]

    def unmanaged_fields(self):
        all_parseq_fields = self.all_parseq_fields()
        deforum_fields = [self.strip_suffixes(property) for property, _ in vars(self.fallback_keys).items() if property not in IGNORED_FIELDS and not property.startswith('_')]
        return [field for field in deforum_fields if field not in all_parseq_fields]


class ParseqControlNetKeysDecorator(ParseqAbstractDecorator):
    def __init__(self, adapter: ParseqAdapter, cn_keys):
        super().__init__(adapter, cn_keys)


class ParseqAnimKeysDecorator(ParseqAbstractDecorator):
    def __init__(self, adapter: ParseqAdapter, anim_keys):
        super().__init__(adapter, anim_keys)

        # Parseq treats input values as absolute values. So if you want to 
        # progressively rotate 180 degrees over 4 frames, you specify: 45, 90, 135, 180.
        # However, many animation parameters are relative to the previous frame if there is enough
        # loopback strength. So if you want to rotate 180 degrees over 5 frames, the animation engine expects:
        # 45, 45, 45, 45. Therefore, for such parameter, we use the fact that Parseq supplies delta values.
        optional_delta = '_delta' if self.adapter.use_deltas else ''
        self.angle_series = super().parseq_to_series('angle' + optional_delta)
        self.zoom_series = super().parseq_to_series('zoom' + optional_delta)        
        self.translation_x_series = super().parseq_to_series('translation_x' + optional_delta)
        self.translation_y_series = super().parseq_to_series('translation_y' + optional_delta)
        self.translation_z_series = super().parseq_to_series('translation_z' + optional_delta)
        self.rotation_3d_x_series = super().parseq_to_series('rotation_3d_x' + optional_delta)
        self.rotation_3d_y_series = super().parseq_to_series('rotation_3d_y' + optional_delta)
        self.rotation_3d_z_series = super().parseq_to_series('rotation_3d_z' + optional_delta)
        self.perspective_flip_theta_series = super().parseq_to_series('perspective_flip_theta' + optional_delta)
        self.perspective_flip_phi_series = super().parseq_to_series('perspective_flip_phi' + optional_delta)
        self.perspective_flip_gamma_series = super().parseq_to_series('perspective_flip_gamma' + optional_delta)
 
        # Non-motion animation args - never use deltas for these.
        self.perspective_flip_fv_series = super().parseq_to_series('perspective_flip_fv')
        self.noise_schedule_series = super().parseq_to_series('noise')
        self.strength_schedule_series = super().parseq_to_series('strength')
        self.sampler_schedule_series = super().parseq_to_series('sampler_schedule')
        self.contrast_schedule_series = super().parseq_to_series('contrast')
        self.cfg_scale_schedule_series = super().parseq_to_series('scale')
        self.steps_schedule_series = super().parseq_to_series("steps_schedule")
        self.seed_schedule_series = super().parseq_to_series('seed')
        self.fov_series = super().parseq_to_series('fov')
        self.near_series = super().parseq_to_series('near')
        self.far_series = super().parseq_to_series('far')
        self.subseed_schedule_series = super().parseq_to_series('subseed')
        self.subseed_strength_schedule_series = super().parseq_to_series('subseed_strength')
        self.kernel_schedule_series = super().parseq_to_series('antiblur_kernel')
        self.sigma_schedule_series = super().parseq_to_series('antiblur_sigma')
        self.amount_schedule_series = super().parseq_to_series('antiblur_amount')
        self.threshold_schedule_series = super().parseq_to_series('antiblur_threshold')

        # TODO - move to a different decorator?
        self.prompts = super().parseq_to_series('deforum_prompt') # formatted as "{positive} --neg {negative}"


class ParseqLooperKeysDecorator(ParseqAbstractDecorator):
    def __init__(self, adapter: ParseqAdapter, looper_keys):
        super().__init__(adapter, looper_keys)

        # The Deforum UI offers an "Image strength schedule" in the Guided Images section,
        # which simply overrides the strength schedule if guided images is enabled.
        # In Parseq, we just re-use the same strength schedule.
        self.image_strength_schedule_series = super().parseq_to_series('strength')

        # We explicitly state the mapping for all other guided images fields so we can strip the prefix
        # that we use in Parseq.
        self.blendFactorMax_series = super().parseq_to_series('guided_blendFactorMax')
        self.blendFactorSlope_series = super().parseq_to_series('guided_blendFactorSlope')
        self.tweening_frames_schedule_series = super().parseq_to_series('guided_tweening_frames')
        self.color_correction_factor_series = super().parseq_to_series('guided_color_correction_factor')

