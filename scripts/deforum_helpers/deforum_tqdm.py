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

import os
from math import ceil
import tqdm
from modules.shared import progress_print_out, opts, cmd_opts

class DeforumTQDM:
    def __init__(self, args, anim_args, parseq_args, video_args):
        self._tqdm = None
        self._args = args
        self._anim_args = anim_args
        self._parseq_args = parseq_args
        self._video_args = video_args

    def reset(self):
        from .animation_key_frames import DeformAnimKeys
        from .parseq_adapter import ParseqAdapter
        deforum_total = 0
        # FIXME: get only amount of steps
        parseq_adapter = ParseqAdapter(self._parseq_args, self._anim_args, self._video_args, None, None, mute=True)
        keys = DeformAnimKeys(self._anim_args) if not parseq_adapter.use_parseq else parseq_adapter.anim_keys

        start_frame = 0
        if self._anim_args.resume_from_timestring:
            for tmp in os.listdir(self._args.outdir):
                filename = tmp.split("_")
                # don't use saved depth maps to count number of frames
                if self._anim_args.resume_timestring in filename and "depth" not in filename:
                    start_frame += 1
            start_frame = start_frame - 1
        using_vid_init = self._anim_args.animation_mode == 'Video Input'
        turbo_steps = 1 if using_vid_init else int(self._anim_args.diffusion_cadence)
        if self._anim_args.resume_from_timestring:
            last_frame = start_frame - 1
            if turbo_steps > 1:
                last_frame -= last_frame % turbo_steps
            if turbo_steps > 1:
                turbo_next_frame_idx = last_frame
                turbo_prev_frame_idx = turbo_next_frame_idx
                start_frame = last_frame + turbo_steps
        frame_idx = start_frame
        had_first = False
        while frame_idx < self._anim_args.max_frames:
            strength = keys.strength_schedule_series[frame_idx]
            if not had_first and self._args.use_init and ((self._args.init_image is not None and self._args.init_image != '') or self._args.init_image_box is not None):
                deforum_total += int(ceil(self._args.steps * (1 - strength)))
                had_first = True
            elif not had_first:
                deforum_total += self._args.steps
                had_first = True
            else:
                deforum_total += int(ceil(self._args.steps * (1 - strength)))

            if turbo_steps > 1:
                frame_idx += turbo_steps
            else:
                frame_idx += 1

        self._tqdm = tqdm.tqdm(
            desc="Deforum progress",
            total=deforum_total,
            position=1,
            file=progress_print_out
        )

    def update(self):
        if not opts.multiple_tqdm or cmd_opts.disable_console_progressbars:
            return
        if self._tqdm is None:
            self.reset()
        self._tqdm.update()

    def updateTotal(self, new_total):
        if not opts.multiple_tqdm or cmd_opts.disable_console_progressbars:
            return
        if self._tqdm is None:
            self.reset()
        self._tqdm.total = new_total

    def clear(self):
        if self._tqdm is not None:
            self._tqdm.close()
            self._tqdm = None
