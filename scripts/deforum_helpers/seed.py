# 'Deforum' plugin for Automatic1111's Stable Diffusion WebUI.
# Copyright (C) 2023 Artem Khrapov (kabachuha) and Deforum team listed in AUTHORS.md
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, version 3 of the License.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

# Contact the author (Artem Khrapov): https://github.com/kabachuha/

import random

def next_seed(args):
    if args.seed_behavior == 'iter':
        args.seed += 1 if args.seed_internal % args.seed_iter_N == 0 else 0
        args.seed_internal += 1
    elif args.seed_behavior == 'ladder':
        args.seed += 2 if args.seed_internal == 0 else -1
        args.seed_internal = 1 if args.seed_internal == 0 else 0
    elif args.seed_behavior == 'alternate':
        args.seed += 1 if args.seed_internal == 0 else -1
        args.seed_internal = 1 if args.seed_internal == 0 else 0
    elif args.seed_behavior == 'fixed':
        pass  # always keep seed the same
    else:
        args.seed = random.randint(0, 2**32 - 1)
    return args.seed