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

import cv2
import pkg_resources
from skimage.exposure import match_histograms

def maintain_colors(prev_img, color_match_sample, mode):
    
    match_histograms_kwargs = {'channel_axis': -1}
    
    if mode == 'RGB':
        return match_histograms(prev_img, color_match_sample, **match_histograms_kwargs)
    elif mode == 'HSV':
        prev_img_hsv = cv2.cvtColor(prev_img, cv2.COLOR_RGB2HSV)
        color_match_hsv = cv2.cvtColor(color_match_sample, cv2.COLOR_RGB2HSV)
        matched_hsv = match_histograms(prev_img_hsv, color_match_hsv, **match_histograms_kwargs)
        return cv2.cvtColor(matched_hsv, cv2.COLOR_HSV2RGB)
    else: # LAB
        prev_img_lab = cv2.cvtColor(prev_img, cv2.COLOR_RGB2LAB)
        color_match_lab = cv2.cvtColor(color_match_sample, cv2.COLOR_RGB2LAB)
        matched_lab = match_histograms(prev_img_lab, color_match_lab, **match_histograms_kwargs)
        return cv2.cvtColor(matched_lab, cv2.COLOR_LAB2RGB)
