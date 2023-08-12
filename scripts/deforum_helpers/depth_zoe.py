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

import torch
from zoedepth.models.builder import build_model
from zoedepth.utils.config import get_config

class ZoeDepth:
    def __init__(self, width=512, height=512):
        conf = get_config("zoedepth_nk", "infer")
        conf.img_size = [width, height]
        self.model_zoe = build_model(conf)
        self.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        self.zoe = self.model_zoe.to(self.DEVICE)
        self.width = width
        self.height = height
        
    def predict(self, image):
        self.zoe.core.prep.resizer._Resize__width = self.width
        self.zoe.core.prep.resizer._Resize__height = self.height
        depth_tensor = self.zoe.infer_pil(image, output_type="tensor")
        return depth_tensor
        
    def to(self, device):
        self.DEVICE = device
        self.zoe = self.model_zoe.to(device)
        
    def save_raw_depth(self, depth, filepath):
        depth.save(filepath, format='PNG', mode='I;16')
    
    def delete(self):
        del self.model_zoe
        del self.zoe