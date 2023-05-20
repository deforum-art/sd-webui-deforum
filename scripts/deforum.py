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

from modules import script_callbacks
from scripts.deforum_extend_paths import deforum_sys_extend

def init_deforum():
    deforum_sys_extend()

    from deforum_helpers.ui_right import on_ui_tabs
    script_callbacks.on_ui_tabs(on_ui_tabs)
    from deforum_helpers.ui_settings import on_ui_settings
    script_callbacks.on_ui_settings(on_ui_settings)

init_deforum()
