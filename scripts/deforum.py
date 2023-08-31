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

import modules.paths as ph
from modules import script_callbacks
from modules.shared import cmd_opts
from scripts.deforum_extend_paths import deforum_sys_extend


def init_deforum():
    # use sys.path.extend to make sure all of our files are available for importation
    deforum_sys_extend()

    # create the Models/Deforum folder, where many of the deforum related models/ packages will be downloaded
    os.makedirs(ph.models_path + '/Deforum', exist_ok=True)

    # import our on_ui_tabs and on_ui_settings functions from the respected files
    from deforum_helpers.ui_right import on_ui_tabs
    from deforum_helpers.ui_settings import on_ui_settings

    # trigger webui's extensions mechanism using our imported main functions -
    # first to create the actual deforum gui, then to make the deforum tab in webui's settings section
    script_callbacks.on_ui_tabs(on_ui_tabs)
    script_callbacks.on_ui_settings(on_ui_settings)

init_deforum()

