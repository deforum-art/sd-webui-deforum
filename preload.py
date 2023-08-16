# 'Deforum' plugin for Automatic1111's Stable Diffusion WebUI.
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

def preload(parser):
    parser.add_argument(
        "--deforum-api",
        action="store_true",
        help="Enable the Deforum API",
        default=None,
    )
    parser.add_argument(
        "--deforum-simple-api",
        action="store_true",
        help="Enable the simplified version of Deforum API",
        default=None,
    )
    parser.add_argument(
        "--deforum-run-now",
        type=str,
        help="Comma-delimited list of deforum settings files to run immediately on startup",
        default=None,
    )
    parser.add_argument(
        "--deforum-terminate-after-run-now",
        action="store_true",
        help="Whether to shut down the a1111 process immediately after completing the generations passed in to '--deforum-run-now'.",
        default=None,
    )