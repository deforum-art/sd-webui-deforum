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

from pydantic import BaseModel
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass
from enum import Enum

class Batch(BaseModel):
    deforum_settings : Optional[Union[Dict[str, Any],List[Dict[str, Any]]]]
    options_overrides : Optional[Dict[str, Any]]

class DeforumJobStatusCategory(str, Enum):
    ACCEPTED = "ACCEPTED"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"

class DeforumJobPhase(str, Enum):
    QUEUED = "QUEUED"
    PREPARING = "PREPARING"
    GENERATING = "GENERATING"
    POST_PROCESSING = "POST_PROCESSING"
    DONE = "DONE"

class DeforumJobErrorType(str, Enum):
    NONE = "NONE"
    RETRYABLE = "RETRYABLE"
    TERMINAL = "TERMINAL"

@dataclass(frozen=True)
class DeforumJobStatus(BaseModel):
    id: str
    status : DeforumJobStatusCategory
    phase : DeforumJobPhase
    error_type : DeforumJobErrorType
    phase_progress : float
    started_at: float
    last_updated: float 
    execution_time: float           # time between job start and the last status update
    update_interval_time: float     # time between the last two status updates
    updates: int                    # number of status updates so far
    message: Optional[str]
    outdir: Optional[str]
    timestring: Optional[str]
    deforum_settings : Optional[List[Dict[str, Any]]]
    options_overrides : Optional[Dict[str, Any]]