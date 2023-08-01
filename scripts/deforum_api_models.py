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