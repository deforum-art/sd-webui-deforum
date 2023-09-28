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
import atexit
import json
import random
import tempfile
import traceback
import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from datetime import datetime
from typing import Any, Dict, List
from deforum_api_models import Batch, DeforumJobErrorType, DeforumJobStatusCategory, DeforumJobPhase, DeforumJobStatus
from contextlib import contextmanager
from deforum_extend_paths import deforum_sys_extend

import gradio as gr
from deforum_helpers.args import (DeforumAnimArgs, DeforumArgs,
                                  DeforumOutputArgs, LoopArgs, ParseqArgs,
                                  RootArgs, get_component_names)
from deforum_helpers.opts_overrider import A1111OptionsOverrider
from fastapi import FastAPI, Response, status

from modules.shared import cmd_opts, opts, state


log = logging.getLogger(__name__)
log_level = os.environ.get("DEFORUM_API_LOG_LEVEL") or os.environ.get("SD_WEBUI_LOG_LEVEL") or "INFO"
log.setLevel(log_level)
logging.basicConfig(
    format='%(asctime)s %(levelname)s [%(name)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)

def make_ids(job_count: int):
    batch_id = f"batch({random.randint(0, 1e9)})"
    job_ids = [f"{batch_id}-{i}" for i in range(job_count)]
    return [batch_id, job_ids]


def get_default_value(name:str):
    allArgs = RootArgs() | DeforumAnimArgs() | DeforumArgs() | LoopArgs() | ParseqArgs() | DeforumOutputArgs()
    if name in allArgs and isinstance(allArgs[name], dict):
        return allArgs[name].get("value", None)
    elif name in allArgs:
        return allArgs[name]
    else:
        return None


def run_deforum_batch(batch_id: str, job_ids: [str], deforum_settings_files: List[Any], opts_overrides: Dict[str, Any] = None):
    log.info(f"Starting batch {batch_id} in thread {threading.get_ident()}.")
    try:
        with A1111OptionsOverrider(opts_overrides):

            # Fill deforum args with default values.
            # We are overriding everything with the batch files, but some values are eagerly validated, so must appear valid.
            component_names = get_component_names()
            prefixed_gradio_args = 2
            expected_arg_count = prefixed_gradio_args + len(component_names)
            run_deforum_args = [None] * expected_arg_count
            for idx, name in enumerate(component_names):
                run_deforum_args[prefixed_gradio_args + idx] = get_default_value(name)

            # For some values, defaults don't pass validation...
            run_deforum_args[prefixed_gradio_args + component_names.index('animation_prompts')] = '{"0":"dummy value"}'
            run_deforum_args[prefixed_gradio_args + component_names.index('animation_prompts_negative')] = ''
            run_deforum_args[prefixed_gradio_args + component_names.index('animation_prompts_positive')] = ''

            # Arg 0 is a UID for the batch
            run_deforum_args[0] = batch_id

            # Setup batch override
            run_deforum_args[prefixed_gradio_args + component_names.index('override_settings_with_file')] = True
            run_deforum_args[prefixed_gradio_args + component_names.index('custom_settings_file')] = deforum_settings_files

            # Cleanup old state from previously cancelled jobs
            # WARNING: not thread safe because state is global. If we ever run multiple batches in parallel, this will need to be reworked.
            state.skipped = False
            state.interrupted = False

            # Invoke deforum with appropriate args
            from deforum_helpers.run_deforum import run_deforum 
            run_deforum(*run_deforum_args)

    except Exception as e:
        log.error(f"Batch {batch_id} failed: {e}")
        traceback.print_exc()
        for job_id in job_ids:
            # Mark all jobs in this batch as failed
            JobStatusTracker().fail_job(job_id, 'TERMINAL', {e})


# API to allow a batch of jobs to be submitted to the deforum pipeline.
# A batch is settings object OR a list of settings objects. 
# A settings object is the JSON structure you can find in your saved settings.txt files.
# 
# Request format:
# {
#   "deforum_settings": [
#       { ... settings object ... },
#       { ... settings object ... },
#   ]
# }
# OR:
# {
#   "deforum_settings": { ... settings object ... }
# }
#
# Each settings object in the request represents a job to run as part of the batch.
# Each submitted batch will be given a batch ID which the user can use to query the status of all jobs in the batch.
#
def deforum_api(_: gr.Blocks, app: FastAPI):  

    deforum_sys_extend()

    apiState = ApiState()

    # Submit a new batch
    @app.post("/deforum_api/batches")
    async def run_batch(batch: Batch, response: Response):

        # Extract the settings files from the request
        deforum_settings_data = batch.deforum_settings
        if not deforum_settings_data:
            response.status_code = status.HTTP_400_BAD_REQUEST
            return {"message": "No settings files provided. Please provide an element 'deforum_settings' of type list in the request JSON payload."}
        
        if not isinstance(deforum_settings_data, list):
            # Allow input deforum_settings to be top-level object as well as single object list
            deforum_settings_data = [deforum_settings_data]

        deforum_settings_tempfiles = []
        for data in deforum_settings_data:
            temp_file = tempfile.NamedTemporaryFile(mode='w+t', delete=False)
            json.dump(data, temp_file)
            temp_file.close()
            deforum_settings_tempfiles.append(temp_file)
        
        job_count = len(deforum_settings_tempfiles)
        [batch_id, job_ids] = make_ids(job_count)
        apiState.submit_job(batch_id, job_ids, deforum_settings_tempfiles, batch.options_overrides)

        for idx, job_id in enumerate(job_ids):
            JobStatusTracker().accept_job(batch_id=batch_id, job_id=job_id, deforum_settings=deforum_settings_data[idx], options_overrides=batch.options_overrides)

        response.status_code = status.HTTP_202_ACCEPTED
        return {"message": "Job(s) accepted", "batch_id": batch_id, "job_ids": job_ids }

    # List all batches and theit job ids
    @app.get("/deforum_api/batches")
    async def list_batches(id: str):
        return JobStatusTracker().batches

    # Show the details of all jobs in a batch
    @app.get("/deforum_api/batches/{id}")
    async def get_batch(id: str, response: Response):
        jobsForBatch = JobStatusTracker().batches[id]
        if not jobsForBatch:
            response.status_code = status.HTTP_404_NOT_FOUND
            return {"id": id, "status": "NOT FOUND"}
        return [JobStatusTracker().get(job_id) for job_id in jobsForBatch]

    # Cancel all jobs in a batch
    @app.delete("/deforum_api/batches/{id}")
    async def cancel_batch(id: str, response: Response):
        jobsForBatch = JobStatusTracker().batches[id]
        cancelled_jobs = []
        if not jobsForBatch:
            response.status_code = status.HTTP_404_NOT_FOUND
            return {"id": id, "status": "NOT FOUND"}
        for job_id in jobsForBatch:
            try:
                cancelled = _cancel_job(job_id)
                if cancelled:
                    cancelled_jobs.append(job_id)
            except:
                log.warning(f"Failed to cancel job {job_id}")
       
        return {"ids": cancelled_jobs, "message:": f"{len(cancelled_jobs)} job(s) cancelled." }

    # Show details of all jobs across al batches
    @app.get("/deforum_api/jobs")
    async def list_jobs():
        return JobStatusTracker().statuses

    # Show details of a single job
    @app.get("/deforum_api/jobs/{id}")
    async def get_job(id: str, response: Response):
        jobStatus = JobStatusTracker().get(id)
        if not jobStatus:
            response.status_code = status.HTTP_404_NOT_FOUND
            return {"id": id, "status": "NOT FOUND"}
        return jobStatus

    # Cancel a single job
    @app.delete("/deforum_api/jobs/{id}")
    async def cancel_job(id: str, response: Response):
        try: 
            if _cancel_job(id):
                return {"id": id, "message": "Job cancelled."}
            else:
                response.status_code = status.HTTP_400_BAD_REQUEST
                return {"id": id, "message": f"Job with ID {id} not in a cancellable state. Has it already finished?"}
        except FileNotFoundError as e:
            response.status_code = status.HTTP_404_NOT_FOUND
            return {"id": id, "message": f"Job with ID {id} not found."}

    # Shared logic for job cancellation
    def _cancel_job(job_id:str):
        jobStatus = JobStatusTracker().get(job_id)
        if not jobStatus:
            raise FileNotFoundError(f"Job {job_id} not found.")
        
        if jobStatus.status != DeforumJobStatusCategory.ACCEPTED:
             # Ignore jobs in completed state (error or success)
            return False

        if job_id in ApiState().submitted_jobs:
            # Remove job from queue
            ApiState().submitted_jobs[job_id].cancel()
        if jobStatus.phase != DeforumJobPhase.QUEUED and jobStatus.phase != DeforumJobPhase.DONE:
            # Job must be actively running - interrupt it.
            # WARNING:
            #   - Possible race condition: if job_id just finished after the check and another started, we'll interrupt the wrong job.
            #   - Not thread safe because State object is global. Will break with concurrent jobs.
            state.interrupt()
        JobStatusTracker().cancel_job(job_id, "Cancelled due to user request.")
        return True
    
class Singleton(type):
    _instances = {}
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]

# Maintains persistent state required by API, e.g. thread pook, list of submitted jobs.
class ApiState(metaclass=Singleton):
    
    ## Locking concurrency to 1. Concurrent generation does seem to work, but it's not clear if it's safe.
    ## TODO: more experimentation required.
    deforum_api_executor = ThreadPoolExecutor(max_workers=1)
    submitted_jobs : Dict[str, Any] = {}

    @staticmethod
    def cleanup():
        ApiState().deforum_api_executor.shutdown(wait=False)    

    def submit_job(self, batch_id: str, job_ids: [str], deforum_settings: List[Any], opts_overrides: Dict[str, Any]):
        log.debug(f"Submitting batch {batch_id} to threadpool.")
        future = self.deforum_api_executor.submit(lambda: run_deforum_batch(batch_id, job_ids, deforum_settings, opts_overrides))
        self.submitted_jobs[batch_id] = future

atexit.register(ApiState.cleanup)

# Maintains state that tracks status of submitted jobs, 
# so that clients can query job status.
class JobStatusTracker(metaclass=Singleton):
    statuses: Dict[str, DeforumJobStatus] = {}
    batches: Dict[str, List[str]] = {}

    def accept_job(self, batch_id : str, job_id: str, deforum_settings : List[Dict[str, Any]] , options_overrides : Dict[str, Any]):
        if batch_id in self.batches:
            self.batches[batch_id].append(job_id)
        else:
            self.batches[batch_id] = [job_id]

        now = datetime.now().timestamp()
        self.statuses[job_id] = DeforumJobStatus(
            id=job_id,
            status= DeforumJobStatusCategory.ACCEPTED,
            phase=DeforumJobPhase.QUEUED,
            error_type=DeforumJobErrorType.NONE,
            phase_progress=0.0,
            started_at=now,
            last_updated=now,
            execution_time=0,
            update_interval_time=0,
            updates=0,
            message=None,
            outdir=None,
            timestring=None,
            deforum_settings=deforum_settings,
            options_overrides=options_overrides,
        )

    def update_phase(self, job_id: str, phase: DeforumJobPhase, progress: float = 0):
        if job_id in self.statuses:
            current_status = self.statuses[job_id]
            now = datetime.now().timestamp()
            new_status = replace(
                current_status,
                phase=phase,
                phase_progress=progress,
                last_updated=now,
                execution_time=now-current_status.started_at,
                update_interval_time=now-current_status.last_updated,
                updates=current_status.updates+1
            )
            self.statuses[job_id] = new_status

    def update_output_info(self, job_id: str, outdir: str, timestring: str):
        if job_id in self.statuses:
            current_status = self.statuses[job_id]
            now = datetime.now().timestamp()
            new_status = replace(
                current_status,
                outdir=outdir,
                timestring=timestring,
                last_updated=now,
                execution_time=now-current_status.started_at,
                update_interval_time=now-current_status.last_updated,
                updates=current_status.updates+1
            )
            self.statuses[job_id] = new_status

    def complete_job(self, job_id: str):
        if job_id in self.statuses:
            current_status = self.statuses[job_id]
            now = datetime.now().timestamp()
            new_status = replace(
                current_status,
                status=DeforumJobStatusCategory.SUCCEEDED,
                phase=DeforumJobPhase.DONE,
                phase_progress=1.0,
                last_updated=now,
                execution_time=now-current_status.started_at,
                update_interval_time=now-current_status.last_updated,
                updates=current_status.updates+1
            )
            self.statuses[job_id] = new_status

    def fail_job(self, job_id: str, error_type: str, message: str):
        if job_id in self.statuses:
            current_status = self.statuses[job_id]
            now = datetime.now().timestamp()
            new_status = replace(
                current_status,
                status=DeforumJobStatusCategory.FAILED,
                error_type=error_type,
                message=message,
                last_updated=now,
                execution_time=now-current_status.started_at,
                update_interval_time=now-current_status.last_updated,
                updates=current_status.updates+1
            )
            self.statuses[job_id] = new_status

    def cancel_job(self, job_id: str, message: str):
        if job_id in self.statuses:
            current_status = self.statuses[job_id]
            now = datetime.now().timestamp()
            new_status = replace(
                current_status,
                status=DeforumJobStatusCategory.CANCELLED,
                message=message,
                last_updated=now,
                execution_time=now-current_status.started_at,
                update_interval_time=now-current_status.last_updated,
                updates=current_status.updates+1
            )
            self.statuses[job_id] = new_status


    def get(self, job_id:str):
        return self.statuses[job_id] if job_id in self.statuses else None

def deforum_init_batch(_: gr.Blocks, app: FastAPI):
    deforum_sys_extend()
    settings_files = [open(filename, 'r') for filename in cmd_opts.deforum_run_now.split(",")]
    [batch_id, job_ids] = make_ids(len(settings_files))
    log.info(f"Starting init batch {batch_id} with job(s) {job_ids}...")

    run_deforum_batch(batch_id, job_ids, settings_files, None)

    if cmd_opts.deforum_terminate_after_run_now:
        import os
        os._exit(0)

# A simplified, but safe version of Deforum's API
def deforum_simple_api(_: gr.Blocks, app: FastAPI):
    deforum_sys_extend()
    from fastapi.exceptions import RequestValidationError
    from fastapi.responses import JSONResponse
    from fastapi import FastAPI, Query, Request, UploadFile
    from fastapi.encoders import jsonable_encoder
    from deforum_helpers.general_utils import get_deforum_version
    import uuid, pathlib

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        return JSONResponse(
            status_code=422,
            content=jsonable_encoder({"detail": exc.errors(), "body": exc.body}),
        )

    @app.get("/deforum/api_version")
    async def deforum_api_version():
        return JSONResponse(content={"version": '1.0'})
    
    @app.get("/deforum/version")
    async def deforum_version():
        return JSONResponse(content={"version": get_deforum_version()})
    
    @app.post("/deforum/run")
    async def deforum_run(settings_json:str, allowed_params:str = ""):
        try:
            allowed_params = allowed_params.split(';')
            deforum_settings = json.loads(settings_json)
            with open(os.path.join(pathlib.Path(__file__).parent.absolute(), 'default_settings.txt'), 'r', encoding='utf-8') as f:
                default_settings = json.loads(f.read())
            for k, _ in default_settings.items():
                if k in deforum_settings and k in allowed_params:
                    default_settings[k] = deforum_settings[k]
            deforum_settings = default_settings
            run_id = uuid.uuid4().hex
            deforum_settings['batch_name'] = run_id
            deforum_settings = json.dumps(deforum_settings, indent=4, ensure_ascii=False)
            settings_file = f"{run_id}.txt"
            with open(settings_file, 'w', encoding='utf-8') as f:
                f.write(deforum_settings)
            class SettingsWrapper:
                def __init__(self, filename):
                    self.name = filename
            [batch_id, job_ids] = make_ids(1)
            outdir = os.path.join(os.getcwd(), opts.outdir_samples or opts.outdir_img2img_samples, str(run_id))
            run_deforum_batch(batch_id, job_ids, [SettingsWrapper(settings_file)], None)
            return JSONResponse(content={"outdir": outdir})
        except Exception as e:
            print(e)
            traceback.print_exc()
            return JSONResponse(status_code=500, content={"detail": "An error occurred while processing the video."},)

# Setup A1111 initialisation hooks
try:
    import modules.script_callbacks as script_callbacks    
    if cmd_opts.deforum_api:
        script_callbacks.on_app_started(deforum_api)
    if cmd_opts.deforum_simple_api:
        script_callbacks.on_app_started(deforum_simple_api)
    if cmd_opts.deforum_run_now:       
        script_callbacks.on_app_started(deforum_init_batch)
except:
    pass
