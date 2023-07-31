import atexit
import json
import random
import tempfile
import traceback
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from datetime import datetime
from typing import Any, Dict, List
from deforum_api_models import Batch, DeforumJobErrorType, DeforumJobStatusCategory, DeforumJobPhase, DeforumJobStatus

import gradio as gr
from deforum_helpers.args import (DeforumAnimArgs, DeforumArgs,
                                  DeforumOutputArgs, LoopArgs, ParseqArgs,
                                  RootArgs, get_component_names)
from fastapi import FastAPI, Response, status

from modules.shared import cmd_opts, opts


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

def run_deforum_batch(batch_id : str, deforum_settings : List[Any] ): 

    print("started run_deforum_batch")

    # Fill args with default values.
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

    # Arg 0 is a UID for the batch
    run_deforum_args[0] = batch_id

    # Setup batch override
    run_deforum_args[prefixed_gradio_args + component_names.index('override_settings_with_file')] = True
    run_deforum_args[prefixed_gradio_args + component_names.index('custom_settings_file')] = deforum_settings

    # Invoke deforum with appropriate args
    from deforum_helpers.run_deforum import run_deforum 
    run_deforum(*run_deforum_args)



#
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

    apiState = ApiState()

    @app.post("/deforum_api/batches")
    async def run_batch(batch: Batch, response: Response):

        # Extract the settings files from the request
        deforum_settings_data = batch.deforum_settings
        if not deforum_settings_data:
            response.status_code = status.HTTP_400_BAD_REQUEST
            return {"message": "No settings files provided. Please provide an element 'deforum_settings' of type list in the request JSON payload."}
        
        if not isinstance(deforum_settings_data, list):
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

        for job_id in job_ids:
            JobStatusTracker().accept_job(batch_id=batch_id, job_id=job_id)

        response.status_code = status.HTTP_202_ACCEPTED
        return {"message": "Job(s) accepted", "batch_id": batch_id, "job_ids": job_ids }


    @app.get("/deforum_api/batches/{id}")
    async def get_batch(id: str):
        jobsForBatch = JobStatusTracker().batches[id]
        return [JobStatusTracker().get(job_id) for job_id in jobsForBatch]

    ## TODO ##
    @app.delete("/deforum_api/batches/{id}")
    async def stop_batch(id: str, response: Response):
        response.status_code = status.HTTP_501_NOT_IMPLEMENTED
        return {"id": id, "status": "NOT IMPLEMENTED"}

    @app.get("/deforum_api/jobs")
    async def list_jobs():
        return JobStatusTracker().statuses

    @app.get("/deforum_api/jobs/{id}")
    async def get_job(id: str):
        return JobStatusTracker().get(id)

    ## TODO ##
    @app.delete("/deforum_api/jobs/{id}")
    async def stop_job(id: str, response: Response):
        response.status_code = status.HTTP_501_NOT_IMPLEMENTED
        return {"id": id, "status": "NOT IMPLEMENTED"}

class Singleton(type):
    _instances = {}
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]

# Maintains persistent state required by API, e.g. thread pook, list of submitted jobs.
class ApiState(metaclass=Singleton):
    deforum_api_executor = ThreadPoolExecutor(max_workers=5)
    submitted_jobs : Dict[str, Any] = {}

    @staticmethod
    def cleanup():
        ApiState().deforum_api_executor.shutdown(wait=False)    

    def submit_job(self, batch_id: str, job_ids: [str], deforum_settings: List[Any], opts_overrides: Dict[str, Any] ):
        def task():
            print("started task")
            try:
                if opts_overrides is not None and len(opts_overrides)>1:
                    original_opts = {k: opts.data[k] for k in opts_overrides.keys() if k in opts.data}
                    print(f"Captured options to override: {original_opts}")                
                    print(f"Overriding with: {opts_overrides}")
                    for k, v in opts_overrides.items():
                        setattr(opts, k, v)                
                    run_deforum_batch(batch_id, deforum_settings)
            except Exception as e:
                print(f"Batch {batch_id} failed: {e}")
                traceback.print_exc()
                # Mark all jobs in this batch as failed
                for job_id in job_ids:
                    JobStatusTracker().fail_job(job_id, 'TERMINAL', {e})
            finally:
                if (original_opts is not None):
                    print(f"Restoring options")
                    for k, v in original_opts.items():
                        setattr(opts, k, v)


        print("submitting task")
        future = self.deforum_api_executor.submit(task)
        self.submitted_jobs[batch_id] = future

atexit.register(ApiState.cleanup)

# Maintains state that tracks status of submitted jobs, 
# so that clients can query job status.
class JobStatusTracker(metaclass=Singleton):
    statuses: Dict[str, DeforumJobStatus] = {}
    batches: Dict[str, List[str]] = {}

    def accept_job(self, batch_id : str, job_id: str):
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
            timestring=None
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

    def get(self, job_id:str):
        return self.statuses[job_id]

def deforum_init_batch(_: gr.Blocks, app: FastAPI):
    settings_files = [open(filename, 'r') for filename in cmd_opts.deforum_run_now.split(",")]
    [batch_id, job_ids] = make_ids(len(settings_files))
    print(f"Starting batch with job(s) {job_ids}...")

    run_deforum_batch(batch_id, settings_files)

    if cmd_opts.deforum_terminate_after_run_now:
        import os
        os._exit(0)

# Setup A1111 initialisation hooks
try:
    import modules.script_callbacks as script_callbacks    
    if cmd_opts.deforum_api:
        script_callbacks.on_app_started(deforum_api)
    if cmd_opts.deforum_run_now:       
        script_callbacks.on_app_started(deforum_init_batch)
except:
    pass
