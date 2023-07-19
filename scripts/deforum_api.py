import json
import tempfile
from typing import Any, Dict, List, Optional, Literal

import gradio as gr
from fastapi import FastAPI, Response, status
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel

from modules.shared import cmd_opts

from datetime import datetime

from deforum_helpers.args import DeforumAnimArgs, DeforumArgs, DeforumOutputArgs, LoopArgs, ParseqArgs, RootArgs, get_component_names

from dataclasses import dataclass, replace
import random

def make_batch_id():
    return f"batch({random.randint(0, 1e9)})"   


def get_default_value(name:str):
    allArgs = RootArgs() | DeforumAnimArgs() | DeforumArgs() | LoopArgs() | ParseqArgs() | DeforumOutputArgs()
    if name in allArgs and isinstance(allArgs[name], dict):
        return allArgs[name].get("value", None)
    elif name in allArgs:
        return allArgs[name]
    else:
        return None

def run_deforum_batch(batch_id, settings_files): 

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

    # Arg 0 is a UID
    run_deforum_args[0] = batch_id

    # Setup batch override
    run_deforum_args[prefixed_gradio_args + component_names.index('override_settings_with_file')] = True
    run_deforum_args[prefixed_gradio_args + component_names.index('custom_settings_file')] = settings_files

    # Invoke deforum with appropriate args
    from deforum_helpers.run_deforum import run_deforum 
    run_deforum(*run_deforum_args)


def deforum_api(_: gr.Blocks, app: FastAPI):

    @app.post("/deforum_api/batch")
    async def run_batch(batch: Batch, response: Response):

        # Extract the settings files from the request
        settings_files_data = batch.settings_files
        settings_files_tempfiles = []
        for data in settings_files_data:
            temp_file = tempfile.NamedTemporaryFile(mode='w+t', delete=False)
            json.dump(data, temp_file)
            temp_file.close()
            settings_files_tempfiles.append(temp_file)
        
        batch_id = make_batch_id()
        await run_in_threadpool(lambda: run_deforum_batch(batch_id, settings_files_tempfiles))

        response.status_code = status.HTTP_202_ACCEPTED
        job_ids = [f"{batch_id}-{i}" for i in range(len(settings_files_tempfiles))]
        return {"message": "Job(s) accepted", "job_ids": job_ids }


    @app.get("/deforum_api/batch")
    async def list_batches():
        return JobStatusTracker().statuses

    @app.get("/deforum_api/batch/{id}")
    async def get_batch(id: str):
        return JobStatusTracker().get(id)

    @app.delete("/deforum_api/batch/{id}")
    async def stop_batch(id: str):
        return {"id": id, "status": "TODO"}


class Batch(BaseModel):
    settings_files : Optional[List[Dict[str, Any]]]

@dataclass(frozen=True)
class DeforumRunStatus(BaseModel):
    id: str
    status : Literal['IN_PROGRESS', 'SUCCESS', 'FAILED']
    phase : Literal['PREPARING', 'GENERATING', 'POST_PROCESSING', 'DONE']
    error_type : Optional[Literal['NONE', 'RETRYABLE', 'TERMINAL']]
    phase_progress : float
    started_at: int
    last_updated: int
    message: Optional[str]

class Singleton(type):
    _instances = {}
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]

class JobStatusTracker(metaclass=Singleton):
    def __init__(self):
        self.statuses: Dict[str, DeforumRunStatus] = {}

    def start_job(self, job_id: str):
        self.statuses[job_id] = DeforumRunStatus(
            id=job_id,
            status='IN_PROGRESS',
            phase='PREPARING',
            error_type='NONE',
            phase_progress=0.0,
            started_at=datetime.now().timestamp(),
            last_updated=datetime.now().timestamp(),
            message=None
        )

    def update_phase(self, job_id: str, phase: str, progress: float = 0):
        if job_id in self.statuses:
            current_status = self.statuses[job_id]
            new_status = replace(
                current_status,
                phase=phase,
                phase_progress=progress,
                last_updated=datetime.now().timestamp(),
            )
            self.statuses[job_id] = new_status

    def complete_job(self, job_id: str):
        if job_id in self.statuses:
            current_status = self.statuses[job_id]
            new_status = replace(
                current_status,
                status='SUCCESS',
                phase='DONE',
                phase_progress=1.0,
                last_updated=datetime.now().timestamp(),
            )
            self.statuses[job_id] = new_status

    def fail_job(self, job_id: str, error_type: str, message: str):
        if job_id in self.statuses:
            current_status = self.statuses[job_id]
            new_status = replace(
                current_status,
                status='FAILED',
                error_type=error_type,
                last_updated=datetime.now().timestamp(),
                message=message
            )
            self.statuses[job_id] = new_status

    def get(self, job_id:str):
        return self.statuses[job_id]

def deforum_init_batch(_: gr.Blocks, app: FastAPI):
    settings_files = [open(filename, 'r') for filename in cmd_opts.deforum_run_now.split(",")]
    batch_id = make_batch_id()
    job_ids = [f"{batch_id}-{i}" for i in range(len(settings_files))]
    print(f"Starting batch with job(s) {job_ids}...")
    run_deforum_batch(batch_id, settings_files)

    if cmd_opts.deforum_terminate_after_run_now:
        import os
        os._exit(0)


try:
    import modules.script_callbacks as script_callbacks    
    if cmd_opts.deforum_api:
        script_callbacks.on_app_started(deforum_api)
    if cmd_opts.deforum_run_now:       
        script_callbacks.on_app_started(deforum_init_batch)

except:
    pass
