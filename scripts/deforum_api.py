import json
import tempfile
from typing import Any, Dict, List, Optional

import gradio as gr
from fastapi import FastAPI, Response, status
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel

from modules.shared import cmd_opts

from deforum_helpers.args import DeforumAnimArgs, DeforumArgs, DeforumOutputArgs, LoopArgs, ParseqArgs, RootArgs, get_component_names

def get_default_value(name:str):
    allArgs = RootArgs() | DeforumAnimArgs() | DeforumArgs() | LoopArgs() | ParseqArgs() | DeforumOutputArgs()
    if name in allArgs and isinstance(allArgs[name], dict):
        return allArgs[name].get("value", None)
    elif name in allArgs:
        return allArgs[name]
    else:
        return None

def run_deforum_batch(settings_files): 

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
        
        await run_in_threadpool(lambda: run_deforum_batch(settings_files_tempfiles))

        response.status_code = status.HTTP_202_ACCEPTED
        return {"message": "Job accepted"}


    @app.get("/deforum_api/batch")
    async def list_batches():
        return {"TODO": "TODO"}

    @app.get("/deforum_api/batch/{id}")
    async def get_batch(id: int):
        return {"id": id, "batch": "TODO"}
    
    @app.get("/deforum_api/batch/{id}/status")
    async def get_batch_status(id: int):
        return {"id": id, "status": "TODO"}

    @app.delete("/deforum_api/batch/{id}")
    async def stop_batch(id: int):
        return {"id": id, "status": "TODO"}


class Batch(BaseModel):
    settings_files : Optional[List[Dict[str, Any]]]


def deforum_init_batch(_: gr.Blocks, app: FastAPI):
    settings_files = [open(filename, 'r') for filename in cmd_opts.deforum_run_now.split(",")]
    run_deforum_batch(settings_files)
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
