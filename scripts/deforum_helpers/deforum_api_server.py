import contextlib
import time
import threading
import uvicorn
from fastapi import APIRouter, FastAPI, Response, status
import uvicorn
from uvicorn.config import Config
import asyncio
from modules.shared import cmd_opts
from modules.api.api import api_middleware 
from .args import get_component_names, RootArgs, DeforumAnimArgs, DeforumArgs, LoopArgs, ParseqArgs, DeforumOutputArgs

import tempfile

from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import json


def get_default_value(name:str):
    allArgs = RootArgs() | DeforumAnimArgs() | DeforumArgs() | LoopArgs() | ParseqArgs() | DeforumOutputArgs()
    if name in allArgs and isinstance(allArgs[name], dict):
        return allArgs[name].get("value", None)
    elif name in allArgs:
        return allArgs[name]
    else:
        return None


class DeforumApiServer():

    def __init__(self) -> None:
        self.app = FastAPI()
        api_middleware(self.app) # re-use middleware config from webui api
        self.setup_resources(self.app)
        
        config = Config(self.app, host="0.0.0.0" if cmd_opts.listen else "127.0.0.1",
                port=7864, # TODO Make configurable
                timeout_keep_alive=0,
                log_level="debug")                 
        self.server = uvicorn.Server(config)
        self.thread = threading.Thread(daemon=True, target=self.server.run)

    def start(self):
        self.thread.start()
        #TODO may need this for win?
        # asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())        
        self.wait_for_started()

    async def wait_for_started(self):
        while not self.server.started:
            await asyncio.sleep(0.1)

    def stop(self):
        if self.thread.is_alive():
            self.server.should_exit = True
            while self.thread.is_alive():
                continue            


    def setup_resources(self, app):

        @app.post("/deforum_api/batch")
        async def run_batch(batch: self.Batch):

            # Extract the settings files from the request
            settings_files_data = batch.settings_files
            settings_files_tempfiles = []
            for data in settings_files_data:
                temp_file = tempfile.NamedTemporaryFile(mode='w+t', delete=False)
                json.dump(data, temp_file)
                temp_file.close()
                settings_files_tempfiles.append(temp_file)
         
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
            run_deforum_args[prefixed_gradio_args + component_names.index('custom_settings_file')] = settings_files_tempfiles

            # Invoke deforum with appropriate args
            from deforum_helpers.run_deforum import run_deforum 
            run_deforum(*run_deforum_args)          

            return {"hello": "world"}    
 
    class Batch(BaseModel):
        settings_files : Optional[List[Dict[str, Any]]]
