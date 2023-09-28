import logging
from typing import Any, Dict
from modules.shared import opts

log = logging.getLogger(__name__)

class A1111OptionsOverrider(object):
    def __init__(self, opts_overrides: Dict[str, Any]):
        self.opts_overrides = opts_overrides

    def __enter__(self):
        if self.opts_overrides is not None and len(self.opts_overrides)>0:
            self.original_opts = {k: opts.data[k] for k in self.opts_overrides.keys() if k in opts.data}
            log.debug(f"Captured options to override: {self.original_opts}")                
            log.info(f"Setting options: {self.opts_overrides}")
            for k, v in self.opts_overrides.items():
                setattr(opts, k, v)
        else:
            self.original_opts = None
        return self
 
    def __exit__(self, exception_type, exception_value, traceback):
        if (exception_type is not None):
            log.warning(f"Error during batch execution: {exception_type} - {exception_value}")
            log.debug(f"{traceback}")
        if (self.original_opts is not None):
            log.info(f"Restoring options: {self.original_opts}")
            for k, v in self.original_opts.items():
                setattr(opts, k, v)
