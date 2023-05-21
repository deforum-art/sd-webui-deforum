import os
from modules import script_callbacks
import modules.paths as ph
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
