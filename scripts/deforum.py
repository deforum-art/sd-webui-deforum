from modules import script_callbacks
from scripts.deforum_extend_paths import deforum_sys_extend

def init_deforum():
    deforum_sys_extend()

    from deforum_helpers.ui_right import on_ui_tabs
    script_callbacks.on_ui_tabs(on_ui_tabs)
    from deforum_helpers.ui_settings import on_ui_settings
    script_callbacks.on_ui_settings(on_ui_settings)

init_deforum()