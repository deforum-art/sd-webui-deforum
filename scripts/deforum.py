import sys, os
from modules import script_callbacks

def deforum_sys_extend():
    deforum_folder_name = os.path.sep.join(os.path.abspath(__file__).split(os.path.sep)[:-2])

    basedirs = [os.getcwd()]
    if 'google.colab' in sys.modules:
        basedirs.append('/content/gdrive/MyDrive/sd/stable-diffusion-webui') # for TheLastBen's colab
    for basedir in basedirs:
        deforum_paths_to_ensure = [
            os.path.join(deforum_folder_name, 'scripts'),
            os.path.join(deforum_folder_name, 'scripts', 'deforum_helpers', 'src')
            ]
        for deforum_scripts_path_fix in deforum_paths_to_ensure:
            if not deforum_scripts_path_fix in sys.path:
                sys.path.extend([deforum_scripts_path_fix])

def init_deforum():
    deforum_sys_extend()

    from deforum_helpers.ui_right import on_ui_tabs
    script_callbacks.on_ui_tabs(on_ui_tabs)
    from deforum_helpers.ui_settings import on_ui_settings
    script_callbacks.on_ui_settings(on_ui_settings)

init_deforum()