import os
import sys

def deforum_sys_extend():
    deforum_folder_name = os.path.sep.join(os.path.abspath(__file__).split(os.path.sep)[:-2])

    basedirs = [os.getcwd()]
    if 'google.colab' in sys.modules:
        basedirs.append('/content/gdrive/MyDrive/sd/stable-diffusion-webui')  # for TheLastBen's colab
    for _ in basedirs:
        deforum_paths_to_ensure = [
            os.path.join(deforum_folder_name, 'scripts'),
            os.path.join(deforum_folder_name, 'scripts', 'deforum_helpers', 'src')
        ]
        for deforum_scripts_path_fix in deforum_paths_to_ensure:
            if deforum_scripts_path_fix not in sys.path:
                sys.path.extend([deforum_scripts_path_fix])
