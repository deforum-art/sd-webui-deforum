# This file is used to map deprecated setting names in a dictionary
# and print a message containing the old and the new names
# if the latter is removed completely, put a warning
deprecation_map = {
    "histogram_matching": None,
    "flip_2d_perspective": "enable_perspective_flip",
    "skip_video_for_run_all": "skip_video_creation",
    "color_coherence": [
        ("Match Frame 0 HSV", "HSV", False),
        ("Match Frame 0 LAB", "LAB", False),
        ("Match Frame 0 RGB", "RGB", False),
        # ,("removed_value", None, True) # for removed values, if we'll need in the future
    ],
    "hybrid_composite": [
        (False, "None", False),
        (True, "Normal", False),
    ],
    "optical_flow_redo_generation": [
        (False, "None", False),
        (True, "DIS Fine", False),
    ],
    "optical_flow_cadence": [
        (False, "None", False),
        (True, "DIS Fine", False),
    ]
}

def handle_deprecated_settings(settings_json):
    for setting_name, deprecation_info in deprecation_map.items():
        if setting_name in settings_json:
            if deprecation_info is None:
                print(f"WARNING: Setting '{setting_name}' has been removed. It will be discarded and the default value used instead!")
            elif isinstance(deprecation_info, str):
                print(f"WARNING: Setting '{setting_name}' has been renamed to '{deprecation_info}'. The saved settings file will reflect the change")
                settings_json[deprecation_info] = settings_json.pop(setting_name)
            elif isinstance(deprecation_info, list):
                for old_value, new_value, is_removed in deprecation_info:
                    if settings_json[setting_name] == old_value:
                        if is_removed:
                            print(f"WARNING: Value '{old_value}' for setting '{setting_name}' has been removed. It will be discarded and the default value used instead!")
                        else:
                            print(f"WARNING: Value '{old_value}' for setting '{setting_name}' has been replaced with '{new_value}'. The saved settings file will reflect the change")
                            settings_json[setting_name] = new_value
