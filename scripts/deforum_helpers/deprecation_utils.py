# This file is used to map deprecated setting names in a dictionary
# and print a message containing the old and the new names

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
    ],
    "cn_1_resize_mode": [
        ("Envelope (Outer Fit)", "Outer Fit (Shrink to Fit)", False),
        ("Scale to Fit (Inner Fit)", "Inner Fit (Scale to Fit)", False),
    ],
    "cn_2_resize_mode": [
        ("Envelope (Outer Fit)", "Outer Fit (Shrink to Fit)", False),
        ("Scale to Fit (Inner Fit)", "Inner Fit (Scale to Fit)", False),
    ],
    "cn_3_resize_mode": [
        ("Envelope (Outer Fit)", "Outer Fit (Shrink to Fit)", False),
        ("Scale to Fit (Inner Fit)", "Inner Fit (Scale to Fit)", False),
    ],
    "use_zoe_depth": ("depth_algorithm", [("True", "Zoe+AdaBins (old)"), ("False", "Midas+AdaBins (old)")]),
}

def dynamic_num_to_schedule_formatter(old_value):
    return f"0:({old_value})"
    
for i in range(1, 6): # 5 CN models in total
    deprecation_map[f"cn_{i}_weight"] = dynamic_num_to_schedule_formatter
    deprecation_map[f"cn_{i}_guidance_start"] = dynamic_num_to_schedule_formatter
    deprecation_map[f"cn_{i}_guidance_end"] = dynamic_num_to_schedule_formatter

def handle_deprecated_settings(settings_json):
    # Set legacy_colormatch mode to True when importing old files, so results are backwards-compatible. Print a message about it too
    if 'legacy_colormatch' not in settings_json:
        settings_json['legacy_colormatch'] = True
        print('\033[33mlegacy_colormatch is missing from settings file, so we are setting it to *True* for backwards compatability. You are welcome to test your file with that setting being disabled for better color coherency.\033[0m')
        print("")
    for setting_name, deprecation_info in deprecation_map.items():
        if setting_name in settings_json:
            if deprecation_info is None:
                print(f"WARNING: Setting '{setting_name}' has been removed. It will be discarded and the default value used instead!")
            elif isinstance(deprecation_info, tuple):
                new_setting_name, value_map = deprecation_info
                old_value = str(settings_json.pop(setting_name))  # Convert the boolean value to a string for comparison
                new_value = next((v for k, v in value_map if k == old_value), None)
                if new_value is not None:
                    print(f"WARNING: Setting '{setting_name}' has been renamed to '{new_setting_name}' with value '{new_value}'. The saved settings file will reflect the change")
                    settings_json[new_setting_name] = new_value
            elif callable(deprecation_info):
                old_value = settings_json[setting_name]
                if isinstance(old_value, (int, float)):
                    new_value = deprecation_info(old_value)
                    print(f"WARNING: Value '{old_value}' for setting '{setting_name}' has been replaced with '{new_value}'. The saved settings file will reflect the change")
                    settings_json[setting_name] = new_value
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