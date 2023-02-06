# This file is used to map deprecated setting names in a dictionary
# and print a message containing the old and the new names
# if the latter is removed completely, put a warning

# as of 2023-02-05
# "histogram_matching" -> None

deprecation_map = {
    "histogram_matching": None,
    "flip_2d_perspective": "enable_perspective_flip"
}

def handle_deprecated_settings(settings_json):
    for old_name, new_name in deprecation_map.items():
        if old_name in settings_json:
            if new_name is None:
                print(f"WARNING: Setting '{old_name}' has been removed. It will be discarded and the default value used instead!")
            else:
                print(f"WARNING: Setting '{old_name}' has been renamed to '{new_name}'. The saved settings file will reflect the change")
                settings_json[new_name] = settings_json.pop(old_name)
