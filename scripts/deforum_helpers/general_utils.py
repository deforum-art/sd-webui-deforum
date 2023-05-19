import os
import shutil
import hashlib
from modules.shared import opts
from basicsr.utils.download_util import load_file_from_url

def debug_print(message):
    DEBUG_MODE = opts.data.get("deforum_debug_mode_enabled", False)
    if DEBUG_MODE:
        print(message)
            
def checksum(filename, hash_factory=hashlib.blake2b, chunk_num_blocks=128):
    h = hash_factory()
    with open(filename,'rb') as f: 
        while chunk := f.read(chunk_num_blocks*h.block_size): 
            h.update(chunk)
    return h.hexdigest()

def get_os():
    import platform
    return {"Windows": "Windows", "Linux": "Linux", "Darwin": "Mac"}.get(platform.system(), "Unknown")

# used in src/rife/inference_video.py and more, soon
def duplicate_pngs_from_folder(from_folder, to_folder, img_batch_id, orig_vid_name):
    import cv2
    #TODO: don't copy-paste at all if the input is a video (now it copy-pastes, and if input is deforum run is also converts to make sure no errors rise cuz of 24-32 bit depth differences)
    temp_convert_raw_png_path = os.path.join(from_folder, to_folder)
    os.makedirs(temp_convert_raw_png_path, exist_ok=True)
                
    frames_handled = 0
    for f in os.listdir(from_folder):
        if ('png' in f or 'jpg' in f) and '-' not in f and '_depth_' not in f and ((img_batch_id is not None and f.startswith(img_batch_id) or img_batch_id is None)):
            frames_handled +=1
            original_img_path = os.path.join(from_folder, f)
            if orig_vid_name is not None:
                shutil.copy(original_img_path, temp_convert_raw_png_path)
            else:
                image = cv2.imread(original_img_path)
                new_path = os.path.join(temp_convert_raw_png_path, f)
                cv2.imwrite(new_path, image, [cv2.IMWRITE_PNG_COMPRESSION, 0])
    return frames_handled
    
def convert_images_from_list(paths, output_dir, format):
    import os
    from PIL import Image
    # Ensure that the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Loop over all input images
    for i, path in enumerate(paths):
        # Open the image
        with Image.open(path) as img:
            # Generate the output filename
            filename = f"{i+1:09d}.{format}"
            # Save the image to the output directory
            img.save(os.path.join(output_dir, filename))
            
def get_deforum_version():
    from modules import extensions as mext
    try:
        for ext in mext.extensions:
            if ext.name in ["deforum", "deforum-for-automatic1111-webui", "sd-webui-deforum"] and ext.enabled:
                ext.read_info_from_repo() # need this call to get exten info on ui-launch, not to be removed
                return ext.version
        return "Unknown"
    except:
        return "Unknown"
        
def custom_placeholder_format(value_dict, placeholder_match):
    key = placeholder_match.group(1).lower()
    value = value_dict.get(key, key) or "_"
    if isinstance(value, dict) and value:
        first_key = list(value.keys())[0]
        value = str(value[first_key][0]) if isinstance(value[first_key], list) and value[first_key] else str(value[first_key])
    return str(value)[:50]

def test_long_path_support(base_folder_path):
    long_folder_name = 'A' * 300
    long_path = os.path.join(base_folder_path, long_folder_name)
    try:
        os.makedirs(long_path)
        shutil.rmtree(long_path)
        return True
    except OSError:
        return False

def get_max_path_length(base_folder_path):
    if get_os() == 'Windows':
        return (32767 if test_long_path_support(base_folder_path) else 260) - len(base_folder_path) - 1
    return 4096 - len(base_folder_path) - 1
   
def substitute_placeholders(template, arg_list, base_folder_path):
    import re
    # Find and update timestring values if resume_from_timestring is True
    resume_from_timestring = next((arg_obj.resume_from_timestring for arg_obj in arg_list if hasattr(arg_obj, 'resume_from_timestring')), False)
    resume_timestring = next((arg_obj.resume_timestring for arg_obj in arg_list if hasattr(arg_obj, 'resume_timestring')), None)

    if resume_from_timestring and resume_timestring:
        for arg_obj in arg_list:
            if hasattr(arg_obj, 'timestring'):
                arg_obj.timestring = resume_timestring

    max_length = get_max_path_length(base_folder_path)
    values = {attr.lower(): getattr(arg_obj, attr)
              for arg_obj in arg_list
              for attr in dir(arg_obj) if not callable(getattr(arg_obj, attr)) and not attr.startswith('__')}
    formatted_string = re.sub(r"{(\w+)}", lambda m: custom_placeholder_format(values, m), template)
    formatted_string = re.sub(r'[<>:"/\\|?*\s,]', '_', formatted_string)
    return formatted_string[:max_length]

def count_files_in_folder(folder_path):
    import glob
    file_pattern = folder_path + "/*"
    file_count = len(glob.glob(file_pattern))
    return file_count
    
def clean_gradio_path_strings(input_str):
    if isinstance(input_str, str) and input_str.startswith('"') and input_str.endswith('"'):
        return input_str[1:-1]
    else:
        return input_str
        
def download_file_with_checksum(url, expected_checksum, dest_folder, dest_filename):
    expected_full_path = os.path.join(dest_folder, dest_filename)
    if not os.path.exists(expected_full_path) and not os.path.isdir(expected_full_path):
        load_file_from_url(url=url, model_dir=dest_folder, file_name=dest_filename, progress=True)
        if checksum(expected_full_path) != expected_checksum:
            raise Exception(f"Error while downloading {dest_filename}.]nPlease manually download from: {url}\nAnd place it in: {dest_folder}")