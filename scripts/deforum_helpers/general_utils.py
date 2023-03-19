import os, shutil
import hashlib

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
    if not os.path.exists(temp_convert_raw_png_path):
                os.makedirs(temp_convert_raw_png_path)
                
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
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

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
            if ext.name in ["deforum", "deforum-for-automatic1111-webui"] and ext.enabled:
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
    max_length = get_max_path_length(base_folder_path)
    values = {attr.lower(): getattr(arg_obj, attr)
              for arg_obj in arg_list
              for attr in dir(arg_obj) if not callable(getattr(arg_obj, attr)) and not attr.startswith('__')}
    formatted_string = re.sub(r"{(\w+)}", lambda m: custom_placeholder_format(values, m), template)
    formatted_string = re.sub(r'[<>:"/\\|?*\s,]', '_', formatted_string)
    return formatted_string[:max_length]
