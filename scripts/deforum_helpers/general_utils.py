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
    import os, cv2, shutil #, subprocess
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