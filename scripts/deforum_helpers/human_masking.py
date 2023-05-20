import os, cv2
import torch
from pathlib import Path
from multiprocessing import freeze_support

def extract_frames(input_video_path, output_imgs_path):
    # Open the video file
    vidcap = cv2.VideoCapture(input_video_path)
    
    # Get the total number of frames in the video
    frame_count = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Create the output directory if it does not exist
    os.makedirs(output_imgs_path, exist_ok=True)
        
    # Extract the frames
    for i in range(frame_count):
        success, image = vidcap.read()
        if success:
            cv2.imwrite(os.path.join(output_imgs_path, f"frame{i}.png"), image)
    print(f"{frame_count} frames extracted and saved to {output_imgs_path}")
    
def video2humanmasks(input_frames_path, output_folder_path, output_type, fps):
    # freeze support is needed for video outputting
    freeze_support()
    
    # check if input path exists and is a directory
    if not os.path.exists(input_frames_path) or not os.path.isdir(input_frames_path):
        raise ValueError("Invalid input path: {}".format(input_frames_path))
        
    # check if output path exists and is a directory
    if not os.path.exists(output_folder_path) or not os.path.isdir(output_folder_path):
        raise ValueError("Invalid output path: {}".format(output_folder_path))
    
    # check if output_type is valid
    valid_output_types = ["video", "pngs", "both"]
    if output_type.lower() not in valid_output_types:
        raise ValueError("Invalid output type: {}. Must be one of {}".format(output_type, valid_output_types))
    
    # try to predict where torch cache lives, so we can try and fetch models from cache in the next step
    predicted_torch_model_cache_path = os.path.join(Path.home(), ".cache", "torch", "hub", "hithereai_RobustVideoMatting_master")
    predicted_rvm_cache_testilfe = os.path.join(predicted_torch_model_cache_path, "hubconf.py")

    # try to fetch the models from cache, and only if it can't be find, download from the internet (to enable offline usage)
    try:
        # Try to fetch the models from cache
        convert_video = torch.hub.load(predicted_torch_model_cache_path, "converter", source='local')
        model = torch.hub.load(predicted_torch_model_cache_path, "resnet50", source='local').cuda()
    except:
        # Download from the internet if not found in cache
        convert_video = torch.hub.load("hithereai/RobustVideoMatting", "converter")
        model = torch.hub.load("hithereai/RobustVideoMatting", "resnet50").cuda()
        
    output_alpha_vid_path = os.path.join(output_folder_path, "human_masked_video.mp4")
    # extract humans masks from the input folder' imgs.
    # in this step PNGs will be extracted only if output_type is set to PNGs. Otherwise a video will be made, and in the case of Both, the video will be extracted in the next step to PNGs
    convert_video(
    model, 
    input_source=input_frames_path,  # full path of the folder that contains all of the extracted input imgs
    output_type='video' if output_type.upper() in ("VIDEO", "BOTH") else 'png_sequence',
    output_alpha=output_alpha_vid_path if output_type.upper() in ("VIDEO", "BOTH") else output_folder_path,
    output_video_mbps=4,
    output_video_fps=fps, 
    downsample_ratio=None, # None for auto
    seq_chunk=12, # Process n frames at once for better parallelism
    progress=True # show extraction progress
    )

    if output_type.lower() == "both":
        extract_frames(output_alpha_vid_path, output_folder_path)    
