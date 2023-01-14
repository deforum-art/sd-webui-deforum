import os, time, cv2
import torch
from multiprocessing import freeze_support

def extract_frames(input_video_path, output_imgs_path):
    # Open the video file
    vidcap = cv2.VideoCapture(input_video_path)
    
    # Get the total number of frames in the video
    frame_count = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Create the output directory if it does not exist
    if not os.path.exists(output_imgs_path):
        os.makedirs(output_imgs_path)
        
    # Extract the frames
    for i in range(frame_count):
        success, image = vidcap.read()
        if success:
            cv2.imwrite(os.path.join(output_imgs_path, f"frame{i}.png"), image)
    print(f"{frame_count} frames extracted and saved to {output_imgs_path}")
    
def video2humanmasks(input_frames_path, output_folder_path, output_type, fps):
    # freeze support is needed for video outputting
    freeze_support()
    model = torch.hub.load("hithereai/RobustVideoMatting", "mobilenetv3").cuda() 
    convert_video = torch.hub.load("hithereai/RobustVideoMatting", "converter")
    
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

    if output_type.upper() == "BOTH":
        extract_frames(output_alpha_vid_path, output_folder_path)
    