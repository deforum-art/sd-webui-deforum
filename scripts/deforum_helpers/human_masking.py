import torch
from RobustVideoMatting.model import MattingNetwork
from RobustVideoMatting.inference import convert_video

def video2humanmasks(input_frames_path, output_folder_path):
    # load the the RVM model - options are: resnet50 or mobilenetv3 (default; quiker and performs almost the same as resnet)
    model = torch.hub.load("PeterL1n/RobustVideoMatting", "mobilenetv3").cuda() 
    convert_video = torch.hub.load("PeterL1n/RobustVideoMatting", "converter")

    # extract humans masks from the input folder with out PNGs
    convert_video(
    model, 
    input_source=input_frames_path,  # A video file or an image sequence directory
    output_type='png_sequence',      # Choose "video" or "png_sequence"
    output_alpha=output_folder_path, # Output the raw alpha prediction
    downsample_ratio=None,           # None for auto
    seq_chunk=12,                    # Process n frames at once for better parallelism
    progress=True                    # show extraction progress
    )