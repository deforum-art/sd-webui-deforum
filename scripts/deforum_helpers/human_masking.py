import torch
from RobustVideoMatting.model import MattingNetwork
from RobustVideoMatting.inference import convert_video


def video2humanmasks(input_video_path, output_folder_path):
    model = torch.hub.load("PeterL1n/RobustVideoMatting", "mobilenetv3").cuda() 
    convert_video = torch.hub.load("PeterL1n/RobustVideoMatting", "converter")

    convert_video(
    model, 
    input_source=input_video_path,        # A video file or an image sequence directory.
    output_type='png_sequence',             # Choose "video" or "png_sequence"
    output_alpha=output_folder_path,          # [Optional] Output the raw alpha prediction.
    downsample_ratio=None,           # A hyperparameter to adjust or use None for auto.
    seq_chunk=12,                    # Process n frames at once for better parallelism.
    progress=True
    )