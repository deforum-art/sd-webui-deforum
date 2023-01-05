import os
import cv2
import shutil
import math
import requests
from modules.shared import state

def get_frame_name(path):
    name = os.path.basename(path)
    name = os.path.splitext(name)[0]
    return name

def vid2frames(video_path, video_in_frame_path, n=1, overwrite=True): 
    #get the name of the video without the path and ext
    name = get_frame_name(video_path)
    if n < 1: n = 1 #HACK Gradio interface does not currently allow min/max in gr.Number(...) 

    if video_path.startswith('http://') or video_path.startswith('https://'):
        response = requests.head(video_path)
        if response.status_code == 404 or response.status_code != 200:
            raise ConnectionError("Init video url or mask video url is not valid")
    else:
        if not os.path.exists(video_path):
            raise RuntimeError("Init video path or mask video path is not valid")

    input_content = []
    if os.path.exists(video_in_frame_path) :
        input_content = os.listdir(video_in_frame_path)

    # check if existing frame is the same video, if not we need to erase it and repopulate
    if len(input_content) > 0:
        #get the name of the existing frame
        content_name = get_frame_name(input_content[0])
        if not content_name.startswith(name):
            overwrite = True
    vidcap = cv2.VideoCapture(video_path)

    # grab the frame count to check against existing directory len 
    frame_count = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT)) 
    
    # raise error if the user wants to skip more frames than exist
    if n >= frame_count : 
        raise RuntimeError('Skipping more frames than input video contains. extract_nth_frames larger than input frames')
    
    expected_frame_count = math.ceil(frame_count / n) 
    # Check to see if the frame count is matches the number of files in path
    if overwrite or expected_frame_count != len(input_content):
        shutil.rmtree(video_in_frame_path)
        os.makedirs(video_in_frame_path, exist_ok=True) # just deleted the folder so we need to make it again
        input_content = os.listdir(video_in_frame_path)
    
    if len(input_content) == 0:
        success,image = vidcap.read()
        count = 0
        t=1
        success = True
        while success:
            if state.interrupted:
                return
            if count % n == 0:
                cv2.imwrite(video_in_frame_path + os.path.sep + name + f"{t:05}.jpg" , image)     # save frame as JPEG file
                t += 1
            success,image = vidcap.read()
            count += 1
        print("Converted %d frames" % count)
    else: print("Frames already unpacked")

def get_next_frame(outdir, video_path, frame_idx, mask=False):
    frame_path = 'inputframes'
    if (mask): frame_path = 'maskframes'
    return os.path.join(outdir, frame_path, get_frame_name(video_path) + f"{frame_idx+1:05}.jpg")
