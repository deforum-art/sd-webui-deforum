import os
import cv2
from modules.shared import opts

# Resume requires at least two actual frames in order to work
# 'Actual' frames are defined as frames that go through generation
# - Can't resume from a single frame.
# - If you have a cadence of 10, you need at least 10 frames in order to resume. 
# - Resume grabs the last actual frame and the 2nd to last actual frame
#   in order to work with cadence properly and feed it the prev_img/next_img

def get_resume_vars(folder, timestring, cadence):
    DEBUG_MODE = opts.data.get("deforum_debug_mode_enabled", False)
    # count previous frames
    frame_count = 0
    for item in os.listdir(folder):
        # don't count txt files or mp4 files
        if ".txt" in item or ".mp4" in item: 
            pass
        else:
            filename = item.split("_")
            # other image file types may be supported in the future,
            # so we just count files containing timestring
            # that don't contain the depth keyword (depth maps are saved in same folder)
            if timestring in filename and "depth" not in filename:
                frame_count += 1
                # add this to debugging var
                if DEBUG_MODE:
                    print(f"\033[36mResuming:\033[0m File: {filename}")

    print(f"\033[36mResuming:\033[0m Current frame count: {frame_count}")

    # get last frame from frame count corrected for any trailing cadence frames
    last_frame = frame_count - (frame_count % cadence)

    # calculate previous actual frame
    prev_frame = last_frame - cadence

    # calculate next actual frame
    next_frame = last_frame - 1
   
    # get prev_img/next_img from prev/next frame index (files start at 0, so subtract 1 for index var)
    path = os.path.join(folder, f"{timestring}_{prev_frame:09}.png")  
    prev_img = cv2.imread(path)
    path = os.path.join(folder, f"{timestring}_{next_frame:09}.png")  
    next_img = cv2.imread(path)

    # report resume last/next in console
    print(f"\033[36mResuming:\033[0m Last frame: {prev_frame} - Next frame: {next_frame} ")

    # returns:
    #   last frame count, accounting for cadence
    #   next frame count, accounting for cadence
    #   prev frame's image cv2 BGR
    #   next frame's image cv2 BGR
    return prev_frame, next_frame, prev_img, next_img
