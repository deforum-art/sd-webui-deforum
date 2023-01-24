import os
from rife.inference_video import run_rife_new_video_infer #*

def extract_number(string):
    if len(string) > 1 and string[1:].isdigit():
        return int(string[1:])
    else:
        return (-1)
        
def extract_rife_name(string):
    parts = string.split()
    if len(parts) != 2:
        raise ValueError("Input string should contain exactly 2 words")
    if parts[0] != "RIFE":
        raise ValueError("First word should be 'RIFE'")
    if parts[1][0] != "v" or not parts[1][1:].replace('.','').isdigit():
        raise ValueError("Second word should start with 'v' followed by 2 numbers")
    return "RIFE"+parts[1][1:].replace('.','')

   
def video_infer_wrap(frame_interpolation_engine=None, frame_interpolation_x_amount="Disabled", frame_interpolation_slow_mo_amount="Disabled", orig_vid_path=None, orig_vid_fps=None, deforum_models_path=None, real_audio_track=None, raw_output_imgs_path=None, img_batch_id=None, ffmpeg_crf=None, ffmpeg_preset=None):

    if frame_interpolation_x_amount != "Disabled":
        
        # extract clean numbers from values of 'x2' etc'
        interp_amount_clean_num = extract_number(frame_interpolation_x_amount)
        interp_slow_mo_clean_num = extract_number(frame_interpolation_slow_mo_amount)
        
        # for future, other models, check if the interpolation model is rife or something else
        actual_model_folder_name = None
        
        # **HANDLE RIFE INTERPOLATIONS** Other models might come in the future
        if frame_interpolation_engine.startswith("RIFE"):
            # change rife model name. e.g: 'RIFE v4.3' becomes 'RIFE43' 
            actual_model_folder_name = extract_rife_name(frame_interpolation_engine)

            if interp_amount_clean_num not in range(2, 11):
                raise Error("frame_interpolation_x_amount must be between 2x and 10x")
                    
            fps = orig_vid_fps * interp_amount_clean_num
            # calculate fps param to pass if slow_mo is not disabled
            if interp_slow_mo_clean_num != -1:
                if int(interp_slow_mo_clean_num) not in [2,4,8]:
                    raise Error("frame_interpolation_slow_mo_amount must be 2x, 4x or 8x")
                fps = orig_vid_fps * interp_amount_clean_num / interp_slow_mo_clean_num
            # run actual interpo
            if actual_model_folder_name is not None:
                run_rife_new_video_infer(interp_x_amount=interp_amount_clean_num, slow_mo_x_amount=interp_slow_mo_clean_num, output=None, model=actual_model_folder_name, fps=fps, deforum_models_path=deforum_models_path, audio_track=real_audio_track, raw_output_imgs_path=raw_output_imgs_path, img_batch_id=img_batch_id, ffmpeg_crf=ffmpeg_crf, ffmpeg_preset=ffmpeg_preset)
             