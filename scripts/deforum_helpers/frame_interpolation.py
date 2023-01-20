from rife.inference_video import run_rife_new_video_infer #*

def extract_number(string):
    if len(string) > 1 and string[1:].isdigit():
        return int(string[1:])
    else:
        raise ValueError("Input string should be a single character followed by a number")
        
def extract_rife_name(string):
    parts = string.split()
    if len(parts) != 2:
        raise ValueError("Input string should contain exactly 2 words")
    if parts[0] != "RIFE":
        raise ValueError("First word should be 'RIFE'")
    if parts[1][0] != "v" or not parts[1][1:].replace('.','').isdigit():
        raise ValueError("Second word should start with 'v' followed by 2 numbers")
    return "RIFE"+parts[1][1:].replace('.','')

   
def video_infer_wrap(frame_interpolation_engine=None, frame_interpolation_x_amount="Disabled", frame_interpolation_slow_mo_amount="Disabled", orig_vid_path=None, orig_vid_fps=None, deforum_models_path=None, add_soundtrack=None):
    
    if frame_interpolation_x_amount != "Disabled":
        
        # for future, other models, check if the interpolation model is rife or something else
        actual_model_folder_name = None
        
        # **HANDLE RIFE INTERPOLATIONS** Other models might come in the future
        if frame_interpolation_engine.startswith("RIFE"):
            # handle rife model name. e.g: 'RIFE v4.3' becomes 'RIFE43' 
            actual_model_folder_name = extract_rife_name(frame_interpolation_engine)
        
            #handle multi (how man times to interpolate) param, string to int and check value in range
            multi = extract_number(frame_interpolation_x_amount)
            if multi not in range(2, 11):
                raise Error("frame_interpolation_x_amount must be between 2x and 10x")
                    
            fps = None
            # calculate fps param to pass if slow_mo is not disabled
            if frame_interpolation_slow_mo_amount != 'Disabled':
                x_slow_mo = extract_number(frame_interpolation_slow_mo_amount)
                if x_slow_mo not in [2,4,8]:
                    raise Error("frame_interpolation_slow_mo_amount must be 2x, 4x or 8x")
                fps = orig_vid_fps * multi / x_slow_mo
            # run actual interpo
            if actual_model_folder_name is not None:
                run_rife_new_video_infer(video=orig_vid_path, output=None, model=actual_model_folder_name, fps=fps, multi=multi, deforum_models_path=deforum_models_path, add_soundtrack=add_soundtrack)