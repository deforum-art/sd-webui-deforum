from rife.inference_video import * #run_video_infer

def video_infer_wrap(frame_interpolation_engine="RIFE46", frame_interpolation_x_amount="Disabled", frame_interpolation_slow_mo_amount="Disabled", orig_vid_path=None, orig_vid_fps=None):
    
    fps = None
    
    frame_interpolation_engine = "RIFE46"
    if frame_interpolation_x_amount != "Disabled":
        run_video_infer(video=orig_vid_path, output=None, model=frame_interpolation_engine, fps=fps)