import os
import json
from scripts.deforum_api_models import DeforumJobStatus, DeforumJobStatusCategory, DeforumJobPhase
import requests
from moviepy.editor import VideoFileClip
import glob
from utils import wait_for_job_to_complete, wait_for_job_to_enter_phase, wait_for_job_to_enter_status, API_BASE_URL

from scripts.deforum_helpers.subtitle_handler import get_user_values

def test_post_process_FILM(snapshot):
    with open('tests/testdata/simple.input_settings.txt', 'r') as settings_file:
        deforum_settings = json.load(settings_file)
        
    deforum_settings["frame_interpolation_engine"] = "FILM"
    deforum_settings["frame_interpolation_x_amount"] = 3
    deforum_settings["frame_interpolation_slow_mo_enabled"] = False

    response = requests.post(API_BASE_URL+"/batches", json={
        "deforum_settings":[deforum_settings],
        "options_overrides": {
            "deforum_save_gen_info_as_srt": True,
            "deforum_save_gen_info_as_srt_params": get_user_values(),
            }
        })
    response.raise_for_status()
    job_id = response.json()["job_ids"][0]
    jobStatus = wait_for_job_to_complete(job_id)

    assert jobStatus.status == DeforumJobStatusCategory.SUCCEEDED, f"Job {job_id} failed: {jobStatus}"

    # Ensure parameters used at each frame have not regressed
    srt_filenname =  os.path.join(jobStatus.outdir, f"{jobStatus.timestring}.srt")
    with open(srt_filenname, 'r') as srt_file:
        assert srt_file.read() == snapshot

    # Ensure interpolated video format is as expected
    video_filenames = glob.glob(f'{jobStatus.outdir}/*FILM*.mp4', recursive=True)
    assert len(video_filenames) == 1, "Expected one FILM video to be generated"

    interpolated_video_filename =  video_filenames[0]
    clip = VideoFileClip(interpolated_video_filename)
    assert clip.fps == deforum_settings['fps'] * deforum_settings["frame_interpolation_x_amount"]  , "Video FPS does not match input settings (fps * interpolation amount)"
    assert clip.duration * clip.fps == deforum_settings['max_frames'] * deforum_settings["frame_interpolation_x_amount"], "Video frame count does not match input settings (including interpolation)"
    assert clip.size == [deforum_settings['W'], deforum_settings['H']] , "Video dimensions are not as expected"    

def test_post_process_RIFE(snapshot):
    with open('tests/testdata/simple.input_settings.txt', 'r') as settings_file:
        deforum_settings = json.load(settings_file)
        
    deforum_settings["frame_interpolation_engine"] = "RIFE v4.6"
    deforum_settings["frame_interpolation_x_amount"] = 3
    deforum_settings["frame_interpolation_slow_mo_enabled"] = False

    response = requests.post(API_BASE_URL+"/batches", json={
        "deforum_settings":[deforum_settings],
        "options_overrides": {
            "deforum_save_gen_info_as_srt": True,
            "deforum_save_gen_info_as_srt_params": get_user_values(),
            }
        })
    response.raise_for_status()
    job_id = response.json()["job_ids"][0]
    jobStatus = wait_for_job_to_complete(job_id)

    assert jobStatus.status == DeforumJobStatusCategory.SUCCEEDED, f"Job {job_id} failed: {jobStatus}"

    # Ensure parameters used at each frame have not regressed
    srt_filenname =  os.path.join(jobStatus.outdir, f"{jobStatus.timestring}.srt")
    with open(srt_filenname, 'r') as srt_file:
        assert srt_file.read() == snapshot

    # Ensure interpolated video format is as expected
    video_filenames = glob.glob(f'{jobStatus.outdir}/*RIFE*.mp4', recursive=True)
    assert len(video_filenames) == 1, "Expected one RIFE video to be generated"

    interpolated_video_filename =  video_filenames[0]
    clip = VideoFileClip(interpolated_video_filename)
    assert clip.fps == deforum_settings['fps'] * deforum_settings["frame_interpolation_x_amount"]  , "Video FPS does not match input settings (fps * interpolation amount)"
    assert clip.duration * clip.fps == deforum_settings['max_frames'] * deforum_settings["frame_interpolation_x_amount"], "Video frame count does not match input settings (including interpolation)"
    assert clip.size == [deforum_settings['W'], deforum_settings['H']] , "Video dimensions are not as expected"        

def test_post_process_UPSCALE(snapshot):
    with open('tests/testdata/simple.input_settings.txt', 'r') as settings_file:
        deforum_settings = json.load(settings_file)
        
    deforum_settings["r_upscale_video"] = True
    deforum_settings["r_upscale_factor"] = "x4"
    deforum_settings["r_upscale_model"] = "realesrgan-x4plus"

    response = requests.post(API_BASE_URL+"/batches", json={
        "deforum_settings":[deforum_settings],
        "options_overrides": {
            "deforum_save_gen_info_as_srt": True,
            "deforum_save_gen_info_as_srt_params": get_user_values(),
            }
        })
    response.raise_for_status()
    job_id = response.json()["job_ids"][0]
    jobStatus = wait_for_job_to_complete(job_id)

    assert jobStatus.status == DeforumJobStatusCategory.SUCCEEDED, f"Job {job_id} failed: {jobStatus}"

    # Ensure parameters used at each frame have not regressed
    srt_filenname =  os.path.join(jobStatus.outdir, f"{jobStatus.timestring}.srt")
    with open(srt_filenname, 'r') as srt_file:
        assert srt_file.read() == snapshot

    # Ensure interpolated video format is as expected
    video_filenames = glob.glob(f'{jobStatus.outdir}/*Upscaled*.mp4', recursive=True)
    assert len(video_filenames) == 1, "Expected one upscaled video to be generated"

    interpolated_video_filename =  video_filenames[0]
    clip = VideoFileClip(interpolated_video_filename)
    assert clip.fps == deforum_settings['fps'] , "Video FPS does not match input settings"
    assert clip.duration * clip.fps == deforum_settings['max_frames'], "Video frame count does not match input settings"
    assert clip.size == [deforum_settings['W']*4, deforum_settings['H']*4] , "Video dimensions are not as expected (including upscaling)"


def test_post_process_UPSCALE_FILM(snapshot):
    with open('tests/testdata/simple.input_settings.txt', 'r') as settings_file:
        deforum_settings = json.load(settings_file)
        
    deforum_settings["r_upscale_video"] = True
    deforum_settings["r_upscale_factor"] = "x4"
    deforum_settings["r_upscale_model"] = "realesrgan-x4plus"
    deforum_settings["frame_interpolation_engine"] = "FILM"
    deforum_settings["frame_interpolation_x_amount"] = 3
    deforum_settings["frame_interpolation_slow_mo_enabled"] = False
    deforum_settings["frame_interpolation_use_upscaled"] = True    

    response = requests.post(API_BASE_URL+"/batches", json={
        "deforum_settings":[deforum_settings],
        "options_overrides": {
            "deforum_save_gen_info_as_srt": True,
            "deforum_save_gen_info_as_srt_params": get_user_values(),
            }
        })
    response.raise_for_status()
    job_id = response.json()["job_ids"][0]
    jobStatus = wait_for_job_to_complete(job_id)

    assert jobStatus.status == DeforumJobStatusCategory.SUCCEEDED, f"Job {job_id} failed: {jobStatus}"

    # Ensure parameters used at each frame have not regressed
    srt_filenname =  os.path.join(jobStatus.outdir, f"{jobStatus.timestring}.srt")
    with open(srt_filenname, 'r') as srt_file:
        assert srt_file.read() == snapshot

    # Ensure interpolated video format is as expected
    video_filenames = glob.glob(f'{jobStatus.outdir}/*FILM*upscaled*.mp4', recursive=True)
    assert len(video_filenames) == 1, "Expected one upscaled video to be generated"

    interpolated_video_filename =  video_filenames[0]
    clip = VideoFileClip(interpolated_video_filename)
    assert clip.fps == deforum_settings['fps'] * deforum_settings["frame_interpolation_x_amount"]  , "Video FPS does not match input settings (fps * interpolation amount)"
    assert clip.duration * clip.fps == deforum_settings['max_frames'] * deforum_settings["frame_interpolation_x_amount"], "Video frame count does not match input settings (including interpolation)"
    assert clip.size == [deforum_settings['W']*4, deforum_settings['H']*4] , "Video dimensions are not as expected (including upscaling)"    
