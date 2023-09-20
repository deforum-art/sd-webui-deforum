# Copyright (C) 2023 Deforum LLC
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, version 3 of the License.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

# Contact the authors: https://deforum.github.io/

import json
import math
import os
from pathlib import Path

import requests
from moviepy.editor import VideoFileClip
from utils import (API_BASE_URL, wait_for_job_to_complete,
                   wait_for_job_to_enter_phase, wait_for_job_to_enter_status)

from scripts.deforum_api_models import (DeforumJobPhase, DeforumJobStatus,
                                        DeforumJobStatusCategory)
from scripts.deforum_helpers.subtitle_handler import get_user_values


def test_simple_settings(snapshot):
    with open('tests/testdata/simple.input_settings.txt', 'r') as settings_file:
        deforum_settings = json.load(settings_file)

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

    # Ensure video format is as expected
    video_filename =  os.path.join(jobStatus.outdir, f"{jobStatus.timestring}.mp4")
    clip = VideoFileClip(video_filename)
    assert clip.fps == deforum_settings['fps'] , "Video FPS does not match input settings"
    assert clip.duration * clip.fps == deforum_settings['max_frames'] , "Video frame count does not match input settings"
    assert clip.size == [deforum_settings['W'], deforum_settings['H']] , "Video dimensions are not as expected"
        

def test_api_cancel_active_job():
    with open('tests/testdata/simple.input_settings.txt', 'r') as settings_file:
        data = json.load(settings_file)
        response = requests.post(API_BASE_URL+"/batches", json={"deforum_settings":[data]})
        response.raise_for_status()
        job_id = response.json()["job_ids"][0]
        wait_for_job_to_enter_phase(job_id, DeforumJobPhase.GENERATING)

        cancel_url = API_BASE_URL+"/jobs/"+job_id
        response = requests.delete(cancel_url)
        response.raise_for_status()
        assert response.status_code == 200, f"DELETE request to {cancel_url} failed: {response.status_code}"

        jobStatus = wait_for_job_to_complete(job_id)

        assert jobStatus.status == DeforumJobStatusCategory.CANCELLED, f"Job {job_id} did not cancel: {jobStatus}"


def test_3d_mode(snapshot):
    with open('tests/testdata/simple.input_settings.txt', 'r') as settings_file:
        deforum_settings = json.load(settings_file)
        
    deforum_settings['animation_mode'] = "3D"
    
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

    # Ensure video format is as expected
    video_filename =  os.path.join(jobStatus.outdir, f"{jobStatus.timestring}.mp4")
    clip = VideoFileClip(video_filename)
    assert clip.fps == deforum_settings['fps'] , "Video FPS does not match input settings"
    assert clip.duration * clip.fps == deforum_settings['max_frames'] , "Video frame count does not match input settings"
    assert clip.size == [deforum_settings['W'], deforum_settings['H']] , "Video dimensions are not as expected"


def test_with_parseq_inline_without_overrides(snapshot):
    with open('tests/testdata/simple.input_settings.txt', 'r') as settings_file:
        deforum_settings = json.load(settings_file)

    with open('tests/testdata/parseq.json', 'r') as parseq_file:
        parseq_data = json.load(parseq_file)
       
    deforum_settings['parseq_manifest'] = json.dumps(parseq_data)
    deforum_settings["parseq_non_schedule_overrides"] = False
    
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

    # Ensure video format is as expected
    video_filename =  os.path.join(jobStatus.outdir, f"{jobStatus.timestring}.mp4")
    clip = VideoFileClip(video_filename)
    assert clip.fps == deforum_settings['fps'] , "Video FPS does not match input settings"
    assert clip.duration * clip.fps == deforum_settings['max_frames'] , "Video frame count does not match input settings"
    assert clip.size == [deforum_settings['W'], deforum_settings['H']] , "Video dimensions are not as expected"


def test_with_parseq_inline_with_overrides(snapshot):
    with open('tests/testdata/simple.input_settings.txt', 'r') as settings_file:
        deforum_settings = json.load(settings_file)

    with open('tests/testdata/parseq.json', 'r') as parseq_file:
        parseq_data = json.load(parseq_file)
       
    deforum_settings['parseq_manifest'] = json.dumps(parseq_data)
    deforum_settings["parseq_non_schedule_overrides"] = True
    
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

    # Ensure video format is as expected
    video_filename =  os.path.join(jobStatus.outdir, f"{jobStatus.timestring}.mp4")
    clip = VideoFileClip(video_filename)
    expected_frame_count = len(parseq_data['rendered_frames'])
    assert clip.fps == parseq_data['options']['output_fps'] , "Video FPS does not match input settings"
    assert math.ceil(clip.duration * clip.fps) == expected_frame_count, "Video frame count does not match input settings"
    assert clip.size == [deforum_settings['W'], deforum_settings['H']] , "Video dimensions are not as expected"

# def test_with_parseq_url():

def test_with_hybrid_video(snapshot):
    with open('tests/testdata/simple.input_settings.txt', 'r') as settings_file:
        deforum_settings = json.load(settings_file)

    with open('tests/testdata/parseq.json', 'r') as parseq_file:
        parseq_data = json.load(parseq_file)
        
    init_video_local_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "testdata", "example_init_vid.mp4")
    deforum_settings['video_init_path'] = init_video_local_path
    deforum_settings['extract_nth_frame'] = 200 # input video is 900 frames, so we should keep 5 frames
    deforum_settings["hybrid_generate_inputframes"] = True
    deforum_settings["hybrid_composite"] = "Normal"
    
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

    # Ensure video format is as expected
    video_filename =  os.path.join(jobStatus.outdir, f"{jobStatus.timestring}.mp4")
    clip = VideoFileClip(video_filename)
    assert clip.fps == deforum_settings['fps'] , "Video FPS does not match input settings"
    assert clip.duration == 5 / deforum_settings['fps'], "Video frame count does not match input settings"
    assert clip.size == [deforum_settings['W'], deforum_settings['H']] , "Video dimensions are not as expected"

