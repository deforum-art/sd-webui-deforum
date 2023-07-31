import os
import json
from tenacity import retry, stop_after_delay, wait_fixed
from scripts.deforum_api_models import DeforumJobStatus, DeforumJobStatusCategory
from pydantic_requests import PydanticSession
import requests

from scripts.deforum_helpers.subtitle_handler import get_user_values

SERVER_BASE_URL = "http://localhost:7860"
API_ROOT = "/deforum_api"
API_BASE_URL = SERVER_BASE_URL + API_ROOT

#
# Start server with:
#   python -m coverage run --data-file=.coverage.server launch.py --skip-prepare-environment --disable-nan-check  --no-half --disable-opt-split-attention --add-stop-route --api --deforum-api --ckpt ./test/test_files/empty.pt
#

@retry(wait=wait_fixed(2), stop=stop_after_delay(120))
def wait_for_job_to_complete(id):
    with PydanticSession(
        {200: DeforumJobStatus}, headers={"accept": "application/json"}
    ) as session:
        response = session.get(API_BASE_URL+"/jobs/"+id)
        response.raise_for_status()
        jobStatus : DeforumJobStatus = response.model
        print(f"Waiting for job {id}: status={jobStatus.status}; phase={jobStatus.phase}; execution_time:{jobStatus.execution_time}s")
        assert jobStatus.status != DeforumJobStatusCategory.ACCEPTED
        return jobStatus


def test_simple_settings(snapshot):
    with open('tests/testdata/test1.input_settings.txt', 'r') as settings_file:
        data = json.load(settings_file)
        response = requests.post(API_BASE_URL+"/batches", json={
            "deforum_settings":[data],
            "options_overrides": {
                "deforum_save_gen_info_as_srt": True,
                "deforum_save_gen_info_as_srt_params": get_user_values(),
                }
            })
        response.raise_for_status()
        job_id = response.json()["job_ids"][0]
        jobStatus = wait_for_job_to_complete(job_id)

        assert jobStatus.status == DeforumJobStatusCategory.SUCCEEDED, f"Job {job_id} failed: {jobStatus}"

        srt_filenname =  os.path.join(jobStatus.outdir, f"{jobStatus.timestring}.srt")
        with open(srt_filenname, 'r') as srt_file:
            assert srt_file.read() == snapshot






