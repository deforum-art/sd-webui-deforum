from tenacity import retry, stop_after_delay, wait_fixed
from pydantic_requests import PydanticSession
import requests
from scripts.deforum_api_models import DeforumJobStatus, DeforumJobStatusCategory, DeforumJobPhase

SERVER_BASE_URL = "http://localhost:7860"
API_ROOT = "/deforum_api"
API_BASE_URL = SERVER_BASE_URL + API_ROOT

@retry(wait=wait_fixed(2), stop=stop_after_delay(900))
def wait_for_job_to_complete(id : str):
    with PydanticSession(
        {200: DeforumJobStatus}, headers={"accept": "application/json"}
    ) as session:
        response = session.get(API_BASE_URL+"/jobs/"+id)
        response.raise_for_status()
        jobStatus : DeforumJobStatus = response.model
        print(f"Waiting for job {id}: status={jobStatus.status}; phase={jobStatus.phase}; execution_time:{jobStatus.execution_time}s")
        assert jobStatus.status != DeforumJobStatusCategory.ACCEPTED
        return jobStatus
    
@retry(wait=wait_fixed(1), stop=stop_after_delay(120))
def wait_for_job_to_enter_phase(id : str, phase : DeforumJobPhase):
    with PydanticSession(
        {200: DeforumJobStatus}, headers={"accept": "application/json"}
    ) as session:
        response = session.get(API_BASE_URL+"/jobs/"+id)
        response.raise_for_status()
        jobStatus : DeforumJobStatus = response.model
        print(f"Waiting for job {id} to enter phase {phase}. Currently: status={jobStatus.status}; phase={jobStatus.phase}; execution_time:{jobStatus.execution_time}s")
        assert jobStatus.phase != phase
        return jobStatus
    
@retry(wait=wait_fixed(1), stop=stop_after_delay(120))
def wait_for_job_to_enter_status(id : str, status : DeforumJobStatusCategory):
    with PydanticSession(
        {200: DeforumJobStatus}, headers={"accept": "application/json"}
    ) as session:
        response = session.get(API_BASE_URL+"/jobs/"+id)
        response.raise_for_status()
        jobStatus : DeforumJobStatus = response.model
        print(f"Waiting for job {id} to enter status {status}. Currently: status={jobStatus.status}; phase={jobStatus.phase}; execution_time:{jobStatus.execution_time}s")
        assert jobStatus.status == status
        return jobStatus


def gpu_disabled():
    response = requests.get(SERVER_BASE_URL+"/sdapi/v1/cmd-flags")
    response.raise_for_status()
    cmd_flags = response.json()
    return cmd_flags["use_cpu"] == ["all"]

        


