import pytest
import subprocess
from tenacity import retry, stop_after_delay, wait_fixed
import requests


def pytest_addoption(parser):
    parser.addoption("--start-server", action="store_true", help="start the server before the test run (if not specified you must start the server manually)")

@pytest.fixture
def cmdopt(request):
    return request.config.getoption("--start-server")


@retry(wait=wait_fixed(5), stop=stop_after_delay(60))
def wait_for_service(url):
    response = requests.get(url)
    assert response.status_code == 200


@pytest.fixture(scope="session", autouse=True)
def start_server(request):
    if request.config.getoption("--start-server"):
        print("Starting server...")
        subprocess.Popen(["python", "-m", "coverage", "run", "--data-file=.coverage.server", "launch.py", "--skip-prepare-environment", "--skip-torch-cuda-test", "--test-server", "--no-half", "--disable-opt-split-attention", "--use-cpu", "all", "--add-stop-route", "--api", "--deforum-api"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE)
        wait_for_service('http://localhost:7680')
       
    else:
        print("Assuming server is already running...")