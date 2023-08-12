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

import pytest
import subprocess
import sys
import os
from subprocess import Popen, PIPE, STDOUT
from pathlib import Path
from tenacity import retry, stop_after_delay, wait_fixed
import threading
import requests

def pytest_addoption(parser):
    parser.addoption("--start-server", action="store_true", help="start the server before the test run (if not specified, you must start the server manually)")

@pytest.fixture
def cmdopt(request):
    return request.config.getoption("--start-server")

@retry(wait=wait_fixed(5), stop=stop_after_delay(60))
def wait_for_service(url):
    response = requests.get(url, timeout=(5, 5))
    print(f"Waiting for server to respond 200 at {url} (response: {response.status_code})...")    
    assert response.status_code == 200

@pytest.fixture(scope="session", autouse=True)
def start_server(request):
    if request.config.getoption("--start-server"):

        # Kick off server subprocess
        script_directory = os.path.dirname(__file__)
        a1111_directory = Path(script_directory).parent.parent.parent  # sd-webui/extensions/deforum/tests/ -> sd-webui
        print(f"Starting server in {a1111_directory}...")
        proc = Popen(["python", "-m", "coverage", "run", "--data-file=.coverage.server", "launch.py",
                      "--skip-prepare-environment", "--skip-torch-cuda-test", "--test-server", "--no-half",
                      "--disable-opt-split-attention", "--use-cpu", "all", "--add-stop-route", "--api", "--deforum-api", "--listen"],
            cwd=a1111_directory,
            stdout=PIPE,
            stderr=STDOUT,
            universal_newlines=True)
        
        # ensure server is killed at the end of the test run
        request.addfinalizer(proc.kill)

        # Spin up separate thread to capture the server output to file and stdout
        def server_console_manager():
            with proc.stdout, open('serverlog.txt', 'ab') as logfile:
                for line in proc.stdout:
                    sys.stdout.write(f"[SERVER LOG] {line}")
                    sys.stdout.flush()
                    logfile.write(line.encode('utf-8'))
                    logfile.flush()
                proc.wait()
        
        threading.Thread(target=server_console_manager).start()
        
        # Wait for deforum API to respond
        wait_for_service('http://localhost:7860/deforum_api/jobs/')
       
    else:
        print("Checking server is already running / waiting for it to come up...")
        wait_for_service('http://localhost:7860/deforum_api/jobs/')