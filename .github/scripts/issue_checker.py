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

import os
import re
from github import Github

# Get GitHub token from environment variables
token = os.environ['GITHUB_TOKEN']
g = Github(token)

# Get the current repository
print(f"Repo is {os.environ['GITHUB_REPOSITORY']}")
repo = g.get_repo(os.environ['GITHUB_REPOSITORY'])

# Get the issue number from the event payload
#issue_number = int(os.environ['ISSUE_NUMBER'])

for issue in repo.get_issues():
    print(f"Processing issue №{issue.number}")
    if issue.pull_request:
        continue

    # Get the issue object
    #issue = repo.get_issue(issue_number)

    # Define the keywords to search for in the issue
    keywords = ['Python', 'Commit hash', 'Launching Web UI with arguments', 'Model loaded', 'deforum']
    excuse = 'I have a good reason for not including the log'

    # Check if ALL of the keywords are present in the issue
    def check_keywords(issue_body, keywords):
        for keyword in keywords:
            if not re.search(r'\b' + re.escape(keyword) + r'\b', issue_body, re.IGNORECASE):
                if not re.search(re.escape(excuse), issue_body, re.IGNORECASE):
                    return False
        return True

    # Check if the issue title has at least a specified number of words
    def check_title_word_count(issue_title, min_word_count):
        words = issue_title.replace("/", " ").replace("\\\\", " ").split()
        return len(words) >= min_word_count

    # Check if the issue title is concise
    def check_title_concise(issue_title, max_word_count):
        words = issue_title.replace("/", " ").replace("\\\\", " ").split()
        return len(words) <= max_word_count

    # Check if the commit ID is in the correct hash form
    def check_commit_id_format(issue_body):
        match = re.search(r'webui commit id - ([a-fA-F0-9]+|\[[a-fA-F0-9]+\])', issue_body)
        if not match:
            print('webui_commit_id not found')
            return False
        webui_commit_id = match.group(1)
        print(f'webui_commit_id {webui_commit_id}')
        webui_commit_id = webui_commit_id.replace("[", "").replace("]", "")
        if not (7 <= len(webui_commit_id) <= 40):
            print(f'invalid length!')
            return False
        match = re.search(r'deforum exten commit id - ([a-fA-F0-9]+|\[[a-fA-F0-9]+\])', issue_body)
        if match:
            print('deforum commit id not found')
            return False
        t2v_commit_id = match.group(1)
        print(f'deforum_commit_id {t2v_commit_id}')
        t2v_commit_id = t2v_commit_id.replace("[", "").replace("]", "")
        if not (7 <= len(t2v_commit_id) <= 40):
            print(f'invalid length!')
            return False
        return True

    # Only if a bug report
    if '[Bug]' in issue.title and not '[Feature Request]' in issue.title:
        print('The issue is eligible')
        # Initialize an empty list to store error messages
        error_messages = []

        # Check for each condition and add the corresponding error message if the condition is not met
        if not check_keywords(issue.body, keywords):
            error_messages.append("Include **THE FULL LOG FROM THE START OF THE WEBUI** in the issue description.")

        if not check_title_word_count(issue.title, 3):
            error_messages.append("Make sure the issue title has at least 3 words.")

        if not check_title_concise(issue.title, 13):
            error_messages.append("The issue title should be concise and contain no more than 13 words.")

        # if not check_commit_id_format(issue.body):
            # error_messages.append("Provide a valid commit ID in the format 'commit id - [commit_hash]' **both** for the WebUI and the Extension.")
            
        # If there are any error messages, close the issue and send a comment with the error messages
        if error_messages:
            print('Invalid issue, closing')
            # Add the "not planned" label to the issue
            not_planned_label = repo.get_label("wrong format")
            issue.add_to_labels(not_planned_label)
            
            # Close the issue
            issue.edit(state='closed')
            
            # Generate the comment by concatenating the error messages
            comment = "This issue has been closed due to incorrect formatting. Please address the following mistakes and reopen the issue (click on the 'Reopen' button below):\n\n"
            comment += "\n".join(f"- {error_message}" for error_message in error_messages)

            # Add the comment to the issue
            issue.create_comment(comment)
        elif repo.get_label("wrong format") in issue.labels:
            print('Issue is fine')
            issue.edit(state='open')
            issue.delete_labels()
            bug_label = repo.get_label("bug")
            issue.add_to_labels(bug_label)
            comment = "Thanks for addressing your formatting mistakes. The issue has been reopened now."
            issue.create_comment(comment)
