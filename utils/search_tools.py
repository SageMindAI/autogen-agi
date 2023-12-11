"""
This file contains functions for searching the web for domain knowledge.
"""

import requests
import subprocess

from prompts.misc_prompts import (
    RESEARCH_AGENT_RATE_URLS_PROMPT,
    RESEARCH_AGENT_SUMMARIZE_DOMAIN_PROMPT,
    RESEARCH_AGENT_RATE_REPOS_PROMPT,
    RESEARCH_AGENT_SUMMARIZE_REPO_PROMPT
)

from utils.ddgsearch import ddgsearch
from duckduckgo_search import ddg, DDGS

from pprint import pprint
import base64

from time import sleep

from dotenv import load_dotenv

load_dotenv()

import os
import json
import logging

from utils.misc import (
    light_gpt4_wrapper_autogen,
)

from utils.fetch_docs import fetch_and_save

logger = logging.getLogger(__name__)

google_search_api_key = os.environ["GOOGLE_SEARCH_API_KEY"]
google_custom_search_id = os.environ["GOOGLE_CUSTOM_SEARCH_ENGINE_ID"]
github_personal_access_token = os.environ["GITHUB_PERSONAL_ACCESS_TOKEN"]

config_list3 = [
    {
        "model": "gpt-3.5-turbo",
        "api_key": os.environ["OPENAI_API_KEY"],
    }
]

config_list4 = [
    {
        "model": "gpt-4-1106-preview",
        "api_key": os.environ["OPENAI_API_KEY"],
    }
]

llm_config4 = {
    "seed": 42,
    "config_list": config_list4,
    "temperature": 0.1,
}

WORK_DIR = "working"
DOMAIN_KNOWLEDGE_DOCS_DIR = "docs"
DOMAIN_KNOWLEDGE_STORAGE_DIR = "storage"
COMM_DIR = "url_search_results"

SEARCH_RESULTS_FILE = f"{COMM_DIR}\search_results.json"


def check_for_resource(file_path):
    # Check if the file exists
    if os.path.exists(file_path):
        return True
    return False

def wait_for_resource(file_path):
    # Check if the flag file exists in the
    while not os.path.exists(file_path):
        print(f"Waiting for resource '{file_path}'...")
        sleep(5)

    # Read in the contents
    with open(file_path, "r") as f:
        contents = f.read()
    # Delete the flag file
    os.remove(file_path)

    return contents

def google_custom_search(query, numresults=10, start=1):
    search_url = "https://www.googleapis.com/customsearch/v1"
    params = {
        'q': query,
        'cx': google_custom_search_id,
        'key': google_search_api_key,
        'num': numresults,
        'start': start
    }

    response = requests.get(search_url, params=params)
    search_results = []

    if response.status_code == 200:
        results = response.json().get('items', [])
        search_results = results
    else:
        print(f"Error: {response.status_code}")

    return search_results

def google_github_search(query, numresults=10, page=1):
    logger.info("Searching google for relevant github repos...")
    excluded_urls = ["github.com/topics"]
    query = f"{query} site:github.com"

    start = (page - 1) * numresults + 1
    github_urls = []

    while len(github_urls) < numresults:
        search_results = google_custom_search(query, numresults, start=start)

        for result in search_results:
            url = result['link']
            if any(excluded_url in url for excluded_url in excluded_urls):
                continue
            if "github.com" in url and url not in github_urls:
                github_urls.append(url)
                if len(github_urls) >= numresults:
                    return github_urls

        start += numresults  # Increment start for pagination
        if start > 50:  # Google's Custom Search API has a limit of 100 results
            print("Maximum number of results reached.")
            break

    return github_urls

def get_repo_details(repo_url):
    logger.info(f"Extracting repo details for: `{repo_url}`...")
    api_url = repo_url.replace("https://github.com/", "https://api.github.com/repos/")
    headers = {'Authorization': f'token {github_personal_access_token}'}
    repo_response = requests.get(api_url, headers=headers)
    readme_response = requests.get(api_url + "/contents/README.md", headers=headers)

    if repo_response.status_code == 200 and readme_response.status_code == 200:
        repo_data = repo_response.json()
        readme_data = readme_response.json()

        # Decode the Base64 encoded README content
        readme_content = base64.b64decode(readme_data.get('content', '')).decode('utf-8')

        return {
            'url': repo_url,
            'name': repo_data['full_name'],
            'description': repo_data['description'],
            'readme': readme_content
        }
    else:
        return None

def search_github_repositories(query, numresults=10, current_page=1):
    repo_urls = google_github_search(query, numresults, current_page)
    repositories_info = []

    for url in repo_urls:
        repo_details = get_repo_details(url)
        if repo_details:
            repositories_info.append(repo_details)

    return repositories_info

def ddg_github_search(query, numresults=10):
    excluded_urls = ["github.com/topics"]

    query = f"{query} site:github.com"
    
    github_urls = []

    # Initialize page count and results per page
    page = 1
    results_per_page = 5  # Adjust as needed

    while len(github_urls) < numresults:
        with DDGS() as ddgs:
            try:
                # Fetch a set number of results per 'page'
                current_results = [r for r in ddgs.text(query, max_results=results_per_page * page)]
                # Only process the latest set of results
                new_results = current_results[results_per_page * (page - 1):]

                for result in new_results:
                    url = result['href']
                    if any(excluded in url for excluded in excluded_urls):
                        continue
                    if "github.com" in url and url not in github_urls:
                        github_urls.append(url)
                        if len(github_urls) >= numresults:
                            return github_urls

                page += 1
                print("PAGE:", page, "RESULTS:", len(github_urls), "URLS:", github_urls, "\n")

            except Exception as e:
                print(f"Error occurred: {e}")
                continue

    return github_urls

def download_repository(repo_url, directory):
    # Extract repo name from URL
    repo_name = repo_url.rstrip('/').split('/')[-1]
    # Ensure the directory variable does not end with a '/'
    directory = directory.rstrip('/')
    clone_url = f"git clone {repo_url} {directory}/{repo_name}"

    logger.info(f"Cloning: {clone_url}\n")

    try:
        subprocess.run(clone_url, check=True, shell=True)
        return f"Repository {repo_name} downloaded to {directory}/{repo_name}"
    except subprocess.CalledProcessError as e:
        return f"Error: {str(e)}"
    
"""
This function searches github for repositories related to the domain description and clones the top repository to the "DOMAIN_KNOWLEDGE_DIR" directory.

Returns the domain name and description of the top repository found.
"""
def find_relevant_github_repo(domain_description, rating_threshold=6):
    repo_rating_response = []
    current_page = 0
    max_pages = 5
    while len(repo_rating_response) == 0:
        if current_page >= max_pages:
            raise Exception(f"Could not find a relevant repository for domain description: {domain_description}")
        current_page += 1
        repos = search_github_repositories(domain_description, numresults=10, current_page=current_page)

        # Truncate each repo readme to 5000 chars
        for repo in repos:
            repo['readme'] = repo['readme'][:5000]

        # Convert the list of url descriptions to a string
        str_desc = ""
        for repo in repos:
            str_desc += f"URL: {repo['url']}\nNAME: {repo['name']}\n\DESCRIPTION: {repo['description']}\nREADME:\n{'*' * 50}\n{repo['readme']}\n{'*' * 50}\n\n"

        rate_repos_query = RESEARCH_AGENT_RATE_REPOS_PROMPT.format(
            domain_description=domain_description,
            repository_descriptions=str_desc,
        )

        repo_rating_response = light_gpt4_wrapper_autogen(rate_repos_query, return_json=True)
        repo_rating_response = repo_rating_response["repository_ratings"]

        logger.info("Repo rating analysis:\n")
        for repo in repo_rating_response:
            logger.info(f"Repo: {repo['title']}")
            logger.info(f"Analysis: {repo['analysis']}")
            logger.info(f"Rating: {repo['rating']}\n")

        # filter out repos with a rating below the threshold
        repo_rating_response = [repo for repo in repo_rating_response if int(repo["rating"]) >= rating_threshold]

        if len(repo_rating_response) == 0:
            logger.info(f"No relevant repositories found above the threshold rating of {rating_threshold}. Trying again...")

    # Sort the list by the "rating" key
    repo_rating_response = sorted(repo_rating_response, key=lambda x: int(x["rating"]), reverse=True)

    # Get the top repository
    top_repo = repo_rating_response[0]

    logger.info(f"Top repository: {top_repo['title']}\n")

    # Convert the list of domain descriptions to a string
    str_desc = ""
        # Get "readme" from matching item in url_descriptions
    for repo_desc in repos:
        if repo_desc["url"] == top_repo["url"]:
            top_repo["readme"] = repo_desc["readme"]
            top_repo["name"] = repo_desc["name"]
            break

    str_desc += f"URL: {top_repo['url']}\n\Title:\n{top_repo['title']}\nDescription:\n{'*' * 50}\n{top_repo['readme']}\n{'*' * 50}\n\n"

    summarize_repo_message = RESEARCH_AGENT_SUMMARIZE_REPO_PROMPT.format(
        readme_content=str_desc
    )

    repo_summary_response = light_gpt4_wrapper_autogen(summarize_repo_message, return_json=True)

    logger.info(f"Repo summary:\n{repo_summary_response}\n")

    # Create the domain directory under the "DOMAIN_KNOWLEDGE_DIR"
    # The domain name is the repo name without the org/user
    domain_name = top_repo["name"].split("/")[1]
    domain_dir = os.path.join(DOMAIN_KNOWLEDGE_DOCS_DIR, domain_name)

    logger.info(f"Downloading repo: {top_repo['name']}...")

    # Download the domain content to the domain directory
    download_repository(top_repo["url"], DOMAIN_KNOWLEDGE_DOCS_DIR)

    logger.info(f"Repo downloaded: {top_repo['name']}.")

    domain_description = repo_summary_response["repo_description"]
    # Save the domain description to the domain directory
    with open(os.path.join(domain_dir, "domain_description.txt"), "w") as f:
        f.write(domain_description)

    return domain_name, domain_description

def research_domain_knowledge(domain_description):
    ddgsearch(domain_description, SEARCH_RESULTS_FILE, 10, False)

    url_descriptions = wait_for_resource(SEARCH_RESULTS_FILE)

    url_descriptions = json.loads(url_descriptions)

    # print("GOT URL DESCRIPTIONS: ", url_descriptions)


    # Convert the list of url descriptions to a string
    str_desc = ""
    for desc in url_descriptions:
        print("URL TITLE: ", desc["title"])
        str_desc += f"URL: {desc['url']}\n\Title:\n{desc['title']}\nDescription:\n{'*' * 50}\n{desc['useful_text']}\n{'*' * 50}\n\n"

    rate_url_query = RESEARCH_AGENT_RATE_URLS_PROMPT.format(
        domain_description=domain_description,
        url_descriptions=str_desc,
    )

    url_rating_response = light_gpt4_wrapper_autogen(rate_url_query, return_json=True)
    url_rating_response = url_rating_response["items"]

    print("URL_RATING_ANALYSIS:\n")
    pprint(url_rating_response) 

    # Sort the list by the "rating" key
    url_rating_response = sorted(url_rating_response, key=lambda x: x["rating"], reverse=True)

    # Get the top 5 domain names
    top_5_urls = url_rating_response[:5]


    # Convert the list of domain descriptions to a string
    str_desc = ""
    for desc in top_5_urls:
        # Get "useful_text" from matching item in url_descriptions
        for url_desc in url_descriptions:
            if url_desc["url"] == desc["url"]:
                desc["useful_text"] = url_desc["useful_text"]
                break
        str_desc += f"URL: {desc['url']}\n\Title:\n{desc['title']}\nDescription:\n{'*' * 50}\n{desc['useful_text']}\n{'*' * 50}\n\n"

    summarize_domain_message = RESEARCH_AGENT_SUMMARIZE_DOMAIN_PROMPT.format(
        example_domain_content=str_desc
    )

    domain_summary_response = light_gpt4_wrapper_autogen(summarize_domain_message, return_json=True)

    print("DOMAIN_SUMMARY_ANALYSIS:\n", domain_summary_response)

    # Create the domain directory under the "DOMAIN_KNOWLEDGE_DIR"
    domain_name = domain_summary_response["domain_name"]
    # Snake case the domain name
    domain_name = domain_name.replace(" ", "_").lower()
    domain_description = domain_summary_response["analysis"]
    domain_dir = os.path.join(DOMAIN_KNOWLEDGE_DOCS_DIR, domain_name)
    if not os.path.exists(domain_dir):
        os.makedirs(domain_dir)

    # Save the domain description to the domain directory
    with open(os.path.join(domain_dir, "domain_description.txt"), "w") as f:
        f.write(domain_description)

    print(f"DOWNLOADING DOMAIN CONTENT FOR DOMAIN: {domain_name}...")

    # Download the domain content to the domain directory
    for i, url in enumerate(top_5_urls):
        fetch_and_save(url["url"], url["url"], domain_dir, set())

    print(f"DOMAIN CONTENT DOWNLOADED FOR DOMAIN: {domain_name}.")

    return domain_name, domain_description
