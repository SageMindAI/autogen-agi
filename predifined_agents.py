import autogen
import requests
import subprocess

from src.agent_prompts import (
    PYTHON_EXPERT_AGENT_SYSTEM_PROMPT,
    FUNCTION_CALLING_AGENT_SYSTEM_PROMPT,
    USER_PROXY_SYSTEM_PROMPT,
    AGENT_AWARENESS_SYSTEM_PROMPT,
    ARCHIVE_AGENT_MATCH_DOMAIN_MESSAGE,
    CREATIVE_SOLUTION_AGENT_SYSTEM_PROMPT,
    AGI_GESTALT_SYSTEM_PROMPT,
    EFFICIENCY_OPTIMIZER_SYSTEM_PROMPT,
    EMOTIONAL_INTELLIGENCE_EXPERT_SYSTEM_PROMPT,
    OUT_OF_THE_BOX_THINKER_SYSTEM_PROMPT,
    PROJECT_MANAGER_SYSTEM_PROMPT,
    STRATEGIC_PLANNING_AGENT_SYSTEM_PROMPT,
    FIRST_PRINCIPLES_THINKER_SYSTEM_PROMPT,
    TASK_HISTORY_REVIEW_AGENT,
    RESEARCH_AGENT_RATE_URLS_MESSAGE,
    RESEARCH_AGENT_SUMMARIZE_MESSAGE,
    RESEARCH_AGENT_RATE_REPOS_MESSAGE,
    TASK_COMPREHENSION_AGENT_SYSTEM_PROMPT,
    RESEARCH_AGENT_SUMMARIZE_REPO_MESSAGE
)

from autogen import OpenAIWrapper

from ddgsearch import ddgsearch
from duckduckgo_search import ddg, DDGS

from pprint import pprint
import base64

from time import sleep

# NOTE: AssistantAgent and UserProxyAgent are small wrappers around ConversableAgent.

from dotenv import load_dotenv

load_dotenv()

import os
import copy
import json

from src.utils.agent_utils import get_end_intent
from src.utils.misc import (
    light_llm_wrapper,
    light_gpt4_wrapper_autogen,
    light_llm4_wrapper,
    extract_json_response,
)

from src.utils.rag_tools import get_informed_answer

from src.utils.fetch_docs import fetch_and_save

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

user_proxy = autogen.UserProxyAgent(
    name="UserProxy",
    system_message=USER_PROXY_SYSTEM_PROMPT,
    human_input_mode="TERMINATE",
    is_termination_msg=lambda x: get_end_intent(x) == "end",
    code_execution_config=False,
    llm_config=llm_config4,
)

code_reviewer = autogen.AssistantAgent(
    name="CodeReviewer",
    system_message="""You are an expert at reviewing code and suggesting improvements. Pay particluar attention to any potential syntax errors. Also, remind the Coding agent that they should always provide FULL and COMPLILABLE code and not shorten code blocks with comments such as '# Other class and method definitions remain unchanged...' or '# ... (previous code remains unchanged)'.""",
    llm_config=llm_config4,
)

agent_awareness_expert = autogen.AssistantAgent(
    name="AgentAwarenessExpert",
    system_message=AGENT_AWARENESS_SYSTEM_PROMPT,
    llm_config=llm_config4,
)

python_expert = autogen.AssistantAgent(
    name="PythonExpert",
    llm_config=llm_config4,
    system_message=PYTHON_EXPERT_AGENT_SYSTEM_PROMPT,
)

functions = [
    {
        "name": "read_file",
        "description": "Reads a file and returns the contents.",
        "parameters": {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": f"The absolute or relative path to the file. NOTE: By default the current working directory for this function is {WORK_DIR}.",
                },
            },
            "required": ["file_path"],
        },
    },
    # Definition for read_multiple_files
    {
        "name": "read_multiple_files",
        "description": "Reads multiple files and returns the contents.",
        "parameters": {
            "type": "object",
            "properties": {
                "file_paths": {
                    "type": "array",
                    "description": f"A list of absolute or relative paths to the files. NOTE: By default the current working directory for this function is {WORK_DIR}.",
                    "items": {"type": "string"},
                },
            },
            "required": ["file_paths"],
        },
    },
    # Definition for read_directory_contents
    {
        "name": "read_directory_contents",
        "description": "Reads the contents of a directory and returns the contents (i.e. the file names).",
        "parameters": {
            "type": "object",
            "properties": {
                "directory_path": {
                    "type": "string",
                    "description": f"The absolute or relative path to the directory. NOTE: By default the current working directory for this function is {WORK_DIR}.",
                },
            },
            "required": ["directory_path"],
        },
    },
    {
        "name": "save_file",
        "description": "Saves a file to disk.",
        "parameters": {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": f"The absolute or relative path to the file. NOTE: By default the current working directory for this function is {WORK_DIR}. This function does NOT allow for overwriting files.",
                },
                "file_contents": {
                    "type": "string",
                    "description": "The contents of the file to be saved.",
                },
            },
            "required": ["file_path", "file_contents"],
        },
    },
    # Definition for save_multiple_files
    {
        "name": "save_multiple_files",
        "description": "Saves multiple files to disk.",
        "parameters": {
            "type": "object",
            "properties": {
                "file_paths": {
                    "type": "array",
                    "description": f"A list of absolute or relative paths to the files. NOTE: By default the current working directory for this function is {WORK_DIR}. This function does NOT allow for overwriting files.",
                    "items": {"type": "string"},
                },
                "file_contents": {
                    "type": "array",
                    "description": "A list of the contents of the files to be saved.",
                    "items": {"type": "string"},
                },
            },
            "required": ["file_paths", "file_contents"],
        },
    },
    {
        "name": "execute_code_block",
        "description": f"""Execute a code block and return the output. The code block must be a string and labelled with the language. If the first line inside the code block is '# filename: <filename>' it will be saved to disk. Currently supported languages are: python and bash. NOTE: By default the current working directory for this function is {WORK_DIR}.""",
        "parameters": {
            "type": "object",
            "properties": {
                "lang": {
                    "type": "string",
                    "description": "The language of the code block.",
                },
                "code_block": {
                    "type": "string",
                    "description": "The block of code to be executed.",
                },
            },
            "required": ["lang", "code_block"],
        },
    },
    # TODO: This function should search the "DOMAIN_KNOWLEDGE_DIR", scan all sub-directories, read in their "description.txt" content, see if any directories match the domain, and return that directory name if it exists. Otherwise it should (optionally) spawn a new task to create that domain directory, find the best resource online for that domain, and save the information (html, pdf, etc.) to that directory. After this process, the "domain expert" will give a RAG response.
    ## OOORR: We have an "archive" bot that excells in searching for relevant information, and a "research" bot that excells in finding resources online and downloading them for the "archive" bot to process.
    # TODO: Give the agents the entire documenation of the "agent hierarchy" plan in their system message
    {
        "name": "consult_archive_agent",
        "description": """Ask a question to the archive agent. The archive agent has access to certain specific domain knowledge. The agent will search for any available domain knowledge that matches the domain related to the question and use that knowledge to formulate their response. The domains are generally very specific and niche content that would most likely be outside of GPT4 knowledge (such as detailed technical/api documentation).""",
        "parameters": {
            "type": "object",
            "properties": {
                "domain_description": {
                    "type": "string",
                    "description": f"The description of the domain of knowledge. The expert will use this description to do a similarity check against the available domain descriptions.",
                },
                "question": {
                    "type": "string",
                    "description": f"The question to ask the archive agent. Make sure you are explicit, specific, and detailed in your question.",
                },
            },
            "required": ["domain_description", "question"],
        },
    },
]


def read_file(file_path):
    resolved_path = os.path.abspath(os.path.normpath(f"{WORK_DIR}/{file_path}"))
    with open(resolved_path, "r") as f:
        return f.read()
    
def read_directory_contents(directory_path):
    resolved_path = os.path.abspath(os.path.normpath(f"{WORK_DIR}/{directory_path}"))
    return os.listdir(resolved_path)

def read_multiple_files(file_paths):
    resolved_paths = [os.path.abspath(os.path.normpath(f"{WORK_DIR}/{file_path}")) for file_path in file_paths]
    file_contents = []
    for resolved_path in resolved_paths:
        with open(resolved_path, "r") as f:
            file_contents.append(f.read())
    return file_contents

def save_file(file_path, file_contents):
    resolved_path = os.path.abspath(os.path.normpath(f"{WORK_DIR}/{file_path}"))
    # Throw error if file already exists
    if os.path.exists(resolved_path):
        raise Exception(f"File already exists at {resolved_path}.")
    
    # Create directory if it doesn't exist
    directory = os.path.dirname(resolved_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

    with open(resolved_path, "w") as f:
        f.write(file_contents)

    return f"File saved to {resolved_path}."

def save_multiple_files(file_paths, file_contents):
    resolved_paths = [os.path.abspath(os.path.normpath(f"{WORK_DIR}/{file_path}")) for file_path in file_paths]
    # Throw error if file already exists
    for resolved_path in resolved_paths:
        if os.path.exists(resolved_path):
            raise Exception(f"File already exists at {resolved_path}.")

    for i, resolved_path in enumerate(resolved_paths):
        # Create directory if it doesn't exist
        directory = os.path.dirname(resolved_path)
        if not os.path.exists(directory):
            os.makedirs(directory)

        with open(resolved_path, "w") as f:
            f.write(file_contents[i])

    return f"Files saved to {resolved_paths}."


code_execution_agent = autogen.AssistantAgent(
    name="CodeExecutionAgent",
    system_message="THIS AGENT IS ONLY USED FOR EXECUTING CODE. DO NOT USE THIS AGENT FOR ANYTHING ELSE.",
    llm_config=llm_config4,
    # NOTE: DO NOT use the last_n_messages parameter. It will cause the execution to fail.
    code_execution_config={"work_dir": WORK_DIR},
)


def execute_code_block(lang, code_block):
    # delete the "last_n_messages" parameter of code_execution_agent._code_execution_config
    code_execution_agent._code_execution_config.pop("last_n_messages", None)

    exitcode, logs = code_execution_agent.execute_code_blocks([(lang, code_block)])
    exitcode2str = "execution succeeded" if exitcode == 0 else "execution failed"
    return f"exitcode: {exitcode} ({exitcode2str})\nCode output: {logs}"


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


def google_github_search(query, numresults=10):
    print("SEARCHING GOOGLE FOR RELEVANT GITHUB REPOS...")
    excluded_urls = ["github.com/topics"]
    query = f"{query} site:github.com"

    start = 1  # Pagination start
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

    print("GOOGLE SEARCH RESULTS:", github_urls)
    return github_urls

def get_repo_details(repo_url):
    print(f"EXTRACTING GITHUB REPO DETAILS FOR `{repo_url}`...")
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

def search_github_repositories(query, numresults=10):
    repo_urls = google_github_search(query, numresults)
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

    print("CLONING:", clone_url, "\n")

    try:
        subprocess.run(clone_url, check=True, shell=True)
        return f"Repository {repo_name} downloaded to {directory}/{repo_name}"
    except subprocess.CalledProcessError as e:
        return f"Error: {str(e)}"
    
def find_relevant_github_repo(domain_description):
    repos = search_github_repositories(domain_description)

    # Truncate each repo readme to 5000 chars
    for repo in repos:
        repo['readme'] = repo['readme'][:5000]

    # Convert the list of url descriptions to a string
    str_desc = ""
    for repo in repos:
        # print("repo NAME: ", repo["name"])
        str_desc += f"URL: {repo['url']}\nNAME: {repo['name']}\n\DESCRIPTION: {repo['description']}\nREADME:\n{'*' * 50}\n{repo['readme']}\n{'*' * 50}\n\n"

    rate_repos_query = RESEARCH_AGENT_RATE_REPOS_MESSAGE.format(
        domain_description=domain_description,
        repository_descriptions=str_desc,
    )

    repo_rating_response = light_gpt4_wrapper_autogen(rate_repos_query, return_json=True)
    repo_rating_response = repo_rating_response["repository_ratings"]

    print("REPO_RATING_ANALYSIS:\n")
    pprint(repo_rating_response) 

    # Sort the list by the "rating" key
    repo_rating_response = sorted(repo_rating_response, key=lambda x: int(x["rating"]), reverse=True)

    

    # Get the top repository
    top_repo = repo_rating_response[0]

    print("TOP REPO:", top_repo, "\n")

    # Convert the list of domain descriptions to a string
    str_desc = ""
        # Get "readme" from matching item in url_descriptions
    for repo_desc in repos:
        if repo_desc["url"] == top_repo["url"]:
            top_repo["readme"] = repo_desc["readme"]
            top_repo["name"] = repo_desc["name"]
            break

    str_desc += f"URL: {top_repo['url']}\n\Title:\n{top_repo['title']}\nDescription:\n{'*' * 50}\n{top_repo['readme']}\n{'*' * 50}\n\n"

    summarize_repo_message = RESEARCH_AGENT_SUMMARIZE_REPO_MESSAGE.format(
        readme_content=str_desc
    )

    repo_summary_response = light_gpt4_wrapper_autogen(summarize_repo_message, return_json=True)

    print("REPO_SUMMARY:\n", repo_summary_response, "\n")

    # Create the domain directory under the "DOMAIN_KNOWLEDGE_DIR"
    domain_description = repo_summary_response["repo_description"]
    # The domain name is the repo name without the org/user
    domain_name = top_repo["name"].split("/")[1]
    domain_dir = os.path.join(DOMAIN_KNOWLEDGE_DOCS_DIR, domain_name)


    print(f"DOWNLOADING REPO: {top_repo['name']}...")

    # Download the domain content to the domain directory
    download_repository(top_repo["url"], DOMAIN_KNOWLEDGE_DOCS_DIR)

    print(f"REPO DOWNLOADED: {top_repo['name']}.")

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

    rate_url_query = RESEARCH_AGENT_RATE_URLS_MESSAGE.format(
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

    summarize_domain_message = RESEARCH_AGENT_SUMMARIZE_MESSAGE.format(
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

def consult_archive_agent(domain_description, question):
    # Traverse the first level of DOMAIN_KNOWLEDGE_DOCS_DIR and find the best match for the domain_description
    domain_descriptions = []
    for root, dirs, files in os.walk(DOMAIN_KNOWLEDGE_DOCS_DIR):
        print("FOUND DIRS:", dirs, root, files)
        for dir in dirs:
            # Get the files in the directory
            domain_name = dir
            for file in os.listdir(os.path.join(root, dir)):
                if file == "domain_description.txt":
                    with open(os.path.join(root, dir, file), "r") as f:
                        domain_descriptions.append(
                            {"domain_name": domain_name, "domain_description": f.read()}
                        )
        break

    # Convert the list of domain descriptions to a string
    str_desc = ""
    for desc in domain_descriptions:
        str_desc += f"Domain: {desc['domain_name']}\n\nDescription:\n{'*' * 50}\n{desc['domain_description']}\n{'*' * 50}\n\n"

    find_domain_query = ARCHIVE_AGENT_MATCH_DOMAIN_MESSAGE.format(
        domain_description=domain_description,
        available_domains=str_desc,
    )

    domain_response = light_gpt4_wrapper_autogen(find_domain_query, return_json=True)
    domain_response = domain_response["items"]
    print("DOMAIN_SEARCH_ANALYSIS:\n")
    pprint(domain_response)

    # Sort the domain_response by the "rating" key
    domain_response = sorted(domain_response, key=lambda x: int(x["rating"]), reverse=True)

    top_domain = domain_response[0]

    DOMAIN_RESPONSE_THRESHOLD = 5

    # If the top result has a rating below the threshold, research the domain knowledge online

    if top_domain["rating"] < DOMAIN_RESPONSE_THRESHOLD:
        print(f"Domain not found for domain description: {domain_description}")
        print("Searching for domain knowledge online...")
        domain, domain_description = find_relevant_github_repo(domain_description)
    else :
        domain = top_domain["domain"]
        domain_description = top_domain["domain_description"]

    return get_informed_answer(
        domain=domain,
        domain_description=domain_description,
        question=question,
        docs_dir=DOMAIN_KNOWLEDGE_DOCS_DIR,
        storage_dir=DOMAIN_KNOWLEDGE_STORAGE_DIR,
        rerank=True,
    )


function_llm_config = copy.deepcopy(llm_config4)
function_llm_config["functions"] = functions

function_calling_agent = autogen.AssistantAgent(
    name="FunctionCallingAgent",
    system_message=FUNCTION_CALLING_AGENT_SYSTEM_PROMPT,
    llm_config=function_llm_config,
    function_map={
        "read_file": read_file,
        "read_multiple_files": read_multiple_files,
        "read_directory_contents": read_directory_contents,
        "save_file": save_file,
        "save_multiple_files": save_multiple_files,
        "execute_code_block": execute_code_block,
        "consult_archive_agent": consult_archive_agent,
    },
)

creative_solution_agent = autogen.AssistantAgent(
    name="CreativeSolutionAgent",
    system_message=CREATIVE_SOLUTION_AGENT_SYSTEM_PROMPT,
    llm_config=llm_config4,
)

out_of_the_box_thinker_agent = autogen.AssistantAgent(
    name="OutOfTheBoxThinkerAgent",
    system_message=OUT_OF_THE_BOX_THINKER_SYSTEM_PROMPT,
    llm_config=llm_config4,
)

agi_gestalt_agent = autogen.AssistantAgent(
    name="AGIGestaltAgent",
    system_message=AGI_GESTALT_SYSTEM_PROMPT,
    llm_config=llm_config4,
)

project_manager_agent = autogen.AssistantAgent(
    name="ProjectManagerAgent",
    system_message=PROJECT_MANAGER_SYSTEM_PROMPT,
    llm_config=llm_config4,
)

first_principles_thinker_agent = autogen.AssistantAgent(
    name="FirstPrinciplesThinkerAgent",
    system_message=FIRST_PRINCIPLES_THINKER_SYSTEM_PROMPT,
    llm_config=llm_config4,
)

strategic_planning_agent = autogen.AssistantAgent(
    name="StrategicPlanningAgent",
    system_message=STRATEGIC_PLANNING_AGENT_SYSTEM_PROMPT,
    llm_config=llm_config4,
)

emotional_intelligence_expert_agent = autogen.AssistantAgent(
    name="EmotionalIntelligenceExpertAgent",
    system_message=EMOTIONAL_INTELLIGENCE_EXPERT_SYSTEM_PROMPT,
    llm_config=llm_config4,
)

efficiency_optimizer_agent = autogen.AssistantAgent(
    name="EfficiencyOptimizerAgent",
    system_message=EFFICIENCY_OPTIMIZER_SYSTEM_PROMPT,
    llm_config=llm_config4,
)

task_history_review_agent = autogen.AssistantAgent(
    name="TaskHistoryReviewAgent",
    system_message=TASK_HISTORY_REVIEW_AGENT,
    llm_config=llm_config4,
)

task_comprehension_agent = autogen.AssistantAgent(
    name="TaskComprehensionAgent",
    system_message=TASK_COMPREHENSION_AGENT_SYSTEM_PROMPT,
    llm_config=llm_config4,
)
