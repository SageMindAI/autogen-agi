"""
This file contains the functions available to the FunctionCallingAgent.
"""


import autogen

from prompts.misc_prompts import (
    ARCHIVE_AGENT_MATCH_DOMAIN_PROMPT,
)

from pprint import pprint

from dotenv import load_dotenv

load_dotenv()

import os

from utils.misc import (
    light_gpt4_wrapper_autogen,
)

from utils.rag_tools import get_informed_answer
from utils.search_tools import find_relevant_github_repo

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

agent_functions = [
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

    find_domain_query = ARCHIVE_AGENT_MATCH_DOMAIN_PROMPT.format(
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
        vector_top_k=80,
        reranker_top_n=20,
        rerank=True,
        fusion=True,
    )

