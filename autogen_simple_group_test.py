import autogen

from better_group_chat import BetterGroupChat, BetterGroupChatManager
from src.agent_prompts import (
    PYTHON_EXPERT_AGENT_SYSTEM_PROMPT,
    FUNCTION_CALLING_AGENT_SYSTEM_PROMPT,
    USER_PROXY_SYSTEM_PROMPT,
    AGENT_AWARENESS_SYSTEM_PROMPT,
    ARCHIVE_AGENT_MATCH_DOMAIN_MESSAGE,
)

from autogen import OpenAIWrapper

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
    get_informed_answer,
)

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

WORK_DIR = "NEW_TEST_DIR"
DOMAIN_KNOWLEDGE_DOCS_DIR = "docs"
DOMAIN_KNOWLEDGE_STORAGE_DIR = "storage"

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


def save_file(file_path, file_contents):
    resolved_path = os.path.abspath(os.path.normpath(f"{WORK_DIR}/{file_path}"))
    # Throw error if file already exists
    if os.path.exists(resolved_path):
        raise Exception(f"File already exists at {resolved_path}.")

    with open(resolved_path, "w") as f:
        f.write(file_contents)

    return f"File saved to {resolved_path}."


code_execution_agent = autogen.AssistantAgent(
    name="CodeExecutionAgent",
    system_message="THIS AGENT IS ONLY USED FOR EXECUTING CODE. DO NOT USE THIS AGENT FOR ANYTHING ELSE.",
    llm_config=llm_config4,
    # NOTE: DO NOT use the last_n_messages parameter. It will cause the execution to fail.
    code_execution_config={"work_dir": WORK_DIR},
)


def execute_code_block(lang, code_block):
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

    find_domain_query = ARCHIVE_AGENT_MATCH_DOMAIN_MESSAGE.format(
        domain_description=domain_description,
        available_domains=str_desc,
    )

    domain_response = light_gpt4_wrapper_autogen(find_domain_query, return_json=True)
    print("DOMAIN_SEARCH_ANALYSIS:\n", domain_response["analysis"])
    domain = domain_response["domain"]
    domain_description = domain_response["domain_description"]
    print("\nDOMAIN:", domain)
    print("\nDOMAIN_DESCRIPTION:\n", domain_description)

    if domain is None:
        return f"Domain not found for domain description: {domain_description}"

    return get_informed_answer(
        domain=domain,
        domain_description=domain_description,
        question=question,
        docs_dir=DOMAIN_KNOWLEDGE_DOCS_DIR,
        storage_dir=DOMAIN_KNOWLEDGE_STORAGE_DIR,
    )


function_llm_config = copy.deepcopy(llm_config4)
function_llm_config["functions"] = functions

function_calling_expert = autogen.AssistantAgent(
    name="FunctionCallingExpert",
    system_message=FUNCTION_CALLING_AGENT_SYSTEM_PROMPT,
    llm_config=function_llm_config,
    function_map={
        "read_file": read_file,
        "save_file": save_file,
        "execute_code_block": execute_code_block,
        "consult_archive_agent": consult_archive_agent,
    },
)

groupchat = BetterGroupChat(
    agents=[
        user_proxy,
        code_reviewer,
        agent_awareness_expert,
        python_expert,
        function_calling_expert,
    ],
    messages=[],
    max_round=100,
    persona_discussion=True,
    inject_persona_discussion=True,
    continue_chat=False,
)
manager = BetterGroupChatManager(groupchat=groupchat, llm_config=llm_config4)

# message = """Please write a python script that prints 10 dad jokes and save it."""

# message = """Please execute the file to show that it works."""

message = """How do I route requests to specific indexes using llama_index?"""

user_proxy.initiate_chat(
    manager,
    clear_history=False,
    message=message,
)

# TODO: Add a function that allows injection of a new agent into the group chat.

# TODO: Add a function that allows spawning a new group chat with a new set of agents.
