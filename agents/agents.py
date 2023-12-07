import autogen
import requests
import subprocess

from prompts.agent_prompts import (
    PYTHON_EXPERT_SYSTEM_PROMPT,
    FUNCTION_CALLING_AGENT_SYSTEM_PROMPT,
    USER_PROXY_SYSTEM_PROMPT,
    AGENT_AWARENESS_SYSTEM_PROMPT,
    CREATIVE_SOLUTION_AGENT_SYSTEM_PROMPT,
    AGI_GESTALT_SYSTEM_PROMPT,
    EFFICIENCY_OPTIMIZER_SYSTEM_PROMPT,
    EMOTIONAL_INTELLIGENCE_EXPERT_SYSTEM_PROMPT,
    OUT_OF_THE_BOX_THINKER_SYSTEM_PROMPT,
    PROJECT_MANAGER_SYSTEM_PROMPT,
    STRATEGIC_PLANNING_AGENT_SYSTEM_PROMPT,
    FIRST_PRINCIPLES_THINKER_SYSTEM_PROMPT,
    TASK_HISTORY_REVIEW_AGENT_SYSTEM_PROMPT,
    TASK_COMPREHENSION_AGENT_SYSTEM_PROMPT,
)

from autogen import OpenAIWrapper

from utils.ddgsearch import ddgsearch
from duckduckgo_search import ddg, DDGS

from pprint import pprint
import base64

from time import sleep

from dotenv import load_dotenv

load_dotenv()

import os
import copy
import json

from utils.agent_utils import get_end_intent
from utils.misc import (
    light_llm_wrapper,
    light_gpt4_wrapper_autogen,
    light_llm4_wrapper,
    extract_json_response,
)

from agents.agent_functions import (
    agent_functions,
    read_file,
    read_multiple_files,
    read_directory_contents,
    save_file,
    save_multiple_files,
    execute_code_block,
    consult_archive_agent,
)

from utils.rag_tools import get_informed_answer

from utils.fetch_docs import fetch_and_save

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
    system_message=PYTHON_EXPERT_SYSTEM_PROMPT,
)

function_llm_config = copy.deepcopy(llm_config4)
function_llm_config["functions"] = agent_functions

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
    system_message=TASK_HISTORY_REVIEW_AGENT_SYSTEM_PROMPT,
    llm_config=llm_config4,
)

task_comprehension_agent = autogen.AssistantAgent(
    name="TaskComprehensionAgent",
    system_message=TASK_COMPREHENSION_AGENT_SYSTEM_PROMPT,
    llm_config=llm_config4,
)
