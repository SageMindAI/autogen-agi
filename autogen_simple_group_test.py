import autogen


from better_group_chat import BetterGroupChat, BetterGroupChatManager

from predifined_agents import (
    user_proxy,
    code_reviewer,
    agent_awareness_expert,
    python_expert,
    function_calling_expert,
    agi_gestalt_agent,
    code_execution_agent,
    creative_solution_agent,
    first_principles_thinker_agent,
    out_of_the_box_thinker_agent,
    strategic_planning_agent,
    project_manager_agent,
    efficiency_optimizer_agent,
    emotional_intelligence_expert_agent,
)

from dotenv import load_dotenv

load_dotenv()

import os

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


AGENT_TEAM = [
    user_proxy,
    code_reviewer,
    agent_awareness_expert,
    python_expert,
    function_calling_expert,
    agi_gestalt_agent,
    code_execution_agent,
    creative_solution_agent,
    first_principles_thinker_agent,
    out_of_the_box_thinker_agent,
    strategic_planning_agent,
    project_manager_agent,
    efficiency_optimizer_agent,
    emotional_intelligence_expert_agent,
]

groupchat = BetterGroupChat(
    agents=AGENT_TEAM,
    messages=[],
    max_round=100,
    persona_discussion=True,
    inject_persona_discussion=True,
    continue_chat=False,
)
manager = BetterGroupChatManager(groupchat=groupchat, llm_config=llm_config4)

# message = """Please write a python script that prints 10 dad jokes and save it."""

# message = """Please execute the file to show that it works."""

message = """Please review the code in the "code_to_improve" directory and improve it with your best judgement. This includes (but is not limited to) re-arranging the code and directory structure if needed, adding comments, and re-naming functions if needed. I don't think you will be able to run the code, but please do your best to make sure there are no syntax errors and it is complieable. Save your results in the "code_improved" directory."""

user_proxy.initiate_chat(
    manager,
    clear_history=False,
    message=message,
)

# TODO: Add a function that allows injection of a new agent into the group chat.

# TODO: Add a function that allows spawning a new group chat with a new set of agents.
