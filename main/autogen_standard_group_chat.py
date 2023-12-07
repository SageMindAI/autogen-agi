import autogen

from autogen import GroupChat, GroupChatManager

from agents.agents import (
    user_proxy,
    code_reviewer,
    agent_awareness_expert,
    python_expert,
    function_calling_agent,
    agi_gestalt_agent,
    creative_solution_agent,
    first_principles_thinker_agent,
    out_of_the_box_thinker_agent,
    strategic_planning_agent,
    project_manager_agent,
    efficiency_optimizer_agent,
    emotional_intelligence_expert_agent,
    task_history_review_agent,
    task_comprehension_agent,
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
    function_calling_agent,
    # agi_gestalt_agent,
    creative_solution_agent,
    first_principles_thinker_agent,
    # out_of_the_box_thinker_agent,
    # strategic_planning_agent,
    project_manager_agent,
    # efficiency_optimizer_agent,
    # emotional_intelligence_expert_agent,
    task_history_review_agent,
    task_comprehension_agent
]

groupchat = GroupChat(
    agents=AGENT_TEAM,
    messages=[],
    max_round=100,
)
manager = GroupChatManager(groupchat=groupchat, llm_config=llm_config4)

message = """I'm interested in building autonomous agents using the autogen python library. Can you show me a complete example of how to do this? The example should show how to instantiate autogen automous agents. The request given to the agents will be: "Please execute a python script that prints 10 dad jokes". I want the agents to run completely autonomously without any human intervention. Note: for env variables please use 'load_dotenv' from the 'dotenv' python library. If you need OpenAI keys use 'os.environ["OPENAI_API_KEY"]' to access them."""

user_proxy.initiate_chat(
    manager,
    clear_history=False,
    message=message,
)
