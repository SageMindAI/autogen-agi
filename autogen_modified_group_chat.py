import autogen


from modified_group_chat import ModifiedGroupChat, ModifiedGroupChatManager

from predifined_agents import (
    user_proxy,
    code_reviewer,
    agent_awareness_expert,
    python_expert,
    function_calling_agent,
    agi_gestalt_agent,
    code_execution_agent,
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

groupchat = ModifiedGroupChat(
    agents=AGENT_TEAM,
    messages=[],
    max_round=100,
    persona_discussion=True,
    inject_persona_discussion=True,
    continue_chat=False,
)
manager = ModifiedGroupChatManager(groupchat=groupchat, llm_config=llm_config4)

message = """I'm interested in building autonomous agents using the autogen python library. Can you show me a complete example of how to do this? The example should show how to correctly configure and instantiate autogen automous agents. The request given to the agents will be: "Please write and then execute a python script that prints 10 dad jokes". I want the agents to run completely autonomously without any human intervention."""

user_proxy.initiate_chat(
    manager,
    clear_history=False,
    message=message,
)

# TODO: Add a function that allows injection of a new agent into the group chat.

# TODO: Add a function that allows spawning a new group chat with a new set of agents.
