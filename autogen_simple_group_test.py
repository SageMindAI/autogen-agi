import autogen
from autogen import GroupChat, GroupChatManager

from better_group_chat import BetterGroupChat, BetterGroupChatManager

from dotenv import load_dotenv

load_dotenv()

import os
import json

from src.utils.agent_utils import get_end_intent

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
    # "request_timeout": 600,
    "seed": 42,
    "config_list": config_list4,
    "temperature": 0.1,
}


user_proxy = autogen.UserProxyAgent(
    name="UserProxy",
    system_message="""You are a proxy for the user. You will be able to see the conversation between the assistants. You will ONLY be prompted when there is a need for human input or the conversation is over. If you are ever prompted directly for a resopnse, always respond with: 'Thank you for the help! I will now end the conversation so the user can respond.'
    
!!!IMPORTANT: NEVER respond with anything other than the above message. If you do, the user will not be able to respond to the assistants.""",
    human_input_mode="TERMINATE",
    is_termination_msg=lambda x: get_end_intent(x) == "end",
    # code_execution_config={"last_n_messages": 3, "work_dir": 'TEST_DIR'},
    code_execution_config=False,
    llm_config=llm_config4,
)

assistant1 = autogen.AssistantAgent(
    name="Jokester",
    system_message="You are an expert at writing jokes and NOTHING ELSE. Secifically you DO NOT write code.",
    llm_config=llm_config4,
    # is_termination_msg=lambda x: get_end_intent(x) == "end",
)

assistant2 = autogen.AssistantAgent(
    name="Critiquer",
    system_message="You are an expert at critiquing jokes and NOTHING ELSE. Secifically you DO NOT write code.",
    llm_config=llm_config4,
    # is_termination_msg=lambda x: get_end_intent(x) == "end",
)

assistant3 = autogen.AssistantAgent(
    name="Coder",
    llm_config=llm_config4,
    # code_execution_config={"last_n_messages": 5, "work_dir": 'TEST_DIR'},
    # is_termination_msg=lambda x: get_end_intent(x) == "end",
)

assistant4 = autogen.AssistantAgent(
    name="Executor",
    system_message="You are an expert at executing code and NOTHING ELSE. Secifically you DO NOT write code. If any other agents provide a code block for you to execute, you must execute it and provide the output.",
    llm_config=llm_config4,
    code_execution_config={"last_n_messages": 5, "work_dir": 'TEST_DIR'},
    # is_termination_msg=lambda x: get_end_intent(x) == "end",
)

groupchat = BetterGroupChat(agents=[user_proxy, assistant1, assistant2, assistant3, assistant4], messages=[], max_round=12, persona_discussion=True, inject_persona_discussion=True)
manager = BetterGroupChatManager(groupchat=groupchat, llm_config=llm_config4)

user_proxy.initiate_chat(manager, message="Please come up with a list of 10 excellent jokes for a standup comedy routine. They must appeal to an adult AND child audience. When you are satisfied that the jokes are sufficient, please provide a python script that will output one of the jokes at random when it is run.")

# TODO: Add ability for the AGENT_COUNCIL to decide if a new agent should be created and added to the groupchat.