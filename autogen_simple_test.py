import autogen

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
    name="User",
    human_input_mode="TERMINATE",
    is_termination_msg=lambda x: get_end_intent(x) == "end",
    code_execution_config={"last_n_messages": 3, "work_dir": 'TEST_DIR'},
    llm_config=llm_config4,
)

assistant = autogen.AssistantAgent(
    name="Assistant",
    llm_config=llm_config4,
    is_termination_msg=lambda x: get_end_intent(x) == "end",
)

user_proxy.initiate_chat(
    assistant,
    message="""Please work together to write some code that echo's the user's input. Use explicit "print" statements to prompt the user.

""",
)