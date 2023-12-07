# filename: autonomous_agents_integration.py
from autogen import AssistantAgent, UserProxyAgent, config_list_from_json
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Load LLM configuration from environment or a file
config_list = config_list_from_json(env_or_file="OAI_CONFIG_LIST")
llm_config = {"config_list": config_list}

# Create an AssistantAgent instance
assistant = AssistantAgent(
    name="assistant",
    llm_config=llm_config,
)

# Create a UserProxyAgent instance with autonomous settings
user_proxy = UserProxyAgent(
    name="user_proxy",
    human_input_mode="NEVER",  # No human input will be solicited
    max_consecutive_auto_reply=10,  # Maximum number of consecutive auto-replies
    code_execution_config={"work_dir": "coding"},  # Working directory for code execution
    llm_config=llm_config,  # LLM configuration for generating replies
)

# Initiate a conversation with a task description
user_proxy.initiate_chat(
    assistant,
    message="Please execute a python script that prints 10 dad jokes.",
)

