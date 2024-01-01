"""
This is a basic autogen example.
"""

# filename: autonomous_agents_integration.py
from autogen import AssistantAgent, UserProxyAgent, config_list_from_json
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Load LLM configuration from environment or a file
config_list = config_list_from_json(env_or_file="OAI_CONFIG_LIST.json")

config_list_mistral = [config for config in config_list if config["model"] == "ollama/mistral"]

config_list_deepseek_coder = [config for config in config_list if config["model"] == "ollama/deepseek-coder:6.7b"]


if len(config_list_mistral) > 0:
    llm_config_user_proxy = {"config_list": config_list_mistral}
else:
    llm_config_user_proxy = {"config_list": config_list}

if len(config_list_deepseek_coder) > 0:
    llm_config_assistant = {"config_list": config_list_deepseek_coder}
else:
    llm_config_assistant = {"config_list": config_list}


print("llm_config_user_proxy:", llm_config_user_proxy)
print("llm_config_assistant:", llm_config_assistant)


# Create an AssistantAgent instance
assistant = AssistantAgent(
    name="assistant",
    llm_config=llm_config_assistant,
)

# Create a UserProxyAgent instance with autonomous settings
user_proxy = UserProxyAgent(
    name="user_proxy",
    human_input_mode="NEVER",  # No human input will be solicited
    max_consecutive_auto_reply=10,  # Maximum number of consecutive auto-replies
    code_execution_config={"work_dir": "working"},  # Working directory for code execution
    llm_config=llm_config_user_proxy,  # LLM configuration for generating replies
)

# Initiate a conversation with a task description
user_proxy.initiate_chat(
    assistant,
    message="Please execute a python script that prints 10 dad jokes.",
)

