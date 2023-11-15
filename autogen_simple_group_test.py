import autogen
from autogen import GroupChat, GroupChatManager

from better_group_chat import BetterGroupChat, BetterGroupChatManager
from src.agent_prompts import DEFAULT_CODING_AGENT_SYSTEM_MESSAGE

from dotenv import load_dotenv

load_dotenv()

import os
import json
import copy

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

work_dir = "TEST_DIR"


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

# assistant1 = autogen.AssistantAgent(
#     name="Jokester",
#     system_message="You are an expert at writing jokes and NOTHING ELSE. Secifically you DO NOT write code.",
#     llm_config=llm_config4,
#     # is_termination_msg=lambda x: get_end_intent(x) == "end",
# )

# assistant2 = autogen.AssistantAgent(
#     name="Critiquer",
#     system_message="You are an expert at critiquing jokes and NOTHING ELSE. Secifically you DO NOT write code.",
#     llm_config=llm_config4,
#     # is_termination_msg=lambda x: get_end_intent(x) == "end",
# )

assistant1 = autogen.AssistantAgent(
    name="CodeReviewer",
    system_message="You are an expert at reviewing code and suggesting improvements.",
    llm_config=llm_config4,
    # is_termination_msg=lambda x: get_end_intent(x) == "end",
)

assistant2 = autogen.AssistantAgent(
    name="AgentAwarenessExpert",
    system_message="""You are an expert at understanding the nature of the agents in the team. Your job is to help guide agents in their task, making sure that suggested actions align with your knowledge. Specifically, you know that:
    - AGENTS: Agents are Large Language Models (LLMs). The most important thing to understand about Large Language Models (LLMs) to get the most leverage out of them is their latent space and associative nature. LLMs embed knowledge, abilities, and concepts ranging from reasoning to planning, and even theory of mind. This collection of abilities and content is referred to as the latent space. Activating the latent space of an LLM requires the correct series of words as inputs, creating a useful internal state of the neural network. This process is similar to how the right cues can prime a human mind to think in a certain way. By understanding and utilizing this associative nature and latent space, you can effectively leverage LLMs for various applications​​.
    - CODE EXECUTION: Some agents have the ability to suggest/write code, while others have the ability to execute code. A single agent may or may not have both of these abilities.
    - READING FILES: Agents cannot "read" (i.e know the contents of) a file unless the file contents are printed to the console and added to the agent conversation history. When analyzing/evaluating code (or any other file), it is IMPORTANT to actually print the content of the file to the console and add it to the agent conversation history. Otherwise, the agent will not be able to access the file contents.
    - CONTEXT KNOWLEDGE: Context knowledge is not accessible to agents unless it is explicitly added to the agent conversation history, UNLESS the agent specifically has functionality to access outside context.
    - AGENT COUNCIL: The agents in a team are guided by an "Agent Council" that is responsible for deciding which agent should act next. The council may also give input into what action the agent should take.
    - FUNCTION CALLING: Some agents have specific functions registered to them. Each registered function has a name, description, and arguments. Agents have been trained to detect when it is appropriate to "call" one of their registered functions. When an agents "calls" a function, they will respond with a JSON object containing the function name and its arguments. Once this message has been sent, the Agent Council will detect which agent has the capability of executing this function. The agent that executes the function may or may not be the same agent that called the function.
    """,
    llm_config=llm_config4,
    # is_termination_msg=lambda x: get_end_intent(x) == "end",
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
                            "description": "The absolute or relative path to the file",
                        },
                    },
                    "required": ["file_path"],
                },
            },
]

def read_file(file_path):
    with open(file_path, 'r') as f:
        return f.read()
    
coder_llm_config = copy.deepcopy(llm_config4)
coder_llm_config["functions"] = functions

assistant3 = autogen.AssistantAgent(
    name="Coder",
    # Clone llm_config4 and update it with the functions
    llm_config = coder_llm_config,
    # llm_config=llm_config4.update({"functions": functions}),
    # llm_config=llm_config4,
    system_message=DEFAULT_CODING_AGENT_SYSTEM_MESSAGE.format(work_dir=work_dir),
    # code_execution_config={"last_n_messages": 5, "work_dir": 'TEST_DIR'},
    # is_termination_msg=lambda x: get_end_intent(x) == "end",
    function_map={"read_file": read_file},
)

assistant4 = autogen.AssistantAgent(
    name="Executor",
    system_message="You are an expert at executing code and NOTHING ELSE. Secifically you DO NOT write code. If any other agents provide a code block for you to execute, you must execute it and provide the output.",
    llm_config=llm_config4,
    code_execution_config={"last_n_messages": 5, "work_dir": work_dir},
    # is_termination_msg=lambda x: get_end_intent(x) == "end",
)

groupchat = BetterGroupChat(agents=[user_proxy, assistant1, assistant2, assistant3, assistant4], messages=[], max_round=100, persona_discussion=True, inject_persona_discussion=True)
manager = BetterGroupChatManager(groupchat=groupchat, llm_config=llm_config4)

# user_proxy.initiate_chat(manager, message="Please come up with a list of 10 excellent jokes for a standup comedy routine. They must appeal to an adult AND child audience. When you are satisfied that the jokes are sufficient, please provide a python script that will output one of the jokes at random when it is run.")

user_proxy.initiate_chat(manager, message="I have a python file that I need cleaning up. The file is located at ../better_group_chat.py. Please clean up the file apply best code practices and good commenting without changing the functionality. When you're done please save the file for me to review.")

# TODO: Add ability for the AGENT_COUNCIL to decide if a new agent should be created and added to the groupchat.

# TODO: Add function to directly "save" files
# TODO: get a better understanding of how agents currently execute and save files
# TODO: Think of ways to give agents a better understanding of their capabilities and limitations