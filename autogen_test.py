import autogen
import os
from dotenv import load_dotenv

from llama_autogen_retriever import LlamaRetrieveUserProxyAgent

load_dotenv()

config_list3 = [
    {
        'model': 'gpt-3.5-turbo-1106',
        'api_key': os.environ["OPENAI_API_KEY"],
    }
]

llm_config3={
    "request_timeout": 600,
    "seed": 42,
    "config_list": config_list3,
    "temperature": 0,
}

config_list4 = [
    {
        'model': 'gpt-4-1106-preview',
        'api_key': os.environ["OPENAI_API_KEY"],
    }
]

llm_config4={
    "request_timeout": 600,
    "seed": 42,
    "config_list": config_list4,
    "temperature": 0,
}


STORAGE_DIR = "./storage/autogen/"
DOCS_DIR = "docs/autogen"

assistant_autogen_expert = autogen.AssistantAgent(
    name="Autogen Expert", 
    llm_config=llm_config4,
    system_message="You are an expert at the python library autogen. Please use your knowledge to help the user solve the problem."
)

# Function to start the chat and solve a problem using RAG with custom retrieval
def solve_problem_with_agents(problem):
    # Reset the assistant agent
    assistant_autogen_expert.reset()

    # Create an instance of your custom RetrieveUserProxyAgent
    my_retrieve_agent = LlamaRetrieveUserProxyAgent(
        name="MyRetrieveAgent",
        retrieve_config={
            "docs_path": DOCS_DIR,
            "storage_path": STORAGE_DIR,
            # "task": "qa",
        },
        human_input_mode="ALWAYS",
        max_consecutive_auto_reply=5,
        is_termination_msg=lambda x: x.get("content", "").rstrip().endswith("TERMINATE"),
        code_execution_config={"work_dir": "autogen_test_workdir"},
        llm_config=llm_config4,
    )

    # Initiate chat with the assistant by providing the problem
    my_retrieve_agent.initiate_chat(assistant_autogen_expert, problem=problem)

    # Retrieve the messages from the assistant
    messages = my_retrieve_agent.chat_messages
    messages = [messages[k] for k in messages.keys()][0]
    answers = [m["content"] for m in messages if m["role"] == "assistant"]

    # Print the answers
    for answer in answers:
        print("Assistant:", answer)


# Example usage
if __name__ == "__main__":
    # Example problem to solve
    problem = "Please give a detailed description of autogen package."
    solve_problem_with_agents(problem)
