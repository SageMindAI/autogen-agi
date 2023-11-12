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

retrieval_model = "gpt-4-1106-preview"


STORAGE_DIR = "./storage/autogen/"
DOCS_DIR = "docs/autogen"


# Function to start the chat and solve a problem using RAG with custom retrieval
def solve_problem_with_agents(problem):

    assistant_autogen_expert = autogen.AssistantAgent(
        name="Autogen Expert", 
        llm_config=llm_config4,
        system_message="You are an expert at the python library autogen. Please use your knowledge to help the user solve the problem. If your context does not have the answer, please use the Autogen_RAG_User agent to retrieve the needed context, or reply 'UPDATE CONTEXT'."
    )
    # Reset the assistant agent
    assistant_autogen_expert.reset()

    # Create an instance of your custom RetrieveUserProxyAgent
    my_retrieve_agent = LlamaRetrieveUserProxyAgent(
        name="Autogen_RAG_User",
        retrieve_config={
            "docs_path": DOCS_DIR,
            "storage_path": STORAGE_DIR,
            "model": retrieval_model,
            # "task": "qa",
        },
        human_input_mode="ALWAYS",
        max_consecutive_auto_reply=1,
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
    problem = "I'm really confused about the RetrieveUserProxyAgent agent. When I use it to get answers, it only retrieves documents after the first message. Is there a setting I can change to make it retrieve documents after each message? Please give me a complete code example."
    solve_problem_with_agents(problem)
