from dotenv import load_dotenv
import logging
import sys
import os.path
import json
import argparse
import openai
from time import sleep

from llama_index import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    load_index_from_storage,
    ServiceContext,
    llms,
    prompts,
    indices,
    retrievers,
    response_synthesizers,
)

from llama_hub.youtube_transcript import YoutubeTranscriptReader
from llama_index.llms import OpenAI

from llama_index.indices.postprocessor import LLMRerank
from llama_index.indices.query.schema import QueryBundle
from llama_index.retrievers import VectorIndexRetriever
from llama_index.response_synthesizers import (
    ResponseMode,
    get_response_synthesizer,
)
from llama_index.prompts import PromptTemplate


llm = OpenAI(model="gpt-3.5-turbo-0613")

load_dotenv()

STORAGE_DIR = "./storage/autogen/"
DOCS_DIR = "docs/autogen"

# logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
# logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))


llm3_general = OpenAI(model="gpt-3.5-turbo-1106", temperature=0.1)
llm3_synthesizer = OpenAI(model="gpt-3.5-turbo-1106", temperature=0.1, max_tokens=1000)
llm4 = OpenAI(model="gpt-4-1106-preview", temperature=0.5)


service_context3_general = ServiceContext.from_defaults(llm=llm3_general)
service_context3_synthesizer = ServiceContext.from_defaults(llm=llm3_synthesizer)
service_context4 = ServiceContext.from_defaults(llm=llm4)

reranker_context = service_context3_general
response_synthesizer_context = service_context4


# check if storage already exists
if not os.path.exists(STORAGE_DIR):
    # load the documents and create the index
    documents = SimpleDirectoryReader(DOCS_DIR).load_data()
    index = VectorStoreIndex.from_documents(documents)
    # store it for later
    index.storage_context.persist(persist_dir=STORAGE_DIR)
else:
    # load the existing index
    storage_context = StorageContext.from_defaults(persist_dir=STORAGE_DIR)
    index = load_index_from_storage(storage_context)


def get_retrieved_nodes(query_str, vector_top_k=10, reranker_top_n=4, rerank=True):
    print(f"GETTING TOP NODES: {vector_top_k}")
    print(f"GETTING RERANKED NODES: {reranker_top_n}")
    query_bundle = QueryBundle(query_str)
    # configure retriever
    retriever = VectorIndexRetriever(
        index=index,
        similarity_top_k=vector_top_k,
    )
    retrieved_nodes = retriever.retrieve(query_bundle)

    # print("RETRIEVED NODES:\n\n")
    # for node in retrieved_nodes:
    #     print(node)
    #     print("\n\n")

    if rerank:
        # configure reranker
        reranker = LLMRerank(
            choice_batch_size=5,
            top_n=reranker_top_n,
            service_context=reranker_context,
        )
        retrieved_nodes = reranker.postprocess_nodes(retrieved_nodes, query_bundle)

    # print("RERANKED NODES:\n\n")
    # for node in retrieved_nodes:
    #     print(node)
    #     print("\n\n")

    return retrieved_nodes


qa_prompt_tmpl_str = (
    f"You are an expert at the autogen python library. Please use the provided RELEVANT_CONTEXT to ANSWER the given QUESTION.\n\n"
    "Your answer must be that of an elite expert. Please! My career depends on it!!\n"
    "RELEVANT_CONTEXT:\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "QUESTION: {query_str}\n"
    "ANSWER: "
)
text_qa_template = PromptTemplate(qa_prompt_tmpl_str)


response_synthesizer = get_response_synthesizer(
    response_mode=ResponseMode.COMPACT,
    text_qa_template=text_qa_template,
    # refine_template=refine_template,
    service_context=response_synthesizer_context,
)

def get_answer(question):
    nodes = get_retrieved_nodes(question, vector_top_k=20, reranker_top_n=7, rerank=False)

    # print("NODES:\n\n")
    # for node in nodes:
    #     print(node)
    #     print("\n\n")

    print("\n\nQUESTION:\n\n")
    print(question)

    response = response_synthesizer.synthesize(question, nodes=nodes)
    # print("LENGTH: ", len(response.source_nodes))
    return response



def main():
    # Set up argument parsing
    # parser = argparse.ArgumentParser(description="Ask the llama_expert a question.")
    # parser.add_argument("question", help="Your question")

    # # Parse arguments
    # args = parser.parse_args()

    question = """
   Given the AUTOGEN_CODE below, how can I update it such that the autogen expert can access relevant docs (i.e. a RAG implementation)? I want to specify a local directory that it will use for retrieval

    AUTOGEN_CODE:
    ```
import autogen
import os

from dotenv import load_dotenv

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


assistant_autogen_expert = autogen.AssistantAgent(
    name="Autogen Expert", 
    llm_config=llm_config4,
    system_message="You are an expert at the python library autogen. Please use your knowledge to help the user solve the problem."
)

user_proxy = autogen.UserProxyAgent(
    name="User",
    human_input_mode="ALWAYS",
    max_consecutive_auto_reply=5,
    is_termination_msg=lambda x: x.get("content", "").rstrip().endswith("TERMINATE"),
    code_execution_config={"work_dir": "autogen_test_workdir"},
    llm_config=llm_config4,
    system_message=\"""Reply TERMINATE if the task has been solved at full satisfaction.
Otherwise, reply CONTINUE, or the reason why the task is not solved yet.\"""
)


# Function to start the chat and solve a problem
def solve_problem_with_agents(problem):
    # Reset the assistant agent
    assistant_autogen_expert.reset()

    # Initiate chat with the assistant by providing the problem
    user_proxy.initiate_chat(assistant_autogen_expert, message=problem)

    # Retrieve the messages from the assistant
    messages = assistant_autogen_expert.chat_messages
    messages = [messages[k] for k in messages.keys()][0]
    answers = [m["content"] for m in messages if m["role"] == "assistant"]

    # Print the answers
    for answer in answers:
        print("Assistant:", answer)

# Example usage
if __name__ == "__main__":
    # Example problem to solve
    problem = "What is autogen?"
    solve_problem_with_agents(problem)
    ```
"""

    # question = "What is autogen?"
    answer = None
    while answer is None or answer.response == "":
        try:
            answer = get_answer(question)
        except openai.RateLimitError as e:
            print("RATE LIMIT ERROR: ", e)
            sleep(5)
            continue
        except IndexError as e:
            print("INDEX ERROR: ", e)
            continue

    print("GOT ANSWER: ", answer.response)

if __name__ == '__main__':
    main()