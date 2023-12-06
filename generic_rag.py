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

STORAGE_DIR = "./storage/test/"
DOCS_DIR = "docs/test"

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
    print("CREATING INDEX")
    # load the documents and create the index
    documents = SimpleDirectoryReader(
        DOCS_DIR,
        recursive=True,
    ).load_data()
    print("DOCUMENTSSSS", documents)
    index = VectorStoreIndex.from_documents(documents)
    # store it for later
    index.storage_context.persist(persist_dir=STORAGE_DIR)
else:
    print("LOADING INDEX")
    # load the existing index
    storage_context = StorageContext.from_defaults(persist_dir=STORAGE_DIR)
    index = load_index_from_storage(storage_context)


def get_retrieved_nodes(query_str, vector_top_k=10, reranker_top_n=4, rerank=False):
    print(f"GETTING TOP NODES: {vector_top_k}")
    query_bundle = QueryBundle(query_str)
    # configure retriever
    retriever = VectorIndexRetriever(
        index=index,
        similarity_top_k=vector_top_k,
    )
    retrieved_nodes = retriever.retrieve(query_bundle)

    # print(f"ORIGINAL NODES: {len(retrieved_nodes)}\n\n")
    # for node in retrieved_nodes:
    #     file_info = node.metadata.get('file_name') or node.metadata.get('file_path')
    #     print(f"FILE INFO: {file_info}")
    #     print(node)
    #     print("\n\n")

    if rerank:
        print(f"GETTING RERANKED NODES: {reranker_top_n}")
        # configure reranker
        reranker = LLMRerank(
            choice_batch_size=5,
            top_n=reranker_top_n,
            service_context=reranker_context,
        )
        print("GOT RERANKER")
        retrieved_nodes = reranker.postprocess_nodes(retrieved_nodes, query_bundle)

        print(f"RERANKED NODES: {len(retrieved_nodes)}\n\n")
        for node in retrieved_nodes:
            file_info = node.metadata.get('file_name') or node.metadata.get('file_path')
            print(f"FILE INFO: {file_info}")
            print(node)
            print("\n\n")

    return retrieved_nodes


qa_prompt_tmpl_str = (
    f"You are a helpful assistant. Please use the provided RELEVANT_CONTEXT to ANSWER the given QUESTION.\n\n"
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
    response_mode=ResponseMode.TREE_SUMMARIZE,
    text_qa_template=text_qa_template,
    # refine_template=refine_template,
    service_context=response_synthesizer_context,
)


def get_answer(question):
    nodes = get_retrieved_nodes(
        question, vector_top_k=30, reranker_top_n=15, rerank=False
    )

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

    question = "What are these documents about?"
    answer = None
    while answer is None or answer.response == "":
        try:
            answer = get_answer(question)
        except openai.RateLimitError as e:
            print("RATE LIMIT ERROR: ", e)
            sleep(5)
            continue
        # except IndexError as e:
        #     print("INDEX ERROR: ", e)
        #     continue

    print("GOT ANSWER: ", answer.response)


if __name__ == "__main__":
    main()
