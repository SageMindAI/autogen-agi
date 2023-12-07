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
from utils.rag_tools import get_informed_answer


load_dotenv()

STORAGE_DIR = "./storage"
DOCS_DIR = "./docs"

domain = "autogen"
domain_description = "autogen autonomous agent framework"


def main():
    question = "What is autogen?"
    answer = None
    while answer is None or answer.response == "":
        try:
            answer = get_informed_answer(
                question,
                docs_dir=DOCS_DIR,
                storage_dir=STORAGE_DIR,
                domain=domain,
                domain_description=domain_description,
                vector_top_k=5,
                reranker_top_n=3,
                rerank=True,
                fusion=True,
            )
        except openai.RateLimitError as e:
            print("RATE LIMIT ERROR: ", e)
            sleep(5)
            continue

    print("GOT ANSWER: ", answer.response)


if __name__ == "__main__":
    main()
