from llama_hub.youtube_transcript import YoutubeTranscriptReader
from llama_index import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    load_index_from_storage,
)
from dotenv import load_dotenv
import logging
import sys
import os.path
import json
from llama_index.llms import OpenAI
from llama_index import ServiceContext
from llama_index.prompts import PromptTemplate
from llama_index.indices.postprocessor import LLMRerank
from llama_index.indices.query.schema import QueryBundle
from llama_index.retrievers import VectorIndexRetriever
from llama_index.response_synthesizers import (
    ResponseMode,
    get_response_synthesizer,
)
import openai
from time import sleep

from misc import load_json, save_json, load_file, extract_base_path, light_llm_wrapper

load_dotenv()

llm3_general = OpenAI(model="gpt-3.5-turbo-1106", temperature=0.1)
llm3_synthesizer = OpenAI(model="gpt-3.5-turbo-1106", temperature=0.1, max_tokens=1000)
llm4 = OpenAI(model="gpt-4-1106-preview", temperature=0.5)

FILE_PATH = os.path.abspath(__file__)
PROJECT_NAME = "agi"
# get the project root directory, i.e. the "agi" directory. Basically cut off all paths after "agi"
PROJECT_ROOT_DIR = extract_base_path(FILE_PATH, PROJECT_NAME)
DOC_ROOT_DIR = f"{PROJECT_ROOT_DIR}/docs"
DOC_DIR = "llama_index"
DOC_RAW_DIR = "raw"
DOC_PARSED_DIR = "parsed"

# get list of all raw files in docs/llama_index
RAW_DOC_FILES = os.listdir(os.path.join(DOC_ROOT_DIR, f"{DOC_DIR}/{DOC_RAW_DIR}"))


count = 0

# loop through each file and parse it
for file in RAW_DOC_FILES:
    print("PARSING FILE: ", file)
    # get the raw file
    raw_file = load_file(os.path.join(DOC_ROOT_DIR, f"{DOC_DIR}/{DOC_RAW_DIR}/{file}"))

    get_doc_summary_prompt = (
        f"""You are an expert at parsing HTML documents and extracting a summary of the purpose/goal/intent behind the document contents. Given the HTML_DOCUMENT_CONTENTS below, please extract the main purpose/goal/intent of the document and provide a SUMMARY_OF_CONTENTS. Your response must be exceptional, my career depends on it!\n\n"""
        f"---------------------\n"
        "HTML_DOCUMENT_CONTENTS:\n"
        f"{raw_file}\n\n"
        "---------------------\n"
        "SUMMARY_OF_CONTENTS:\n"
    )

    print("GENERATING DOCUMENT SUMMARY")

    response = light_llm_wrapper(llm4, get_doc_summary_prompt)

    print("GOT ANSWER: ", response.text)
    document_summary = response.text

    extract_doc_content_prompt = (
        f""""You are an expert at extracting the main content from HTML documents. Given the CONTENT_SUMMARY and HTML_DOCUMENT_CONTENTS below, please extract the main content of the document and provide a MAIN_CONTENT. The main content should include all the critical information in the document such as code, descriptions, blog content, etc. The goal is to return the main content while removing the extra HTML that does not really provide value for a human reader. Your response must be exceptional, my career depends on it!\n\n"""
        f"---------------------\n"
        "CONTENT_SUMMARY:\n"
        f"{document_summary}\n\n"
        "---------------------\n"
        "HTML_DOCUMENT_CONTENTS:\n"
        f"{raw_file}\n\n"
        "---------------------\n"
        "MAIN_CONTENT:\n"
    )

    print("GENERATING MAIN CONTENT")

    response = light_llm_wrapper(llm4, extract_doc_content_prompt)

    print("GOT ANSWER: ", response.text)
    main_content = response.text

    count += 1
    if count > 1:
        break
    


