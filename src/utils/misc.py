
"""
DESCRIPTION: This file contains miscellaneous functions that are used in multiple scripts.
"""

import os
import json
import openai
import autogen
from autogen import OpenAIWrapper
from time import sleep

from llama_index.llms import OpenAI

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

from llama_index.indices.postprocessor import LLMRerank
from llama_index.indices.query.schema import QueryBundle
from llama_index.retrievers import VectorIndexRetriever
from llama_index.response_synthesizers import (
    ResponseMode,
    get_response_synthesizer,
)
from llama_index.prompts import PromptTemplate

from dotenv import load_dotenv

load_dotenv()


config_list3 = [
    {
        "model": "gpt-3.5-turbo-1106",
        "api_key": os.environ["OPENAI_API_KEY"],
    }
]

config_list4 = [
    {
        "model": "gpt-4-1106-preview",
        "api_key": os.environ["OPENAI_API_KEY"],
    }
]


def load_json(filename):
    if os.path.isfile(filename) and os.path.getsize(filename) > 0:
        with open(filename, "r") as file:
            return json.load(file)
    else:
        return []


def save_json(data, filename):
    with open(filename, "w") as file:
        json.dump(data, file, indent=4, ensure_ascii=False)



def load_file(filename):
    with open(filename) as f:
        file = f.read()

        if len(file) == 0:
            raise ValueError("filename cannot be empty.")

        return file
    
def extract_base_path(full_path, target_directory):
    """
    Extracts the base path up to and including the target directory from the given full path.

    :param full_path: The complete file path.
    :param target_directory: The target directory to which the path should be truncated.
    :return: The base path up to and including the target directory, or None if the target directory is not in the path.
    """
    path_parts = full_path.split(os.sep)
    if target_directory in path_parts:
        target_index = path_parts.index(target_directory)
        base_path = os.sep.join(path_parts[:target_index + 1])
        return base_path
    else:
        return None
    
def light_llm_wrapper(llm, query):
    response = None
    while response is None or response.text == "":
        try:
            response = llm.complete(query)
        except openai.RateLimitError as e:
            print("RATE LIMIT ERROR: ", e)
            sleep(5)
            continue
        except IndexError as e:
            print("INDEX ERROR: ", e)
            continue

    return response

def light_llm4_wrapper(query, kwargs={}):
    kwargs["model"] = kwargs.get("model", "gpt-4-1106-preview")
    kwargs["temperature"] = kwargs.get("temperature", 0.1)
    llm4 = OpenAI(**kwargs)
    return light_llm_wrapper(llm4, query)

def light_gpt4_wrapper_autogen(query, return_json=False, system_message=None):
    system_message = system_message or "You are a helpful assistant. A user will ask a question, and you should provide an answer. ONLY return the answer, and nothing more."

    messages = [
        {
            "role": "system",
            "content": system_message
        },
        {
            "role": "user",
            "content": query
        }
    ]

    create_kwargs = {
        "messages": messages,
    }

    if return_json:
        create_kwargs["response_format"] = { "type": "json_object" }

    client = OpenAIWrapper(config_list=config_list4)

    response = client.create(**create_kwargs)

    if return_json:
        response = client.extract_text_or_function_call(response)
        response = autogen.ConversableAgent._format_json_str(response)
        response = json.loads(response)

    return response
    

def map_directory_to_json(dir_path):
    def dir_to_dict(path):
        dir_dict = {'name': os.path.basename(path)}
        if os.path.isdir(path):
            dir_dict['type'] = 'directory'
            dir_dict['children'] = [dir_to_dict(os.path.join(path, x)) for x in os.listdir(path)]
        else:
            dir_dict['type'] = 'file'
        return dir_dict

    root_structure = dir_to_dict(dir_path)
    return json.dumps(root_structure, indent=4)


def remove_substring(string, substring):
    return string.replace(substring, "")


def fix_broken_json(potential_json, max_attempts=5):
    FIX_JSON_PROMPT = f"""You are a helpful assistant. A user will ask a question, and you should provide an answer.
    ONLY return the answer, and nothing more.
    
    Given the following potential JSON text, please fix any broken JSON syntax. Do NOT change the text itself. ONLY respond with the fixed JSON.

    Potential JSON:
    ---
    {potential_json}
    ---

    Response:
    """

    attempts = 0
    error = None
    while attempts < max_attempts:
        try:
            attempts += 1
            client = OpenAIWrapper(config_list=config_list3)
            response = client.create(
                response_format={ "type": "json_object" },
                messages=[{
                    "role": "user",
                    "content": FIX_JSON_PROMPT
                }]
            )
            response = client.extract_text_or_function_call(response)
            response = autogen.ConversableAgent._format_json_str(response)
            response = json.loads(response)
            return response
        except Exception as error:
            print("FIX ATTEMPT FAILED, TRYING AGAIN...", attempts)
            error = error

    raise error

def extract_json_response_oai_wrapper(message):
    client = OpenAIWrapper(config_list=config_list3)
    response = client.extract_text_or_function_call(message)
    response = autogen.ConversableAgent._format_json_str(response)
    try:
        response = json.loads(response)
    except Exception as error:
        response = fix_broken_json(response)

    return response


def extract_json_response(message):
    try:
        response = json.loads(message)
    except Exception as error:
        response = fix_broken_json(message)

    return response





def get_informed_answer(question, domain, domain_description, docs_dir, storage_dir, vector_top_k=20, reranker_top_n=7, rerank=False):


    llm3_general = OpenAI(model="gpt-3.5-turbo-1106", temperature=0.1)
    llm3_synthesizer = OpenAI(model="gpt-3.5-turbo-1106", temperature=0.1, max_tokens=1000)
    llm4 = OpenAI(model="gpt-4-1106-preview", temperature=0.5)


    service_context3_general = ServiceContext.from_defaults(llm=llm3_general)
    service_context3_synthesizer = ServiceContext.from_defaults(llm=llm3_synthesizer)
    service_context4 = ServiceContext.from_defaults(llm=llm4)

    reranker_context = service_context3_general
    response_synthesizer_context = service_context4

    STORAGE_DIR = f"{storage_dir}/{domain}"
    DOCS_DIR = f"{docs_dir}/{domain}"

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
            print(f"GETTING RERANKED NODES: {reranker_top_n}")
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


    nodes = get_retrieved_nodes(question, vector_top_k, reranker_top_n, rerank)

    # print("NODES:\n\n")
    # for node in nodes:
    #     print(node)
    #     print("\n\n")

    print("\n\nQUESTION:\n\n")
    print(question)


    qa_prompt_tmpl_str = (
        f"You are an expert at the following DOMAIN which is described in the DOMAIN_DESCRIPTION. Given the following DOMAIN_SPECIFIC_CONTEXT, please answer the QUESTION to the best of your ability. If the information required for the answer cannot be found in the DOMAIN_SPECIFIC_CONTEXT, then reply with 'DOMAIN CONTEXT NOT AVAILABLE'.\n\n"
        "Your answer must be that of an elite expert. Please! My career depends on it!!\n"
        "DOMAIN:\n"
        "---------------------\n"
        f"{domain}\n"
        "---------------------\n"
        "DOMAIN_DESCRIPTION:\n"
        "---------------------\n"
        f"{domain_description}\n"
        "---------------------\n"
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

    response = response_synthesizer.synthesize(question, nodes=nodes)
    # print("LENGTH: ", len(response.source_nodes))
    return response