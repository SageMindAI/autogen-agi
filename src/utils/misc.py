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
from typing import Callable, List, Optional, Tuple

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
from llama_index.prompts.base import PromptTemplate
from llama_index.prompts.prompt_type import PromptType
from llama_index.indices.postprocessor import LLMRerank
from llama_index.schema import BaseNode, NodeWithScore, MetadataMode
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
        base_path = os.sep.join(path_parts[: target_index + 1])
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


def light_llm3_wrapper(query, kwargs={}):
    kwargs["model"] = kwargs.get("model", "gpt-3.5-turbo-1106")
    kwargs["temperature"] = kwargs.get("temperature", 0.1)
    llm3 = OpenAI(**kwargs)
    return light_llm_wrapper(llm3, query)


def light_llm4_wrapper(query, kwargs={}):
    kwargs["model"] = kwargs.get("model", "gpt-4-1106-preview")
    kwargs["temperature"] = kwargs.get("temperature", 0.1)
    llm4 = OpenAI(**kwargs)
    return light_llm_wrapper(llm4, query)


def light_gpt_wrapper_autogen(client, query, return_json=False, system_message=None):
    system_message = (
        system_message
        or "You are a helpful assistant. A user will ask a question, and you should provide an answer. ONLY return the answer, and nothing more."
    )

    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": query},
    ]

    create_kwargs = {
        "messages": messages,
    }

    if return_json:
        create_kwargs["response_format"] = {"type": "json_object"}

    response = client.create(**create_kwargs)

    if return_json:
        response = client.extract_text_or_function_call(response)
        response = autogen.ConversableAgent._format_json_str(response)
        response = json.loads(response)

    return response


def light_gpt3_wrapper_autogen(query, return_json=False, system_message=None):
    client = OpenAIWrapper(config_list=config_list3)
    return light_gpt_wrapper_autogen(client, query, return_json, system_message)


def light_gpt4_wrapper_autogen(query, return_json=False, system_message=None):
    client = OpenAIWrapper(config_list=config_list4)
    return light_gpt_wrapper_autogen(client, query, return_json, system_message)


def map_directory_to_json(dir_path):
    def dir_to_dict(path):
        dir_dict = {"name": os.path.basename(path)}
        if os.path.isdir(path):
            dir_dict["type"] = "directory"
            dir_dict["children"] = [
                dir_to_dict(os.path.join(path, x)) for x in os.listdir(path)
            ]
        else:
            dir_dict["type"] = "file"
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
                response_format={"type": "json_object"},
                messages=[{"role": "user", "content": FIX_JSON_PROMPT}],
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

CHOICE_SELECT_PROMPT_TMPL = (
    "A list of documents is shown below. Each document has a number next to it along "
    "with a summary of the document. A question is also provided. \n"
    "Respond with the numbers of the documents "
    "you should consult to answer the question, in order of relevance, as well \n"
    "as the relevance score. The relevance score is a number from 1-10 based on "
    "how relevant you think the document is to the question.\n"
    "Do not include any documents that are not relevant to the question. \n"
    "Example format: \n"
    "Document 1:\n<summary of document 1>\n\n"
    "Document 2:\n<summary of document 2>\n\n"
    "...\n\n"
    "Document 10:\n<summary of document 10>\n\n"
    "Question: <question>\n"
    "Answer:\n"
    "Doc: 9, Relevance: 7\n"
    "Doc: 3, Relevance: 4\n"
    "Doc: 7, Relevance: 3\n\n"
    "Let's try this now: \n\n"
    "{context_str}\n"
    "Question: {query_str}\n"
    "Answer:\n"
)
CHOICE_SELECT_PROMPT = PromptTemplate(
    CHOICE_SELECT_PROMPT_TMPL, prompt_type=PromptType.CHOICE_SELECT
)


class BetterLLMRerank(LLMRerank):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.choice_select_prompt = CHOICE_SELECT_PROMPT

    def _postprocess_nodes(
            self,
            nodes: List[NodeWithScore],
            query_bundle: Optional[QueryBundle] = None,
        ) -> List[NodeWithScore]:
            if query_bundle is None:
                raise ValueError("Query bundle must be provided.")
            initial_results: List[NodeWithScore] = []
            for idx in range(0, len(nodes), self.choice_batch_size):
                nodes_batch = [
                    node.node for node in nodes[idx : idx + self.choice_batch_size]
                ]

                query_str = query_bundle.query_str
                fmt_batch_str = self._format_node_batch_fn(nodes_batch)
                # call each batch independently
                print("FMAT_BATCH_STR: ", fmt_batch_str)
                print("QUERY_STR: ", query_str)
                raw_response = self.service_context.llm_predictor.predict(
                    self.choice_select_prompt,
                    context_str=fmt_batch_str,
                    query_str=query_str,
                )
                print("RAW_RESPONSE: ", raw_response)

                raw_choices, relevances = self._parse_choice_select_answer_fn(
                    raw_response, len(nodes_batch)
                )
                choice_idxs = [int(choice) - 1 for choice in raw_choices]
                choice_nodes = [nodes_batch[idx] for idx in choice_idxs]
                relevances = relevances or [1.0 for _ in choice_nodes]
                initial_results.extend(
                    [
                        NodeWithScore(node=node, score=relevance)
                        for node, relevance in zip(choice_nodes, relevances)
                    ]
                )

            return sorted(initial_results, key=lambda x: x.score or 0.0, reverse=True)[
                : self.top_n
            ]
    

    def _parse_choice_select_answer_fn(
        answer: str, num_choices: int, raise_error: bool = False
    ) -> Tuple[List[int], List[float]]:
        """Default parse choice select answer function."""
        answer_lines = answer.split("\n")
        answer_nums = []
        answer_relevances = []
        for answer_line in answer_lines:
            line_tokens = answer_line.split(",")
            if len(line_tokens) != 2:
                if not raise_error:
                    continue
                else:
                    raise ValueError(
                        f"Invalid answer line: {answer_line}. "
                        "Answer line must be of the form: "
                        "answer_num: <int>, answer_relevance: <float>"
                    )
            print("LINE_TOKENS: ", line_tokens)
            answer_num = int(line_tokens[0].split(":")[1].strip())
            if answer_num > num_choices:
                continue
            answer_nums.append(answer_num)
            answer_relevances.append(float(line_tokens[1].split(":")[1].strip()))
        return answer_nums, answer_relevances


    def _format_node_batch_fn(
        summary_nodes: List[BaseNode],
    ) -> str:
        """Default format node batch function.

        Assign each summary node a number, and format the batch of nodes.

        """
        fmt_node_txts = []
        for idx in range(len(summary_nodes)):
            number = idx + 1
            fmt_node_txts.append(
                f"Document {number}:\n"
                f"{summary_nodes[idx].get_content(metadata_mode=MetadataMode.LLM)}"
            )
        return "\n\n".join(fmt_node_txts)




def get_informed_answer(
    question,
    domain,
    domain_description,
    docs_dir,
    storage_dir,
    vector_top_k=50,
    reranker_top_n=15,
    rerank=False,
):
    llm3_general = OpenAI(model="gpt-3.5-turbo-1106", temperature=0.1)
    llm3_synthesizer = OpenAI(
        model="gpt-3.5-turbo-1106", temperature=0.1, max_tokens=1000
    )
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
        documents = SimpleDirectoryReader(
            DOCS_DIR,
            recursive=True,
        ).load_data()
        index = VectorStoreIndex.from_documents(documents)
        print("CREATING INDEX AT: ", STORAGE_DIR)
        # store it for later
        index.storage_context.persist(persist_dir=STORAGE_DIR)
    else:
        # load the existing index
        print("LOADING INDEX AT: ", STORAGE_DIR)
        storage_context = StorageContext.from_defaults(persist_dir=STORAGE_DIR)
        index = load_index_from_storage(storage_context)

    def get_retrieved_nodes(query_str, vector_top_k=40, reranker_top_n=20, rerank=True):
        print(f"GETTING TOP NODES: {vector_top_k}")
        query_bundle = QueryBundle(query_str)
        # configure retriever
        retriever = VectorIndexRetriever(
            index=index,
            similarity_top_k=vector_top_k,
        )
        retrieved_nodes = retriever.retrieve(query_bundle)

        # print("ORIGINAL NODES:\n\n")
        # for node in retrieved_nodes:
        #     file_info = node.metadata.get('file_name') or node.metadata.get('file_path')
        #     print(f"FILE INFO: {file_info}")
        #     print(node)
        #     print("\n\n")


        if rerank:
            print(f"GETTING RERANKED NODES: {reranker_top_n}")
            # configure reranker
            reranker = BetterLLMRerank(
                choice_batch_size=5,
                top_n=reranker_top_n,
                service_context=reranker_context,
            )
            # print("RERANKING NODES...")
            retrieved_nodes = reranker.postprocess_nodes(retrieved_nodes, query_bundle)

            print("RERANKED NODES:\n\n")
            for node in retrieved_nodes:
                file_info = node.metadata.get('file_name') or node.metadata.get('file_path')
                print(f"FILE INFO: {file_info}")
                print(node)
                print("\n\n")

        return retrieved_nodes

    nodes = None

    while nodes is None:
        try:
            nodes = get_retrieved_nodes(question, vector_top_k, reranker_top_n, rerank)
        except IndexError as e:
            print("INDEX ERROR: ", e)
            continue

    # print("NODES:\n\n")
    # for node in nodes:
    #     print(node)
    #     print("\n\n")

    print("\nRAG QUESTION:\n\n")
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
