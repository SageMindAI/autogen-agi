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
from typing import Callable, List, Optional, Tuple, Any

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
from llama_index.prompts.prompt_type import PromptType
from llama_index.bridge.pydantic import BaseModel
from llama_index.prompts.base import BasePromptTemplate, PromptTemplate
from llama_index.indices.postprocessor import LLMRerank
from llama_index.schema import BaseNode, NodeWithScore, MetadataMode
from llama_index.indices.query.schema import QueryBundle
from llama_index.retrievers import VectorIndexRetriever
from llama_index.llm_predictor import LLMPredictor
from llama_index.response_synthesizers import (
    ResponseMode,
    get_response_synthesizer,
)
from llama_index.prompts import PromptTemplate
from pathlib import Path
from llama_index import download_loader
from llama_index.node_parser import LangchainNodeParser

from langchain.text_splitter import RecursiveCharacterTextSplitter, Language

from .misc import extract_json_response

import logging

from dotenv import load_dotenv

load_dotenv()


logger = logging.getLogger(__name__)


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


class JSONLLMPredictor(LLMPredictor):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def predict(
        self,
        prompt: BasePromptTemplate,
        output_cls: Optional[BaseModel] = None,
        **prompt_args: Any,
    ) -> str:
        """Predict."""
        self._log_template_data(prompt, **prompt_args)

        if output_cls is not None:
            output = self._run_program(output_cls, prompt, **prompt_args)
        elif self._llm.metadata.is_chat_model:
            messages = prompt.format_messages(llm=self._llm, **prompt_args)
            messages = self._extend_messages(messages)
            chat_response = self._llm.chat(messages)
            output = chat_response.message.content or ""
        else:
            formatted_prompt = prompt.format(llm=self._llm, **prompt_args)
            formatted_prompt = self._extend_prompt(formatted_prompt)
            response = self._llm.complete(formatted_prompt, return_json=True)
            output = response.text

        logger.debug(output)

        return output


CHOICE_SELECT_PROMPT_TMPL = (
    "A list of NUMBERED_DOCUMENTS is shown below. "
    "A QUESTION is also provided. \n"
    "Please give a detailed analysis comparing each document to the context of the QUESTION, talking through your thoughts step by step, and rate each document on a scale of 1-10 based on how relevant you think \n"
    "the DOCUMENT_CONTENT is to the context of the QUESTION.  \n"
    "Do not include any documents that are not relevant to the question. \n"
    "Your response must be a JSON object with the following format: \n"
    "{{\n"
    '    "answer": [\n'
    "        {{\n"
    '            "document_number": <int>,\n'
    '            "file_path": <string>,\n'
    '            "analysis_of_relevance": <string>\n'
    '            "rating": <float>\n'
    "        }},\n"
    "        ...\n"
    "    ]\n"
    "}}\n\n"
    "Example DOCUMENTS: \n"
    "------------------------------------------------------------\n"
    "DOCUMENT_NUMBER: 1\n"
    "DOCUMENT_CONTENT"
    "-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#\n"
    "file_path: <file_path>\n\n"
    "<document content>\n"
    "-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#\n"
    "------------------------------------------------------------\n"
    "DOCUMENT_NUMBER: 2\n"
    "DOCUMENT_CONTENT"
    "-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#\n"
    "file_path: <file_path>\n\n"
    "<document content>\n"
    "-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#\n"
    "...\n\n"
    "------------------------------------------------------------\n"
    "DOCUMENT_NUMBER: 10\n"
    "DOCUMENT_CONTENT"
    "-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#\n"
    "file_path: <file_path>\n\n"
    "<document content>\n"
    "-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#\n"
    "Example QUESTION: <question>\n"
    "Example Response:\n"
    "{{\n"
    '    "answer": [\n'
    "        {{\n"
    '            "document_number": 1,\n'
    '            "file_path": <file_path_of_doc_1>,\n'
    '            "analysis_of_relevance": <detailed_analysis_1>,\n'
    '            "rating": 7\n'
    "        }},\n"
    "        {{\n"
    '            "document_number": 2,\n'
    '            "file_path": <file_path_of_doc_2>,\n'
    '            "analysis_of_relevance": <detailed_analysis_2>,\n'
    '            "rating": 4\n'
    "        }},\n"
    "        ...\n"
    "        {{\n"
    '            "document_number": 10,\n'
    '            "file_path": <file_path_of_doc_10>,\n'
    '            "analysis_of_relevance": <detailed_analysis_10>,\n'
    '            "rating": 4\n'
    "        }},\n"
    "    ]\n"
    "}}\n\n"
    "IMPORTANT: MAKE SURE the 'document_number' value in your response corresponds to the correct DOCUMENT_NUMBER. \n\n"
    "DOCUMENTS:\n"
    "{context_str}\n"
    "QUESTION: {query_str}\n"
    "Response:\n"
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
            print("RERANKING BATCH...")
            raw_response = self.service_context.llm_predictor.predict(
                self.choice_select_prompt,
                context_str=fmt_batch_str,
                query_str=query_str,
            )
            json_response = extract_json_response(raw_response)

            raw_choices, relevances = self._parse_choice_select_answer_fn(
                json_response, len(nodes_batch)
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
        self, answer: dict, num_choices: int, raise_error: bool = False
    ) -> Tuple[List[int], List[float]]:
        """JSON parse choice select answer function."""
        answer_lines = answer["answer"]
        answer_nums = []
        answer_relevances = []
        for answer_line in answer_lines:
            answer_num = int(answer_line["document_number"])
            # if answer_num > num_choices:
            #     continue
            answer_nums.append(answer_num)
            answer_relevances.append(float(answer_line["rating"]))
        return answer_nums, answer_relevances

    def _format_node_batch_fn(
        self,
        summary_nodes: List[BaseNode],
    ) -> str:
        """Default format node batch function.

        Assign each summary node a number, and format the batch of nodes.

        """
        fmt_node_txts = []
        for idx in range(len(summary_nodes)):
            number = idx + 1
            fmt_node_txts.append(
                "------------------------------------------------------------\n"
                f"DOCUMENT_NUMBER: {number}\n"
                "DOCUMENT_CONTENT\n"
                "-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#\n"
                f"{summary_nodes[idx].get_content(metadata_mode=MetadataMode.LLM)}\n"
                "-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#\n"
            )
        return "\n\n".join(fmt_node_txts)


def create_index(docs_dir, storage_dir):
    IPYNBReader = download_loader("IPYNBReader")
    ipynb_loader = IPYNBReader(concatenate=True)

    # load the documents and create the index
    documents = SimpleDirectoryReader(
        docs_dir, recursive=True, file_extractor={".ipynb": ipynb_loader}
    ).load_data()

    parser = LangchainNodeParser(
        RecursiveCharacterTextSplitter.from_language(
            language=Language.PYTHON, chunk_size=8000, chunk_overlap=1000
        )
    )
    nodes = parser.get_nodes_from_documents(documents)
    index = VectorStoreIndex(nodes)
    print("CREATING INDEX AT: ", storage_dir)
    # store it for later
    index.storage_context.persist(persist_dir=storage_dir)

    return index


def get_informed_answer(
    question,
    domain,
    domain_description,
    docs_dir,
    storage_dir,
    vector_top_k=40,
    reranker_top_n=20,
    rerank=False,
):
    llm3_general = OpenAI(model="gpt-3.5-turbo-1106", temperature=0.1)
    llm3_synthesizer = OpenAI(
        model="gpt-3.5-turbo-1106", temperature=0.1, max_tokens=1000
    )
    llm4 = OpenAI(model="gpt-4-1106-preview", temperature=0.5)

    json_llm_predictor = JSONLLMPredictor(llm=llm3_general)

    service_context3_general = ServiceContext.from_defaults(
        llm_predictor=json_llm_predictor
    )
    service_context3_synthesizer = ServiceContext.from_defaults(llm=llm3_synthesizer)
    service_context4 = ServiceContext.from_defaults(llm=llm4)

    reranker_context = service_context3_general
    response_synthesizer_context = service_context4

    STORAGE_DIR = f"{storage_dir}/{domain}"
    DOCS_DIR = f"{docs_dir}/{domain}"

    # check if storage already exists
    if not os.path.exists(STORAGE_DIR):
        index = create_index(
            docs_dir=DOCS_DIR, storage_dir=STORAGE_DIR
        )
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

            # print(f"RERANKED_NODES: {len(retrieved_nodes)}\n\n")
            # for node in retrieved_nodes:
            #     file_info = node.metadata.get("file_name") or node.metadata.get(
            #         "file_path"
            #     )
            #     print(f"FILE INFO: {file_info}")
            #     print(node)
            #     print("\n\n")

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
