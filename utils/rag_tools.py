"""
DESCRIPTION: This file contains functions for Retriever-Augmented Generation (RAG).
"""
# Standard Library Imports
import os
import logging
from typing import List, Optional, Tuple, Any
from time import sleep

# Third-Party Imports
import openai
from dotenv import load_dotenv

# Local Imports
from autogen.token_count_utils import count_token
from llama_index import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    load_index_from_storage,
    ServiceContext,
)
from llama_index.bridge.pydantic import BaseModel
from llama_index.indices.postprocessor import LLMRerank
from llama_index.indices.query.schema import QueryBundle
from llama_index.llm_predictor import LLMPredictor
from llama_index.llms import OpenAI
from llama_index.node_parser import LangchainNodeParser
from llama_index.prompts.base import BasePromptTemplate, PromptTemplate
from llama_index.retrievers import VectorIndexRetriever, AutoMergingRetriever
from llama_index.retrievers.auto_merging_retriever import AutoMergingRetriever
from llama_index.response_synthesizers import ResponseMode, get_response_synthesizer
from llama_index.schema import BaseNode, NodeWithScore, MetadataMode
from llama_index.storage import StorageContext
from langchain.text_splitter import RecursiveCharacterTextSplitter, Language


# Relative Imports
from .misc import (
    extract_json_response,
    light_gpt4_wrapper_autogen,
    format_incrementally,
)
from prompts.misc_prompts import (
    CHOICE_SELECT_PROMPT,
    RAG_FUSION_PROMPT,
    DOMAIN_QA_PROMPT_TMPL_STR,
    GENERAL_QA_PROMPT_TMPL_STR,
)

# Load environment variables
load_dotenv()

# Logger setup
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


llm3_general = OpenAI(model="gpt-3.5-turbo-1106", temperature=0.1)
llm3_synthesizer = OpenAI(model="gpt-3.5-turbo-1106", temperature=0.1, max_tokens=1000)
llm4 = OpenAI(model="gpt-4-1106-preview", temperature=0.5)


class JSONLLMPredictor(LLMPredictor):
    """
    A class extending LLMPredictor to handle predictions with JSON-specific functionalities.
    """

    def __init__(self, **kwargs):
        """
        Initialize the JSONLLMPredictor.
        """
        super().__init__(**kwargs)

    def predict(
        self,
        prompt: BasePromptTemplate,
        output_cls: Optional[BaseModel] = None,
        **prompt_args: Any,
    ) -> str:
        """
        Make a prediction based on the given prompt and arguments.

        Args:
            prompt (BasePromptTemplate): The prompt template to use.
            output_cls (Optional[BaseModel]): The output class for structured responses.
            **prompt_args (Any): Additional arguments for prompt formatting.

        Returns:
            str: The prediction result.
        """
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


class ModifiedLLMRerank(LLMRerank):
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
        """Assign each summary node a number, and format the batch of nodes."""
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


def rag_fusion(query, number_of_variations=4):
    print("GETTING QUERY VARIATIONS FOR RAG FUSION...")

    rag_fusion_prompt = RAG_FUSION_PROMPT.format(
        query=query, number_of_variations=number_of_variations
    )
    rag_fusion_response = light_gpt4_wrapper_autogen(
        query=rag_fusion_prompt, return_json=True
    )

    print("RAG FUSION RESPONSE: ", rag_fusion_response)

    query_variations = [
        variation["query"] for variation in rag_fusion_response["query_variations"]
    ]

    return query_variations


def get_retrieved_nodes(
    query_str,
    index,
    vector_top_k=40,
    reranker_top_n=20,
    rerank=True,
    score_threshold=5,
    fusion=True,
):
    json_llm_predictor = JSONLLMPredictor(llm=llm3_general)

    service_context3_general = ServiceContext.from_defaults(
        llm_predictor=json_llm_predictor
    )

    reranker_context = service_context3_general
    print(f"GETTING TOP NODES: {vector_top_k}")
    # configure retriever

    if fusion:
        query_variations = rag_fusion(query_str)
        query_variations.append(query_str)
        print("QUERY VARIATIONS: ", query_variations)
    else:
        query_variations = [query_str]

    retrieved_nodes = []
    num_of_variations = len(query_variations)
    print("NUM OF VARIATIONS: ", num_of_variations)
    results_per_variation = int(vector_top_k / num_of_variations)
    print("RESULTS PER VARIATION: ", results_per_variation)
    for variation in query_variations:
        base_retriever = VectorIndexRetriever(
            index=index,
            similarity_top_k=results_per_variation,
        )

        retriever = AutoMergingRetriever(
            base_retriever, index.storage_context, verbose=False
        )
        query_bundle = QueryBundle(variation)
        variation_nodes = None
        while variation_nodes is None:
            try:
                variation_nodes = retriever.retrieve(query_bundle)
            except openai.APITimeoutError as e:
                print("TIMEOUT_ERROR: ", e)
                continue
        variation_nodes = retriever.retrieve(query_bundle)

        print(f"ORIGINAL NODES for query: {variation}\n\n")
        for node in variation_nodes:
            file_info = node.metadata.get("file_name") or node.metadata.get("file_path")
            print(f"FILE INFO: {file_info}")
            print("NODE ID: ", node.id_)
            print("NODE Score: ", node.score)
            print("NODE Length: ", len(node.text))
            print("NODE Text: ", node.text, "\n-----------\n")

        # add variation nodes to retrieved nodes
        retrieved_nodes.extend(variation_nodes)

    # remove duplicate nodes by id_
    print("NODE COUNT BEFORE REMOVING DUPLICATES: ", len(retrieved_nodes))
    retrieved_nodes = list({node.id_: node for node in retrieved_nodes}.values())
    print("NODE COUNT AFTER REMOVING DUPLICATES: ", len(retrieved_nodes))

    if rerank:
        print(f"GETTING RERANKED NODES: {reranker_top_n}")
        # configure reranker
        reranker = ModifiedLLMRerank(
            choice_batch_size=5,
            top_n=reranker_top_n,
            service_context=reranker_context,
        )
        print("RERANKING NODES...")
        retrieved_nodes = reranker.postprocess_nodes(retrieved_nodes, query_bundle)

        # filter out nodes with low scores
        retrieved_nodes = [
            node for node in retrieved_nodes if node.score > score_threshold
        ]

        print(f"RERANKED_NODES: {len(retrieved_nodes)}\n\n")
        for node in retrieved_nodes:
            file_info = node.metadata.get("file_name") or node.metadata.get("file_path")
            print(f"FILE INFO: {file_info}")
            print("NODE ID: ", node.id_)
            print("NODE Score: ", node.score)
            print("NODE Length: ", len(node.text))
            print("NODE Text: ", node.text, "\n-----------\n")

    # count tokens in nodes
    total_tokens = 0
    for node in retrieved_nodes:
        total_tokens += count_token(node.text)

    print("TOTAL NODE TOKENS: ", total_tokens)

    return retrieved_nodes


def get_informed_answer(
    question,
    docs_dir,
    storage_dir,
    domain=None,
    domain_description=None,
    vector_top_k=40,
    reranker_top_n=20,
    rerank=False,
    fusion=False,
):
    service_context3_synthesizer = ServiceContext.from_defaults(llm=llm3_synthesizer)
    service_context4 = ServiceContext.from_defaults(llm=llm4)

    response_synthesizer_context = service_context4

    if domain is None:
        STORAGE_DIR = storage_dir
        DOCS_DIR = docs_dir
    else:
        STORAGE_DIR = f"{storage_dir}/{domain}"
        DOCS_DIR = f"{docs_dir}/{domain}"

    # check if storage already exists
    if not os.path.exists(STORAGE_DIR):
        index = create_index(docs_dir=DOCS_DIR, storage_dir=STORAGE_DIR)
    else:
        # load the existing index
        print("LOADING INDEX AT: ", STORAGE_DIR)
        storage_context = StorageContext.from_defaults(persist_dir=STORAGE_DIR)
        index = load_index_from_storage(storage_context)

    nodes = None

    while nodes is None:
        try:
            nodes = get_retrieved_nodes(
                query_str=question,
                index=index,
                vector_top_k=vector_top_k,
                reranker_top_n=reranker_top_n,
                rerank=rerank,
                fusion=fusion,
            )
        except (
            IndexError
        ) as e:  # This happens with the default re-ranker, but not the modified one due to the JSON response
            print("INDEX ERROR: ", e)
            continue

    # print("NODES:\n\n")
    # for node in nodes:
    #     print(node)
    #     print("\n\n")

    print("\nRAG QUESTION:\n\n")
    print(question)

    if domain is None or domain_description is None:
        text_qa_template_str = GENERAL_QA_PROMPT_TMPL_STR
    else:
        data = {
            "domain": domain,
            "domain_description": domain_description,
        }
        text_qa_template_str = format_incrementally(DOMAIN_QA_PROMPT_TMPL_STR, data)
    text_qa_template = PromptTemplate(text_qa_template_str)

    response_synthesizer = get_response_synthesizer(
        response_mode=ResponseMode.COMPACT,
        text_qa_template=text_qa_template,
        service_context=response_synthesizer_context,
    )

    response = response_synthesizer.synthesize(question, nodes=nodes)
    return response
