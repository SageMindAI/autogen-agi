"""
This file contains functions for Retriever-Augmented Generation (RAG).
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
    download_loader,
)
from llama_index.bridge.pydantic import BaseModel
from llama_index.indices.postprocessor import LLMRerank
from llama_index.indices.query.schema import QueryBundle
from llama_index.llm_predictor import LLMPredictor
from llama_index.llms import OpenAI
from llama_index.node_parser import LangchainNodeParser
from llama_index.prompts.base import BasePromptTemplate
from llama_index.prompts import PromptTemplate
from llama_index.prompts.prompt_type import PromptType
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
    CHOICE_SELECT_PROMPT_TMPL,
    RAG_FUSION_PROMPT,
    DOMAIN_QA_PROMPT_TMPL_STR,
    GENERAL_QA_PROMPT_TMPL_STR,
)

# Configuration and Constants

# Load environment variables
load_dotenv()

# Logger setup
logger = logging.getLogger(__name__)

# OpenAI API key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# LLM Models Configuration
LLM_CONFIGS = {
    "gpt-3.5-turbo": {
        "model": "gpt-3.5-turbo-1106",
        "api_key": OPENAI_API_KEY,
    },
    "gpt-4": {
        "model": "gpt-4-1106-preview",
        "api_key": OPENAI_API_KEY,
    },
}

# Instantiate LLM Models
llm3_general = OpenAI(model=LLM_CONFIGS["gpt-3.5-turbo"]["model"], temperature=0.1)
llm3_synthesizer = OpenAI(
    model=LLM_CONFIGS["gpt-3.5-turbo"]["model"], temperature=0.1, max_tokens=1000
)
llm4 = OpenAI(model=LLM_CONFIGS["gpt-4"]["model"], temperature=0.5)


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
            # return_json=True is the only difference from the original function and currently only applies to the latest OpenAI GPT 3.5 and 4 models.
            response = self._llm.complete(formatted_prompt, return_json=True)
            output = response.text

        logger.debug(output)
        return output


class ModifiedLLMRerank(LLMRerank):
    """
    A class extending LLMRerank to provide customized reranking functionality.
    """

    def __init__(self, **kwargs):
        """
        Initialize the ModifiedLLMRerank.
        """
        super().__init__(**kwargs)
        CHOICE_SELECT_PROMPT = PromptTemplate(
            CHOICE_SELECT_PROMPT_TMPL, prompt_type=PromptType.CHOICE_SELECT
        )
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
            logger.info(f"Reranking batch of {len(nodes_batch)} nodes...")
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
    """
    Creates an index from documents located in the specified directory.

    NOTE: This function will continue to be customized to support more file/data types.

    TODO: Take advantage of the "HierarchicalNodeParser" for generic text.

    Args:
        docs_dir (str): The directory containing the documents.
        storage_dir (str): The directory to store the created index.

    Returns:
        VectorStoreIndex: The created index.
    """
    IPYNBReader = download_loader("IPYNBReader")
    ipynb_loader = IPYNBReader(concatenate=True)

    try:
        documents = SimpleDirectoryReader(
            docs_dir, recursive=True, file_extractor={".ipynb": ipynb_loader}
        ).load_data()
    except Exception as e:
        logger.error(f"Error reading documents: {e}")
        raise

    parser = LangchainNodeParser(
        RecursiveCharacterTextSplitter.from_language(
            language=Language.PYTHON, chunk_size=8000, chunk_overlap=1000
        )
    )
    nodes = parser.get_nodes_from_documents(documents)

    print("Creating index at:", storage_dir)

    index = VectorStoreIndex(nodes)

    try:
        index.storage_context.persist(persist_dir=storage_dir)
    except Exception as e:
        logger.error(f"Error saving index: {e}")
        raise

    return index


def rag_fusion(query, query_context=None, number_of_variations=4):
    """
    Generates query variations for Retriever-Augmented Generation (RAG) fusion.

    Args:
        query (str): The original query.
        query_context (str): Context to enrich the query variations.
        number_of_variations (int): The number of query variations to generate.

    Returns:
        List[str]: A list of query variations.
    """
    logger.info("Getting query variations for RAG fusion...")
    rag_fusion_prompt = RAG_FUSION_PROMPT.format(
        query=query,
        query_context=query_context,
        number_of_variations=number_of_variations,
    )

    try:
        rag_fusion_response = light_gpt4_wrapper_autogen(
            query=rag_fusion_prompt, return_json=True
        )
    except Exception as e:
        logger.error(f"Error in RAG fusion: {e}")
        raise

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
    query_context=None,
):
    """
    Retrieves nodes based on the provided query string and other parameters.

    Args:
        query_str (str): The query string.
        index: The index to search in.
        vector_top_k (int): The number of top vectors to retrieve.
        reranker_top_n (int): The number of top nodes to keep after reranking.
        rerank (bool): Flag to perform reranking.
        score_threshold (int): The threshold for score filtering.
        fusion (bool): Flag to perform fusion.
        query_context: Additional context for the query.

    Returns:
        List: A list of retrieved nodes.
    """
    json_llm_predictor = JSONLLMPredictor(llm=llm3_general)

    service_context3_general = ServiceContext.from_defaults(
        llm_predictor=json_llm_predictor
    )

    reranker_context = service_context3_general

    logger.info(f"Getting top {vector_top_k} nodes")
    # configure retriever

    if fusion:
        query_variations = rag_fusion(query_str, query_context)
        query_variations.append(query_str)
        logger.info(f"Query variations for RAG fusion: {query_variations}")
    else:
        query_variations = [query_str]

    retrieved_nodes = []
    num_of_variations = len(query_variations)
    logger.info(f"Number of variations: {num_of_variations}")
    results_per_variation = int(vector_top_k / num_of_variations)
    logger.info(f"Results per variation: {results_per_variation}")
    for variation in query_variations:
        base_retriever = VectorIndexRetriever(
            index=index,
            similarity_top_k=results_per_variation,
        )

        retriever = AutoMergingRetriever(
            base_retriever, index.storage_context, verbose=False
        )
        query_bundle = QueryBundle(variation)

        variation_nodes = retrieve_nodes_with_retry(retriever, query_bundle)

        logger.debug(f"ORIGINAL NODES for query: {variation}\n\n")
        for node in variation_nodes:
            file_info = node.metadata.get("file_name") or node.metadata.get("file_path")
            node_info = (
                f"FILE INFO: {file_info}\n"
                f"NODE ID: {node.id_}\n"
                f"NODE Score: {node.score}\n"
                f"NODE Length: {len(node.text)}\n"
                f"NODE Text: {node.text}\n-----------\n"
            )
            logger.debug(node_info)

        # add variation nodes to retrieved nodes
        retrieved_nodes.extend(variation_nodes)

    retrieved_nodes = remove_duplicate_nodes(retrieved_nodes)

    if rerank:
        retrieved_nodes = rerank_nodes(
            nodes=retrieved_nodes,
            query_str=query_str,
            query_context=query_context,
            context=reranker_context,
            top_n=reranker_top_n,
            score_threshold=score_threshold,
        )

    total_tokens = sum(count_token(node.text) for node in retrieved_nodes)
    logger.info(f"Total node tokens: {total_tokens}")

    return retrieved_nodes


def retrieve_nodes_with_retry(retriever, query_bundle, max_retries=3):
    """
    Attempts to retrieve nodes with a retry mechanism.

    Args:
        retriever: The retriever to use for node retrieval.
        query_bundle: The query bundle for the retriever.
        max_retries (int): Maximum number of retries.

    Returns:
        List: A list of retrieved nodes.
    """
    attempt = 0
    while attempt < max_retries:
        try:
            return retriever.retrieve(query_bundle)
        except openai.APITimeoutError as e:
            logger.warning(f"Timeout error on attempt {attempt + 1}: {e}")
            attempt += 1
    raise TimeoutError(f"Failed to retrieve nodes after {max_retries} attempts")


def remove_duplicate_nodes(nodes):
    """
    Removes duplicate nodes based on their id.

    Args:
        nodes (List): A list of nodes.

    Returns:
        List: A list of unique nodes.
    """
    logger.info("Removing duplicate nodes")
    logger.info(f"Node count before deduplication: {len(nodes)}")
    nodes = list({node.id_: node for node in nodes}.values())
    logger.info(f"Node count after deduplication: {len(nodes)}")

    return nodes


def rerank_nodes(nodes, query_str, query_context, context, top_n, score_threshold=5):
    """
    Reranks the nodes based on the provided context and thresholds.

    Args:
        nodes (List): A list of nodes to rerank.
        query_bundle: The query bundle for the reranker.
        context: The service context for reranking.
        top_n (int): The number of top nodes to keep after reranking.
        score_threshold (int): The threshold for score filtering.

    Returns:
        List: A list of reranked nodes.
    """
    logger.info(
        f"Reranking top {top_n} nodes with a score threshold of {score_threshold}"
    )
    reranker = ModifiedLLMRerank(
        choice_batch_size=5, top_n=top_n, service_context=context
    )
    if query_context:
        query_str = (
            "\nQUESTION_CONTEXT:\n---------------------\n"
            f"{query_context}\n"
            "---------------------\n"
            f"QUESTION:\n---------------------\n"
            f"{query_str}\n"
            "---------------------\n"
        )

    query_bundle = QueryBundle(query_str)
    reranked_nodes = reranker.postprocess_nodes(nodes, query_bundle)

    logger.debug(f"RERANKED NODES:\n\n")
    for node in reranked_nodes:
        file_info = node.metadata.get("file_name") or node.metadata.get("file_path")
        node_info = (
            f"FILE INFO: {file_info}\n"
            f"NODE ID: {node.id_}\n"
            f"NODE Score: {node.score}\n"
            f"NODE Length: {len(node.text)}\n"
            f"NODE Text: {node.text}\n-----------\n"
        )
        logger.debug(node_info)

    filtered_nodes = [node for node in reranked_nodes if node.score > score_threshold]
    logger.info(
        f"Number of nodes after re-ranking and filtering: {len(filtered_nodes)}"
    )

    return filtered_nodes


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
    """
    Retrieves an informed answer to the given question using the specified document directories and settings.

    Args:
        question (str): The question to retrieve an answer for.
        docs_dir (str): The directory containing the documents to query.
        storage_dir (str): The directory for storing the index.
        domain (str, optional): The specific domain of the question.
        domain_description (str, optional): The description of the domain.
        vector_top_k (int): The number of top vectors to retrieve.
        reranker_top_n (int): The number of top nodes to keep after reranking.
        rerank (bool): Flag to perform reranking.
        fusion (bool): Flag to perform fusion.

    Returns:
        str: The synthesized response to the question.
    """
    service_context3_synthesizer = ServiceContext.from_defaults(llm=llm3_synthesizer)
    service_context4 = ServiceContext.from_defaults(llm=llm4)
    response_synthesizer_context = service_context4

    storage_dir = f"{storage_dir}/{domain}" if domain else storage_dir
    docs_dir = f"{docs_dir}/{domain}" if domain else docs_dir

    if not os.path.exists(storage_dir):
        index = create_index(docs_dir=docs_dir, storage_dir=storage_dir)
    else:
        logger.info(f"Loading index at: {storage_dir}")
        storage_context = StorageContext.from_defaults(persist_dir=storage_dir)
        index = load_index_from_storage(storage_context)

    nodes = None
    max_retries = 3
    attempt = 0
    while nodes is None and attempt < max_retries:
        try:
            nodes = get_retrieved_nodes(
                question,
                index,
                vector_top_k,
                reranker_top_n,
                rerank,
                fusion,
                domain_description,
            )
        except (
            IndexError
        ) as e:  # This happens with the default re-ranker, but not the modified one due to the JSON response
            logger.error(f"Index error: {e}")
            attempt += 1

    if nodes is None:
        raise RuntimeError("Failed to retrieve nodes after multiple attempts.")

    logger.info(f"\nRAG Question:\n{question}")

    text_qa_template_str = (
        GENERAL_QA_PROMPT_TMPL_STR
        if domain is None or domain_description is None
        else format_incrementally(
            DOMAIN_QA_PROMPT_TMPL_STR,
            {"domain": domain, "domain_description": domain_description},
        )
    )
    text_qa_template = PromptTemplate(text_qa_template_str)

    response_synthesizer = get_response_synthesizer(
        response_mode=ResponseMode.COMPACT,
        text_qa_template=text_qa_template,
        service_context=response_synthesizer_context,
    )

    response = response_synthesizer.synthesize(question, nodes=nodes)
    return response
