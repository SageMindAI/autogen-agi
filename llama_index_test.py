"""
This script demonstrates the rag tools available in this project. The rag tool searches the docs directory for a sub directory that matches the domain name and uses the available documents to performa a RAG (Retrieval Augmented Generation) query. By default the search domain of "llama_index" is available.
"""

from dotenv import load_dotenv
import logging
import os
 
from utils.rag_tools import create_index


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
from llama_index.llms import Ollama
# from llama_index.embeddings import OllamaEmbedding
from llama_index import set_global_service_context
from langchain.text_splitter import RecursiveCharacterTextSplitter, Language
from langchain.embeddings import HuggingFaceEmbeddings



# Set to DEBUG for more verbose logging
logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)

load_dotenv()

STORAGE_DIR = "./storage"
DOCS_DIR = "./docs"

# Set to TRUE to use local llms
USE_LOCAL_LLM = True

if USE_LOCAL_LLM:
    llm = Ollama(model="mistral", request_timeout=1000)
    
    embed_model = "local"
    
    # Uncomment to use an OllamaEmbedding model
    # embed_model= OllamaEmbedding(model_name="mistral")
    
    # Uncomment to specify a HuggingFaceEmbeddings model
    # embed_model = HuggingFaceEmbeddings(
    #     model_name="sentence-transformers/all-mpnet-base-v2"
    # )

    service_context = ServiceContext.from_defaults(
        llm=llm,
        embed_model=embed_model
    )
else:
    service_context = None

if not os.path.exists(STORAGE_DIR):
    index = create_index(docs_dir=DOCS_DIR, storage_dir=STORAGE_DIR, service_context=service_context)
else:
    logger.info(f"Loading index at: {STORAGE_DIR}")
    storage_context = StorageContext.from_defaults(persist_dir=STORAGE_DIR)
    index = load_index_from_storage(storage_context, service_context=service_context)


domain = "llama_index"
domain_description = "indexing and retrieval of documents for llms"

def main():
    question = "How can I index various types of documents?"
    print("QUESTION: ", question)
    query_engine = index.as_query_engine(
        similarity_top_k=2,
    )

    response = query_engine.query(question)

    print("GOT ANSWER: ", response)


if __name__ == "__main__":
    main()
