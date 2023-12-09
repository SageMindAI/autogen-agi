from dotenv import load_dotenv
import logging

from utils.rag_tools import get_informed_answer

# Set to DEBUG for more verbose logging
logging.basicConfig(level=logging.INFO)

load_dotenv()

STORAGE_DIR = "./storage"
DOCS_DIR = "./docs"

domain = "llama_index"
domain_description = "indexing and retrieval of documents for llms"


def main():
    question = "How can I index various types of documents?"
    
    answer = get_informed_answer(
        question,
        docs_dir=DOCS_DIR,
        storage_dir=STORAGE_DIR,
        domain=domain,
        domain_description=domain_description,
        vector_top_k=25,
        reranker_top_n=5,
        rerank=True,
        fusion=True,
    )

    print("GOT ANSWER: ", answer.response)


if __name__ == "__main__":
    main()
