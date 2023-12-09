from dotenv import load_dotenv
import logging

from utils.rag_tools import get_informed_answer

# Set to DEBUG for more verbose logging
logging.basicConfig(level=logging.INFO)

load_dotenv()

STORAGE_DIR = "./storage"
DOCS_DIR = "./docs"

# NOTE: This assumes you have an "autogen" directory under the "docs" directory with autogen content (ex: the autogen cloned repo)
domain = "autogen"
domain_description = "autonomous agent frameworks"


def main():
    question = "What is autogen?"
    
    answer = get_informed_answer(
        question,
        docs_dir=DOCS_DIR,
        storage_dir=STORAGE_DIR,
        domain=domain,
        domain_description=domain_description,
        vector_top_k=10,
        reranker_top_n=2,
        rerank=True,
        fusion=True,
    )

    print("GOT ANSWER: ", answer.response)


if __name__ == "__main__":
    main()
