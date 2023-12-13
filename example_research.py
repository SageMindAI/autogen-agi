"""
This script demonstrates the current research tools for the project. The research tools are used to find relevant GitHub repositories for a given query. The search engine options are: "google", "ddg", and "serpapi". The default search engine is "ddg". The "find_relevant_github_repo" function will search for the most relevant github repo and clone it to the "docs" directory. This can then be used to perform a RAG (Retrieval Augmented Generation) query via the "get_informed_answer" function in utils/rag_tools.py (see example_rag.py for an example of how to use this function).
"""

import logging

from utils.search_tools import find_relevant_github_repo

logging.basicConfig(level=logging.INFO)

# NOTE: visit https://serpapi.com/ to get your own API key
# NOTE: visit https://programmablesearchengine.google.com/controlpanel/create to get your own API key

domain, domain_description = find_relevant_github_repo("autogen python framework for autonomous AI agents", search_engine="ddg")
