import logging

from utils.search_tools import find_relevant_github_repo

logging.basicConfig(level=logging.INFO)

domain, domain_description = find_relevant_github_repo("hapi framework for generating rest endpoints. it includes middleware for customizing endpoints and adds relational functionality to a domument database.")