from langchain.document_loaders import AsyncChromiumLoader
from langchain.document_transformers import BeautifulSoupTransformer
import asyncio

def scrape_documentation_page(url):
    # Load HTML with AsyncChromiumLoader for dynamic content handling
    loader = AsyncChromiumLoader([url])
    html = loader.load()

    # Define tags to extract for comprehensive coverage
    tags_to_extract = ["span", "code", "p", "pre", "h1", "h2", "h3", "h4", "h5", "h6", "li", "div", "a", "img", "table", "ul", "ol"]

    # Transform with BeautifulSoupTransformer for efficient parsing
    bs_transformer = BeautifulSoupTransformer()
    docs_transformed = bs_transformer.transform_documents(html, tags_to_extract=tags_to_extract)

    return docs_transformed[0].page_content
