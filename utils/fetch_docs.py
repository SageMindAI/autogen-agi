import requests
from bs4 import BeautifulSoup
import os
import argparse
import urllib.parse
from langchain.document_loaders import AsyncChromiumLoader
from langchain.document_transformers import BeautifulSoupTransformer

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



def fetch_and_save(url, base_url, folder, downloaded):
    try:
        # Check if the URL is a subset of the base URL
        if not url.startswith(base_url):
            print(f"URL not a subset of base URL, skipping: {url}")
            return

        response = requests.get(url)
        response.raise_for_status()

        # Create a path that mirrors the URL structure
        parsed_url = urllib.parse.urlparse(url)
        path = parsed_url.path.lstrip('/').removesuffix('.html')
        local_path = os.path.join(folder, path) + '.html'

        print("LOCAL PATH: ", local_path)

        # Check if file already exists
        if os.path.exists(local_path):
            print(f"File already exists, skipping: {local_path}")
            return

        os.makedirs(os.path.dirname(local_path), exist_ok=True)

        # Use the scrape_documentation_page function to process the page
        page_content = scrape_documentation_page(url)
        with open(local_path, 'w', encoding='utf-8') as f:
            f.write(page_content)
        print(f"Saved HTML: {local_path}")
        downloaded.add(url)

        # Process links within the page
        soup = BeautifulSoup(response.text, 'html.parser')
        for link in soup.find_all('a', href=True):
            href = link['href']
            if not href.startswith('http'):
                href = urllib.parse.urljoin(url, href)
            # Ensuring the link is not going outside the base path
            if href.startswith(base_url) and href not in downloaded:
                print(f"Found link: {href}")
                fetch_and_save(href, base_url, folder, downloaded)

    except requests.HTTPError as e:
        print(f"HTTP Error: {e} for URL: {url}")
    except requests.RequestException as e:
        print(f"Error downloading {url}: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")


def main():
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Fetch and save documentation pages.")
    parser.add_argument("start_url", help="The starting URL of the documentation")
    parser.add_argument("save_folder", help="The directory to save files")

    # Parse arguments
    args = parser.parse_args()

    # Use arguments
    start_url = args.start_url
    save_folder = args.save_folder

    os.makedirs(save_folder, exist_ok=True)

    base_url = start_url
    # base_url = urllib.parse.urljoin(start_url, '/')  # Base URL for subset comparison
    print("BASE URL: ", base_url)
    downloaded = set()  # To keep track of what has been downloaded
    fetch_and_save(start_url, base_url, save_folder, downloaded)
    print(f"Total files downloaded: {len(downloaded)}")

if __name__ == '__main__':
    main()