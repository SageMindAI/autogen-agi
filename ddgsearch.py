import argparse
import bs4
from duckduckgo_search import ddg
import openai
import logging
import threading
import queue
from readability import Document

from src.utils.misc import light_gpt3_wrapper_autogen


from dotenv import load_dotenv
load_dotenv()

# ******
# this is a hack to stop scrapy from logging its version info to stdout
# there should be a better way to do this, but I don't know what it is
import scrapy.utils.log

def null_log_scrapy_info(settings):
    pass

# replace the log_scrapy_info function with a null function
# get the module dictionary that contains the log_scrapy_info function
log_scrapy_info_module_dict = scrapy.utils.log.__dict__

# set the log_scrapy_info function to null
log_scrapy_info_module_dict['log_scrapy_info'] = null_log_scrapy_info
# ******

import scrapy
from scrapy.crawler import CrawlerProcess

def extract_useful_information_from_single_chunk(url, title, text, ix, q=None):
    '''
    This function takes the url, title, and a chunk of text of a webpage, and it asks
    openai to extract only the useful information from the text. It returns the result,
    which is a string of text, and it also puts the result in a queue if a queue is passed in.
    '''
    # in this function, we will take the url, title, and some text extracted from the webpage
    # by bs4, and we will ask openai to extract only the useful information from the text

    logger = logging.getLogger("ddgsearch")
    logger.info(f"extracting useful information from chunk {ix}, title: {title}")

    prompt = f"""
Here is a url: {url}
Here is its title: {title}
Here is some text extracted from the webpage by bs4:
---------
{text}
---------

Web pages can have a lot of useless junk in them. For example, there might be a lot of ads, or a lot of navigation links, 
or a lot of text that is not relevant to the topic of the page. We want to extract only the useful information from the text.

You can use the url and title to help you understand the context of the text.
Please extract only the useful information from the text. Try not to rewrite the text, but instead extract only the useful information from the text.
"""

    response = light_gpt3_wrapper_autogen(prompt)
    # response = openai.completions.create(
    #     model="gpt3.5-turbo-1106",
    #     prompt=prompt,
    #     max_tokens=1000,
    #     temperature=0.2,
    #     top_p=1,
    #     frequency_penalty=0,
    #     presence_penalty=0
    # )


    content = response.choices[0].message.content

    print("OPENAI RESPONSE:", content)

    if q:
        q.put((ix, content))
    logger.info(f"DONE extracting useful information from chunk {ix}, title: {title}")

    text = response.choices[0].message.content

    # sometimes the first line is something like "Useful information extracted from the text:", so we remove that
    lines = text.splitlines()
    if "useful information" in lines[0].lower():
        text = '\n'.join(lines[1:])

    return (ix, text)

def extract_useful_information(url, title, text, max_chunks):
    '''
    This function takes the url, title, and text of a webpage.
    It returns the most useful information from the text.

    , and it calls
    extract_useful_information_from_single_chunk to extract the useful information.

    It does this by breaking the text into chunks, and then calling 
    extract_useful_information_from_single_chunk on each chunk (which is turn calls openai).
    It then concatenates the results from all the chunks.

    It uses threading to do this in parallel, because openai is slow.
    '''

    chunks = [text[i*1000: i*1000+1100] for i in range(len(text)//1000)]
    chunks = chunks[:max_chunks]

    threads = []
    
    q = queue.Queue()

    for ix, chunk in enumerate(chunks):
        t = threading.Thread(target=extract_useful_information_from_single_chunk, args=(url, title, chunk, ix, q))
        threads.append(t)
        t.start()

    # Wait for all threads to complete
    for t in threads:
        t.join()

    # Get all the results from the queue
    results = []
    while not q.empty():
        results.append(q.get())

    logger = logging.getLogger("ddgsearch")
    logger.info (f"Got {len(results)} results from the queue")

    # Sort the results by the index
    results.sort(key=lambda x: x[0])

    # concatenate the text from the results
    text = ''.join([x[1] for x in results])

    return text

def readability(input_text):
    '''
    This function will use the readability library to extract the useful information from the text.
    Document is a class in the readability library. That library is (roughly) a python
    port of readability.js, which is a javascript library that is used by firefox to
    extract the useful information from a webpage. We will use the Document class to
    extract the useful information from the text.
    '''

    doc = Document(input_text)

    summary = doc.summary()

    # the summary is html, so we will use bs4 to extract the text
    soup = bs4.BeautifulSoup(summary, 'html.parser')
    summary_text = soup.get_text()

    return summary_text

def remove_duplicate_empty_lines(input_text):
    lines = input_text.splitlines()

    # this function removes all duplicate empty lines from the lines
    fixed_lines = []
    for index, line in enumerate(lines):
        if line.strip() == '':
            if index != 0 and lines[index-1].strip() != '':
                fixed_lines.append(line)
        else:
            fixed_lines.append(line)

    return '\n'.join(fixed_lines)


class MySpider(scrapy.Spider):
    '''
    This is the spider that will be used to crawl the webpages. We give this to the scrapy crawler.
    '''
    name = 'myspider'
    start_urls = None
    clean_with_llm = False 
    results = []

    def __init__(self, start_urls, clean_with_llm, *args, **kwargs):
        super(MySpider, self).__init__(*args, **kwargs)
        self.start_urls = start_urls
        self.clean_with_llm = clean_with_llm

    def start_requests(self):
        for url in self.start_urls:
            yield scrapy.Request(url, callback=self.parse)

    def parse(self, response):
        logger = logging.getLogger('ddgsearch')
        logger.info(f"***Parsing {response.url}...")

        body_html = response.body.decode('utf-8')

        url = response.url

        soup = bs4.BeautifulSoup(body_html, 'html.parser')
        title = soup.title.string
        text = soup.get_text()
        text = remove_duplicate_empty_lines(text) 

        if self.clean_with_llm:
            useful_text = extract_useful_information(url, title, text, 50)
        else:
            useful_text = readability(body_html)
        useful_text = remove_duplicate_empty_lines(useful_text)

        self.results.append({
            'url': url,
            'title': title,
            'text': text,
            'useful_text': useful_text
        })

def setloglevel(loglevel):
    # this function sets the log level for the script
    if loglevel == 'DEBUG':
        logging_level = logging.DEBUG
    elif loglevel == 'INFO':
        logging_level = logging.INFO
    elif loglevel == 'WARNING':
        logging_level = logging.WARNING
    elif loglevel == 'ERROR':
        logging_level = logging.ERROR
    elif loglevel == 'CRITICAL':
        logging_level = logging.CRITICAL
    else:
        logging_level = logging.INFO 

    # surely there is a better way to do this?
    logger = logging.getLogger('scrapy')
    logger.setLevel(logging_level)
    logger = logging.getLogger('filelock')
    logger.setLevel(logging_level)
    logger = logging.getLogger('py.warnings')
    logger.setLevel(logging_level)
    logger = logging.getLogger('readability')
    logger.setLevel(logging_level)
    logger = logging.getLogger('ddgsearch')
    logger.setLevel(logging_level)
    logger = logging.getLogger('urllib3')
    logger.setLevel(logging_level)
    logger = logging.getLogger('openai')
    logger.setLevel(logging_level)

def ddgsearch(query, numresults=10, clean_with_llm=False, loglevel='ERROR'):
    '''
    This function performs a search on duckduckgo and returns the results.
    It uses the scrapy library to download the pages and extract the useful information.
    It extracts useful information from the pages using either the readability library 
    or openai, depending on the value of clean_with_llm.
    
    query: the query to search for
    numresults: the number of results to return
    clean_with_llm: if True, use openai to clean the text. If False, use readability.
    loglevel: the log level to use, a string. Can be DEBUG, INFO, WARNING, ERROR, or CRITICAL.
    '''
    # set the log level
    setloglevel(loglevel)
      
    # perform the search
    results = ddg(query, max_results=numresults)

    logger = logging.getLogger('ddgsearch')
    logger.info(f"Got {len(results)} results from the search.")
    logger.debug(f"Results: {results}")

    # get the urls
    urls = [result['href'] for result in results]
    urls = urls[:numresults]

    process = CrawlerProcess()
    setloglevel(loglevel) # this is necessary because the crawler process modifies the log level
    process.crawl(MySpider, urls, clean_with_llm)
    process.start()

    # here the spider has finished downloading the pages and cleaning them up
    return MySpider.results

def main():
    # usage: python ddgsearch.py query [--numresults <numresults=10>] [--clean_with_llm] [--outfile <outfile name>] [--loglevel <loglevel=ERROR>] [--noprint]
    # ddgsearch performs the search, gets the results, and downloads the pages and prints the text.
    # parse command line arguments
    parser = argparse.ArgumentParser()

    import os
    import re

    parser.add_argument('query', help='the query to search for')
    parser.add_argument('--numresults', help='the number of results to return', default=10)
    parser.add_argument('--clean_with_llm', help='clean the text with the llm', default=True)
    parser.add_argument('--outfile', help='the name of the file to write the results to', default=None)
    parser.add_argument('--loglevel', help='the log level', default='ERROR')
    parser.add_argument('--noprint', help='do not print the results to the screen', action='store_true')

    args = parser.parse_args()

    query = args.query
    numresults = int(args.numresults)
    clean_with_llm = args.clean_with_llm

    def make_filename_safe(input_string):
        # replace all non-alphanumeric characters with underscores
        return re.sub(r'\W+', '_', input_string)

    default_outfile = os.path.join('working', f'{make_filename_safe(query)}.txt')

    outfile = args.outfile or default_outfile
    loglevel = args.loglevel
    noprint = args.noprint

    results = ddgsearch(query, numresults, clean_with_llm, loglevel)

    def get_result_lines(results, shorten):
        result_lines = []
        for index, results in enumerate(results):
            result_lines.append("***************************************")
            result_lines.append(f"Result {index+1}")
            result_lines.append(f"Url: {results['url']}")
            result_lines.append(f"Title: {results['title']}")
            if shorten:
                result_lines.append("Cleaned Text (shortened):")
                useful_lines = results['useful_text'].splitlines()[:20]
                short_useful_text = '\n'.join(useful_lines)
                result_lines.append(short_useful_text)
            else:
                result_lines.append("Cleaned Text:")
                result_lines.append(results['useful_text'])
                result_lines.append("Full Text:")
                result_lines.append(results['text'])
            result_lines.append("***************************************")
            result_lines.append('')
        return result_lines

    if outfile:
        # make sure this is unicode safe
        with open(outfile, 'w', encoding='utf-8') as f:
            result_lines = get_result_lines(results, shorten=False)
            f.writelines([f"{result}\n" for result in result_lines])

    if not noprint:
        shortened_result_lines = get_result_lines(results, shorten=True)
        for line in shortened_result_lines:
            print(line)
    
if __name__ == '__main__':
    main()