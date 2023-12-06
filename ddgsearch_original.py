import argparse
import bs4
from duckduckgo_search import ddg
import logging
from readability import Document
import json

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

def ddgsearch(query, results_file, numresults=10, clean_with_llm=False, loglevel='ERROR'):
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

    results = MySpider.results

    # save the results to a file
    with open(results_file, 'w') as f:
        json.dump(results, f)

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
    parser.add_argument('--clean_with_llm', help='clean the text with the llm', action='store_true')
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