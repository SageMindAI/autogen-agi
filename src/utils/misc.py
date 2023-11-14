
"""
DESCRIPTION: This file contains miscellaneous functions that are used in multiple scripts.
"""

import os
import json
import openai
import autogen
from autogen import OpenAIWrapper
from time import sleep

from llama_index.llms import OpenAI

from dotenv import load_dotenv

load_dotenv()


config_list3 = [
    {
        "model": "gpt-3.5-turbo",
        "api_key": os.environ["OPENAI_API_KEY"],
    }
]

config_list4 = [
    {
        "model": "gpt-4-1106-preview",
        "api_key": os.environ["OPENAI_API_KEY"],
    }
]


def load_json(filename):
    if os.path.isfile(filename) and os.path.getsize(filename) > 0:
        with open(filename, "r") as file:
            return json.load(file)
    else:
        return []


def save_json(data, filename):
    with open(filename, "w") as file:
        json.dump(data, file, indent=4, ensure_ascii=False)



def load_file(filename):
    with open(filename) as f:
        file = f.read()

        if len(file) == 0:
            raise ValueError("filename cannot be empty.")

        return file
    
def extract_base_path(full_path, target_directory):
    """
    Extracts the base path up to and including the target directory from the given full path.

    :param full_path: The complete file path.
    :param target_directory: The target directory to which the path should be truncated.
    :return: The base path up to and including the target directory, or None if the target directory is not in the path.
    """
    path_parts = full_path.split(os.sep)
    if target_directory in path_parts:
        target_index = path_parts.index(target_directory)
        base_path = os.sep.join(path_parts[:target_index + 1])
        return base_path
    else:
        return None
    
def light_llm_wrapper(llm, query):
    response = None
    while response is None or response.text == "":
        try:
            response = llm.complete(query)
        except openai.RateLimitError as e:
            print("RATE LIMIT ERROR: ", e)
            sleep(5)
            continue
        except IndexError as e:
            print("INDEX ERROR: ", e)
            continue

    return response

def light_llm4_wrapper(query):
    llm4 = OpenAI(model="gpt-4-1106-preview", temperature=0.1)
    return light_llm_wrapper(llm4, query)
    

def map_directory_to_json(dir_path):
    def dir_to_dict(path):
        dir_dict = {'name': os.path.basename(path)}
        if os.path.isdir(path):
            dir_dict['type'] = 'directory'
            dir_dict['children'] = [dir_to_dict(os.path.join(path, x)) for x in os.listdir(path)]
        else:
            dir_dict['type'] = 'file'
        return dir_dict

    root_structure = dir_to_dict(dir_path)
    return json.dumps(root_structure, indent=4)


def remove_substring(string, substring):
    return string.replace(substring, "")


def fix_broken_json(potential_json, max_attempts=5):
    FIX_JSON_PROMPT = f"""You are a helpful assistant. A user will ask a question, and you should provide an answer.
    ONLY return the answer, and nothing more.
    
    Given the following potential JSON text, please fix any broken JSON syntax. Do NOT change the text itself. ONLY respond with the fixed JSON.

    Potential JSON:
    ---
    {potential_json}
    ---

    Response:
    """

    attempts = 0
    error = None
    while attempts < max_attempts:
        try:
            attempts += 1
            client = OpenAIWrapper(config_list=config_list3)
            response = client.create(
                messages=[{
                    "role": "user",
                    "content": FIX_JSON_PROMPT
                }]
            )
            response = client.extract_text_or_function_call(response)
            response = autogen.ConversableAgent._format_json_str(response)
            response = json.loads(response)
            return response
        except Exception as error:
            print("FIX ATTEMPT FAILED, TRYING AGAIN...", attempts)
            error = error

    raise error

def extract_json_response_oai_wrapper(message):
    client = OpenAIWrapper(config_list=config_list3)
    response = client.extract_text_or_function_call(message)
    response = autogen.ConversableAgent._format_json_str(response)
    try:
        response = json.loads(response)
    except Exception as error:
        response = fix_broken_json(response)

    return response


def extract_json_response(message):
    try:
        response = json.loads(message)
    except Exception as error:
        response = fix_broken_json(message)

    return response
