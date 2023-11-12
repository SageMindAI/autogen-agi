
"""
DESCRIPTION: This file contains miscellaneous functions that are used in multiple scripts.
"""

import os
import json
import openai
from time import sleep


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
