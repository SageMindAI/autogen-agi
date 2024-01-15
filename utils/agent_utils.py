"""
This file contains utility functions specific to autogen agents.
"""

import autogen
from autogen import OpenAIWrapper
from .misc import fix_broken_json

from dotenv import load_dotenv

load_dotenv()

import os
import json
import logging

logger = logging.getLogger(__name__)

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

def get_end_intent(message):

    IS_TERMINATE_SYSTEM_PROMPT = """You are an expert in text and sentiment analysis. Based on the provided text, please respond with whether the intent is to end/pause the conversation or contintue the conversation. If the text provides all-caps statements such as "TERMINATE" or "CONTINUE", prioritize these when assesing intent. Your response MUST be in JSON format, with the following format:
    {{
        "analysis": <your analysis of the text>,
        "intent": "end" or "continue"
    }}

    NOTE: If the intent is to get feedback from the User or UserProxy, the intent should be "end".

    IMPORTANT: ONLY respond with the JSON object, and nothing else. If you respond with anything else, the system will not be able to understand your response.

    """

    # TODO: Ensure JSON response with return_json param
    client = OpenAIWrapper(config_list=config_list4)
    response = client.create(
        messages=[
            {"role": "system", "content": IS_TERMINATE_SYSTEM_PROMPT},
            {"role": "user", "content": message["content"]},
        ]
    )
    json_response = autogen.ConversableAgent._format_json_str(response.choices[0].message.content)
    try:
        json_response = json.loads(json_response)
    except Exception as error:
        json_response = fix_broken_json(json_response)
    logger.info("Termination analysis: %s", json_response["analysis"])
    logger.info("Termination intent: %s", json_response["intent"])
    return json_response["intent"]

