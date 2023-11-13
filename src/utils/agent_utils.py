import autogen
from autogen import OpenAIWrapper

from dotenv import load_dotenv

load_dotenv()

import os
import json

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


def get_end_intent(message):

    TERMINATE_SYSTEM_MESSAGE = """You are an expert in text and sentiment analysis. Based on the provided text, please respond with whether the intent is to end the conversation or contintue the conversation. If the text provides all-caps statements such as "TERMINATE" or "CONTINUE", prioritize these when assesing intent. Your response MUST be in JSON format, with the following format:
    {{
        "analysis": <your analysis of the text>,
        "intent": "end" or "continue"
    }}

    IMPORTANT: ONLY respond with the JSON object, and nothing else. If you respond with anything else, the system will not be able to understand your response.

    """

    client = OpenAIWrapper(config_list=config_list4)
    response = client.create(
        messages=[
            {"role": "system", "content": TERMINATE_SYSTEM_MESSAGE},
            {"role": "user", "content": message["content"]},
        ]
    )
    response = client.extract_text_or_function_call(response)
    json_response = autogen.ConversableAgent._format_json_str(response)
    try:
        json_response = json.loads(json_response)
    except Exception as error:
        json_response = fix_broken_json(json_response)
    print("JSON: ", json_response)
    print("INTENT:", json_response["intent"])
    return json_response["intent"]

