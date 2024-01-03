import logging
import os
from dotenv import load_dotenv


from llama_index.llms import OpenAI, Ollama

# Load environment variables
load_dotenv()

# Logger setup
logger = logging.getLogger(__name__)

# OpenAI API key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

USE_LOCAL_LLM = os.getenv("USE_LOCAL_LLM", "False").lower() == "true"
USE_LOCAL_EMBEDDINGS = os.getenv("USE_LOCAL_EMBEDDINGS", "False").lower() == "true"

DEFAULT_MODELS = {
    "GPT3": {
        "model": "gpt-3.5-turbo-1106",
        "context_length": 16384,
        "api_key": OPENAI_API_KEY,
    },
    "GPT4": {
        "model": "gpt-4-1106-preview",
        "context_length": 131072,
        "api_key": OPENAI_API_KEY,
    },
}

LOCAL_MODELS = {
    "mistral": {
        "base_url": "http://0.0.0.0:8000",
        "api_key": "sk-1234",
        "model": "ollama/mistral",
        "context_length": 4096,
    },
    "deepseek-coder": {
        "base_url": "http://0.0.0.0:8503",
        "api_key": "sk-1234",
        "model": "ollama/deepseek-coder:6.7b",
        "context_length": 4096,
    },
}

from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.llms import Ollama as OllamaLangchain

llm3_general = OpenAI(model=DEFAULT_MODELS["GPT3"]["model"], temperature=0.1)
llm4_general = OpenAI(model=DEFAULT_MODELS["GPT4"]["model"], temperature=0.1)

llm3_synthesizer = OpenAI(model=DEFAULT_MODELS["GPT3"]["model"], temperature=0.5)
llm4_synthesizer = OpenAI(model=DEFAULT_MODELS["GPT4"]["model"], temperature=0.5)

llm3_predictor = OpenAI(model=DEFAULT_MODELS["GPT3"]["model"], temperature=0.1)
llm4_predictor = OpenAI(model=DEFAULT_MODELS["GPT4"]["model"], temperature=0.1)

llm_local_general = Ollama(model=LOCAL_MODELS["mistral"]["model"], temperature=0.1, request_timeout=300)
llm_local_synthesizer = Ollama(model=LOCAL_MODELS["mistral"]["model"], temperature=0.5, request_timeout=300)
llm_local_predictor = OllamaLangchain(
    model="mistral", callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]), temperature=0.1, format="json"
)

llm_general = None
llm_synthesizer = None
llm_predictor = None
embed_model = None

if USE_LOCAL_LLM:
    llm_general = llm_local_general
    llm_synthesizer = llm_local_synthesizer
    llm_predictor = llm_local_predictor
else:
    llm_general = llm4_general
    llm_synthesizer = llm4_synthesizer
    llm_predictor = llm3_predictor
    
if USE_LOCAL_EMBEDDINGS:
    embed_model = "local"