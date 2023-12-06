from typing import List
import autogen
import os
from dotenv import load_dotenv
from termcolor import colored
from autogen.agentchat.contrib.retrieve_user_proxy_agent import RetrieveUserProxyAgent
from autogen.agentchat.contrib.retrieve_assistant_agent import RetrieveAssistantAgent
from autogen import Agent

from llama_index import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    load_index_from_storage,
    ServiceContext,
    Document,
    llms,
    prompts,
    indices,
    retrievers,
    response_synthesizers,
)
from typing import Callable, Dict, Optional, Union, List, Tuple, Any

from chromadb.api.types import QueryResult

from llama_index.schema import NodeWithScore

from llama_index.llms import OpenAI

from llama_index.indices.postprocessor import LLMRerank
from llama_index.indices.query.schema import QueryBundle
from llama_index.retrievers import VectorIndexRetriever
from llama_index.response_synthesizers import (
    ResponseMode,
    get_response_synthesizer,
)
from llama_index.prompts import PromptTemplate

from typing import Dict, Union

# load_dotenv()


# config_list3 = [
#     {
#         'model': 'gpt-3.5-turbo-1106',
#         'api_key': os.environ["OPENAI_API_KEY"],
#     }
# ]

# llm_config3={
#     "request_timeout": 600,
#     "seed": 42,
#     "config_list": config_list3,
#     "temperature": 0,
# }

# config_list4 = [
#     {
#         'model': 'gpt-4-1106-preview',
#         'api_key': os.environ["OPENAI_API_KEY"],
#     }
# ]

# llm_config4={
#     "request_timeout": 600,
#     "seed": 42,
#     "config_list": config_list4,
#     "temperature": 0,
# }


# STORAGE_DIR = "./storage/autogen/"
# DOCS_DIR = "docs/autogen"

# logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
# logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))


# llm3_general = OpenAI(model="gpt-3.5-turbo-1106", temperature=0.1)
# llm3_synthesizer = OpenAI(model="gpt-3.5-turbo-1106", temperature=0.1, max_tokens=1000)
# llm4 = OpenAI(model="gpt-4-1106-preview", temperature=0.5)


# service_context3_general = ServiceContext.from_defaults(llm=llm3_general)
# service_context3_synthesizer = ServiceContext.from_defaults(llm=llm3_synthesizer)
# service_context4 = ServiceContext.from_defaults(llm=llm4)

# reranker_context = service_context3_general
# response_synthesizer_context = service_context4


# check if storage already exists
# if not os.path.exists(STORAGE_DIR):
#     # load the documents and create the index
#     documents = SimpleDirectoryReader(DOCS_DIR).load_data()
#     index = VectorStoreIndex.from_documents(documents)
#     # store it for later
#     index.storage_context.persist(persist_dir=STORAGE_DIR)
# else:
#     # load the existing index
#     storage_context = StorageContext.from_defaults(persist_dir=STORAGE_DIR)
#     index = load_index_from_storage(storage_context)


CUSTOM_PROMPT = """You're a retrieve augmented chatbot. You ANSWER the USER_QUESTION based on the
RELEVANT_CONTEXT provided by the user. You should follow the following steps to answer a question:
Step 1, you estimate the user's intent based on the question and context. The intent can be a code generation task or
a question answering task.
Step 2, you reply based on the intent.
If you can't answer the question with or without the current context, you should reply exactly `UPDATE CONTEXT`.
If user's intent is code generation, you must obey the following rules:
Rule 1. You MUST NOT install any packages because all the packages needed are already installed.
Rule 2. You must follow the formats below to write your code:
```language
# your code
```

If user's intent is question answering, you must give as short an answer as possible.

RELAVANT_CONTEXT:
###----------------------------------------###
{input_context}
###----------------------------------------###

USER_QUESTION: 
###----------------------------------------###
{input_question}
###----------------------------------------###
ANSWER:
"""


class LlamaRetrieveUserProxyAgent(RetrieveUserProxyAgent):
    def __init__(self, *args, **kwargs):
        if kwargs.get("retrieve_config", {}).get("customized_prompt", None) is None:
            kwargs["retrieve_config"]["customized_prompt"] = CUSTOM_PROMPT
        # Check retrieve_config for "docs_path"
        if kwargs.get("retrieve_config", {}).get("docs_path", None) is None:
            kwargs["retrieve_config"]["docs_path"] = "docs"
        # Check retrieve_config for "storage_path"
        if kwargs.get("retrieve_config", {}).get("storage_path", None) is None:
            kwargs["retrieve_config"]["storage_path"] = "storage"

        super().__init__(*args, **kwargs)

        # Check retrieve_config for "reranker_context"
        if kwargs.get("retrieve_config", {}).get("reranker_context", None) is None:
            llm = OpenAI(
                model=self.llm_config["config_list"][0]["model"], temperature=0.1
            )
            kwargs["retrieve_config"][
                "reranker_context"
            ] = ServiceContext.from_defaults(llm=llm)

        DOCS_DIR = kwargs["retrieve_config"]["docs_path"]
        STORAGE_DIR = kwargs["retrieve_config"]["storage_path"]
        # check if storage already exists
        if not os.path.exists(STORAGE_DIR):
            # load the documents and create the index
            documents = SimpleDirectoryReader(
                DOCS_DIR,
                recursive=True,
            ).load_data()
            self.index = VectorStoreIndex.from_documents(documents)
            # store it for later
            self.index.storage_context.persist(persist_dir=STORAGE_DIR)
        else:
            # load the existing index
            storage_context = StorageContext.from_defaults(persist_dir=STORAGE_DIR)
            self.index = load_index_from_storage(storage_context)

    def query_vector_db(self, query_texts, n_results=10, **kwargs) -> QueryResult:
        # Implement your custom retrieval logic here using your vector database client

        nodes = self.get_retrieved_nodes(
            str(query_texts), vector_top_k=n_results, reranker_top_n=20, rerank=False
        )
        retrieved_docs = [self.node_to_document(node) for node in nodes]
        return self.convert_documents_to_queryresult(retrieved_docs)

    def retrieve_docs(
        self, problem: str, n_results: int = 40, search_string: str = "", **kwargs
    ):
        results = self.query_vector_db(
            query_texts=problem,
            n_results=n_results,
            search_string=search_string,
            **kwargs,
        )

        self._results = results
        # print("doc_ids: ", results["ids"])

    def _get_context(self, results: Dict[str, Union[List[str], List[List[str]]]]):
        doc_contents = ""
        current_tokens = 0
        _doc_idx = self._doc_idx
        _tmp_retrieve_count = 0
        for idx, doc in enumerate(results["documents"][0]):
            if idx <= _doc_idx:
                continue
            if results["ids"][0][idx] in self._doc_ids:
                continue

            # Extract text from doc dictionary
            doc_text = doc["text"]

            _doc_tokens = self.custom_token_count_function(doc_text, self._model)
            if _doc_tokens > self._context_max_tokens:
                func_print = f"Skip doc_id {results['ids'][0][idx]} as it is too long to fit in the context."
                print(colored(func_print, "green"), flush=True)
                self._doc_idx = idx
                continue
            if current_tokens + _doc_tokens > self._context_max_tokens:
                break
            func_print = f"Adding doc_id {results['ids'][0][idx]} to context."
            print(colored(func_print, "green"), flush=True)
            current_tokens += _doc_tokens

            # Append extracted text to doc_contents
            doc_contents += doc_text + "\n"

            self._doc_idx = idx
            self._doc_ids.append(results["ids"][0][idx])
            self._doc_contents.append(doc_text)
            print("DOC_CONTENTS_LEN: ", len(self._doc_contents))
            _tmp_retrieve_count += 1
            if _tmp_retrieve_count >= self.n_results:
                break
        print("DOC_CONTENTS_LEN: ", len(self._doc_contents))
        return doc_contents

    @staticmethod
    def get_max_tokens(model="gpt-3.5-turbo"):
        print("MODEL_TURBO: ", model)
        if "preview" in model:
            print("TURBO_BABY!!")
            return 128000
        elif "32k" in model:
            return 32000
        elif "16k" in model:
            return 16000
        elif "gpt-4" in model:
            return 8000
        else:
            return 4000

    def get_retrieved_nodes(
        self, query_str, vector_top_k=40, reranker_top_n=20, rerank=True
    ):
        print(f"GETTING TOP NODES: {vector_top_k}")
        query_bundle = QueryBundle(query_str)
        # configure retriever
        retriever = VectorIndexRetriever(
            index=self.index,
            similarity_top_k=vector_top_k,
        )
        print("RETRIEVING_NODES_WITH_QUERY: ", query_str)
        retrieved_nodes = retriever.retrieve(query_bundle)

        # print("RETRIEVED NODES:\n\n")
        # for node in retrieved_nodes:
        #     print(node)
        #     print("\n\n")

        if rerank:
            print(f"GETTING RERANKED NODES: {reranker_top_n}")
            # configure reranker
            reranker = LLMRerank(
                choice_batch_size=5,
                top_n=reranker_top_n,
                service_context=self._retrieve_config["reranker_context"],
            )
            retrieved_nodes = reranker.postprocess_nodes(retrieved_nodes, query_bundle)

        # print("RERANKED NODES:\n\n")
        # for node in retrieved_nodes:
        #     print(node)
        #     print("\n\n")

        return retrieved_nodes

    def convert_documents_to_queryresult(
        self, documents: List[Document]
    ) -> QueryResult:
        # Extracting data from each document
        ids = [doc.id_ for doc in documents]
        embeddings = [[doc.embedding] if doc.embedding else [] for doc in documents]

        # Modify this line to create a list of dictionaries
        doc_texts = [{"text": doc.text} for doc in documents]

        metadatas = [[doc.metadata] if doc.metadata else [] for doc in documents]

        # Assembling the QueryResult
        query_result = QueryResult(
            ids=[ids],  # Wrap in another list as QueryResult expects lists of lists
            embeddings=[embeddings],
            documents=[doc_texts],  # Now a list of dictionaries
            uris=[[]],  # Assuming URIs are not available in Document
            data=[[]],  # Assuming data is not available in Document
            metadatas=[metadatas],
            distances=[[]],  # Assuming distances are not available
        )

        return query_result

    def node_to_document(self, node: NodeWithScore):
        # Extract relevant data from the Node
        # For example, let's assume your Node has 'id', 'content', and 'metadata' attributes
        doc_id = node.node_id
        doc_text = node.text
        doc_metadata = node.metadata

        # Create and return a Document object
        return Document(id=doc_id, text=doc_text, metadata=doc_metadata)

    def _generate_retrieve_user_reply(
        self,
        messages: Optional[List[Dict]] = None,
        sender: Optional[Agent] = None,
        config: Optional[Any] = None,
    ) -> Tuple[bool, Union[str, Dict, None]]:
        pass
