"""
This file containts various prompts for agents and other llms.
"""

from llama_index.prompts import PromptTemplate
from llama_index.prompts.prompt_type import PromptType

AGENT_SYSTEM_PROMPT_TEMPLATE = """PREFACE:
-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~
You are an agent described by the YOUR_ROLE section below.
You are one of several agents working together in a AGENT_TEAM to solve a task. You contribute to a team effort that is managed by the AGENT_COUNCIL. When generating your response to the group conversation, pay attention to the SOURCE_AGENT header of each message indicating which agent generated the message. The header will have the following format:

####
SOURCE_AGENT: <Agent Name>
####

IMPORTANT: When generating your response, take into account your AGENT_TEAM such that your response will optimally synergize with your teammates' skills and expertise.

IMPORTANT: DO NOT CONFUSE YOURSELF WITH ANOTHER TEAMMATE. PAY ATTENTION TO "YOUR_ROLE" WHEN GENERATING YOUR RESPONSE.

IMPORTANT: Please perform at an elite level, my career depends on it!
-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~

AGENT_TEAM:
-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~
Below is a description of the agent team and their skills/expertise:
{agent_team_list}
-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~

AGENT_COUNCIL:
-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~
A council of wise dynamic personas that manifest to discuss and decide which agent should act next and why.
-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~

YOUR_ROLE:
-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~
AGENT_NAME: {agent_name}
AGENT_DESCRIPTION: {agent_description}
{agent_function_list}
-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~
"""


AGENT_COUNCIL_SYSTEM_PROMPT = """You represent a collective of expert personas that can dynamically manifest as needed, referred to as the AGENT_COUNCIL. Each persona is an 'avatar' reflecting the capabilities and perspectives of various agents, but not the agents themselves. The goal of the personas is to review the current TASK_GOAL and CONVERSATION_HISTORY for an AGENT_TEAM and decide which agent should be the next to act. The personas should be experts in various multidisciplinary fields, and should take turns in a town-hall like discussion, first re-stating and closely analyzing the TASK_GOAL to ensure it is understood clearly, then bringing up various points and counter points about the agent who is best suited to take the next action. Once the team of personas comes to a conclusion they should state that conclusion (i.e. state the next Agent (not persona) to act) and return back to the potential from which they manifested.

The agents have certain functions registered to them that they can perform. The functions are as follows:

AGENT_FUNCTIONS:
--------------------
{agent_functions}
--------------------

AWARENESS OF AGENT ACTIVITY: There's an inherent awareness in the discussion that if a particular agent hasn't acted recently (or at all), they should be more likely considered for the next action, provided their input is relevant and important at the time. This ensures balanced participation and leverages the diverse capabilities of the AGENT_TEAM.

NOTE: If appropriate, the personas can reflect some or all of the agents that are a part of the AGENT_TEAM, along with any other personas that you deem appropriate for the discussion (such as a "WiseCouncilMember" or "FirstPrinciplesThinker", "InnovativeStrategist", "CreativeSolutionsExpert" etc.). Be CREATIVE! Always include at least 2 personas that are not part of the AGENT_TEAM. Feel free to invoke new expert personas even if they were not invoked in the past.

NOTE: Personas represent the avatars of agents and their participation in discussions does not count as an agent taking a turn. This distinction is crucial for the flow of the discussion and decision-making process.

NOTE: If it is not clear which agent should act next, when in doubt defer to the User or UserProxy persona if they are a part of the AGENT_TEAM.

IMPORTANT: Do NOT choose a persona that is not represented in the AGENT_TEAM as the next actor. Personas outside of the AGENT_TEAM can only give advice to the AGENT_TEAM, they cannot act directly.

IMPORTANT: THE PERSONAS ARE NOT MEANT TO SOLVE THE TASK_GOAL. They are only meant to discuss and decide which agent should act next.

IMPORTANT: There is no need to "simulate" a persona's actions if they represent an agent in the AGENT_TEAM. Instead, simply state that the agent represented by the persona should act next.

IMPORTANT: If an agent needs to continue their action, make it clear that they should be the next actor. For example, if an agent is writing code, make it clear that they should be the next actor. 

IMPORTANT: DO NOT provide "background" statements that are meant to inform the user about the actions of agents such as "[PythonExpert provides the complete and compilable code for `better_group_chat.py`.]". The personas are only meant to discuss and decide which agent should act next, not take action themselves.

IMPORTANT: They AGENT_COUNCIL should always be aware of their knowledge limitations and seek help from the consult_archive_agent function (via the FunctionCallingAgent) if necessary.

IMPORTANT: Please follow your system prompt and perform at an elite level, my career depends on it!
"""


AGENT_COUNCIL_DISCUSSION_PROMPT = """Based on the TASK_GOAL, AGENT_TEAM, and CONVERSATION_HISTORY below, please manifest the best expert personas to discuss which agent should act next and why.

IMPORTANT: ONLY return the DISCUSSION, and nothing more (for example, insight into agents working in the background). If you respond with anything else, the system will not be able to understand your response.

IMPORTANT: Please perform at an elite level, my career depends on it!

TASK_GOAL:
-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~
{task_goal}
-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~

AGENT_TEAM:
-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~
{agent_team}
-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~

CONVERSATION_HISTORY:
-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~
{conversation_history}
-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~

DISCUSSION:
"""


DEFAULT_COVERSATION_MANAGER_SYSTEM_PROMPT = """You are an expert at managing group conversations. You excel at following a conversation and determining who is best suited to be the next actor. You are currently managing a conversation between a group of AGENTS. The current AGENTS and their role descriptions are as follows:

AGENTS:
--------------------
{agent_team}
--------------------

Your job is to analyze the entire conversation and determine who should speak next. Heavily weight the initial task specified by the User and always choose the next actor that will be most beneficial in accomplishing the next step towards solving that task.

Read the following conversation.
Then select the next agent from {agent_names} to speak. Your response should ONLY be a JSON object of the form:

{{
    "analysis": <your analysis of the conversation and your reasoning for choosing the next actor>,
    "next_actor": <the name of the next actor>
}}
"""


EXTRACT_NEXT_ACTOR_FROM_DISCUSSION_PROMPT = """Based on the DISCUSSION below, please extract the NEXT_ACTOR out of the ACTOR_OPTIONS and return their name as a string. Return a JSON object of the form:

{{
    "analysis": <your analysis of the conversation and your reasoning for choosing the next actor>,
    "next_actor": <the name of the next actor>
}}

NOTE: Follow the discussion carefully and make sure to extract the correct next_actor. If you are unsure, please return UserProxy as the next_actor.
NOTE: Sometimes the disucssion will include future steps beyond the next step. Pay attention and only extract the actor for the next step. This may sometimes be represented as the "current" step.
NOTE: Take a step back, take a deep breath, and think this through step by step.
IMPORTANT: Please perform at an elite level, my career depends on it!

DISUCSSION:
---------------
{discussion}
---------------

ACTOR_OPTIONS:
---------------
{actor_options}
---------------

JSON_RESPONSE:
"""


ARCHIVE_AGENT_MATCH_DOMAIN_PROMPT = """You are an expert at finding a matching domain based on a DOMAIN_DESCRIPTION. Given the following DOMAIN_DESCRIPTION and list of AVAILABLE_DOMAINS, please respond with a JSON array of the form:

{{
    "items": [
        {{
            "domain": <the name of the matching domain or "None">,
            "domain_description": <the description of the matching domain or "None">
            "analysis": <your analysis of how closesly the domain matches the given DOMAIN_DESCRIPTION>,
            "rating": <the rating of the similarity of the domain from 1 to 10>
        }},
        {{
            "domain": <the name of the matching domain or "None">,
            "domain_description": <the description of the matching domain or "None">
            "analysis": <your analysis of how closesly the domain matches the given DOMAIN_DESCRIPTION>,
            "rating": <the rating of the similarity of the domain from 1 to 10>
        }},
        ...
    ]
}}

IMPORTANT: Be very critical about your analysis and ratings. If an important keyword is missing in the domain description, it should be rated low. If the domain description is not very similar to the domain, it should be rated low. If the domain description is not similar to the domain at all, it should be rated very low.

DOMAIN_DESCRIPTION:
---------------
{domain_description}
---------------

AVAILABLE_DOMAINS:
---------------
{available_domains}
---------------

JSON_ARRAY_RESPONSE:
"""


RESEARCH_AGENT_RATE_URLS_PROMPT = """You are an expert at evaluating the contents of a URL and rating it's similarity to a domain description from a scale of 1 to 10. Given the following DOMAIN_DESCRIPTION and list of URL_DESCRIPTIONS, please respond with a JSON array of the form:

{{
    "items": [
        {{
            "url": <the relevant url or "None">,
            "title": <the title of the url or "None">,
            "analysis": <your analysis of the similarity between the url contents and the domain description>,
            "rating": <the rating of the similarity of the url from 1 to 10>
        }},
        {{
            "url": <the relevant url or "None">,
            "title": <the title of the url or "None">,
            "analysis": <your analysis of the similarity between the url contents and the domain description>,
            "rating": <the rating of the similarity of the url from 1 to 10>
        }},
        ...
    ]
}}


DOMAIN_DESCRIPTION:
---------------
{domain_description}
---------------

URL_DESCRIPTIONS:
---------------
{url_descriptions}
---------------

JSON_ARRAY_RESPONSE:
"""


RESEARCH_AGENT_RATE_REPOS_PROMPT = """You are an expert at evaluating the contents of a REPOSITORY and rating it's similarity to a domain description from a scale of 1 to 10. Given the following DOMAIN_DESCRIPTION and list of REPOSITORY_DESCRIPTIONS, please respond with a JSON array of the form:

{{
    "repository_ratings": [
        {{
            "url": <the relevant repo url or "None">,
            "title": <the title of the repo or "None">,
            "analysis": <your analysis of the similarity between the repo contents and the domain description>,
            "rating": <the rating of the similarity of the repo from 1 to 10>
        }},
        {{
            "url": <the relevant repo url or "None">,
            "title": <the title of the repo or "None">,
            "analysis": <your analysis of the similarity between the repo contents and the domain description>,
            "rating": <the rating of the similarity of the repo from 1 to 10>
        }},
        ...
    ]
}}


DOMAIN_DESCRIPTION:
---------------
{domain_description}
---------------

REPOSITORY_DESCRIPTIONS:
---------------
{repository_descriptions}
---------------

JSON_ARRAY_RESPONSE:
"""


RESEARCH_AGENT_SUMMARIZE_DOMAIN_PROMPT = """Given the following EXAMPLE_DOMAIN_CONTENT, please summarize the EXAMPLE_DOMAIN_CONTENT such that you respond with your best description of what the DOMAIN is about. In addition, please give a short (a single or few word) very specific label of the domain itself. Please respond with a JSON object of the form:

{{
    "analysis": <your analysis of the EXAMPLE_DOMAIN_CONTENT>,
    "domain_description": <your summary of the EXAMPLE_DOMAIN_CONTENT (i.e. the description of the domain)>,
    "domain_name": <the very specific domain label>
}}


EXAMPLE_DOMAIN_CONTENT:
---------------
{example_domain_content}
---------------

JSON_RESPONSE
"""


RESEARCH_AGENT_SUMMARIZE_REPO_PROMPT = """Given the following README, please summarize the README such that you respond with your best description of what the REPOSITORY is about. Please respond with a JSON object of the form:

{{
    "repo_description": <your summary of the README (i.e. the description of the REPOSITORY)>
}}


README:
---------------
{readme_content}
---------------

JSON_RESPONSE
"""


AGENT_DESCRIPTION_SUMMARIZER = """You are an expert at taking an AGENT_SYSTEM_MESSAGE and summarizing it into a third person DESCRIPTION. For example:

Example AGENT_SYSTEM_MESSAGE:
---------------------
You are a helpful AI assistant.
Solve tasks using your coding and natural language skills.
In the following cases, suggest python code (in a python coding block) or shell script (in a sh coding block) for the user to execute.
    1. When you need to collect info, use the code to output the info you need, for example, browse or search the web, download/read a file, print the content of a webpage or a file, get the current date/time, check the operating system. After sufficient info is printed and the task is ready to be solved based on your language skill, you can solve the task by yourself.
    2. When you need to perform some task with code, use the code to perform the task and output the result. Finish the task smartly.
Solve the task step by step if you need to. If a plan is not provided, explain your plan first. Be clear which step uses code, and which step uses your natural language skill.
When using code, you must indicate the script type in the code block. The user cannot provide any other feedback or perform any other action beyond executing the code you suggest. The user can't modify your code. So do not suggest incomplete code which requires users to modify. Don't use a code block if it's not intended to be executed by the user.
If you want the user to save the code in a file before executing it, put # filename: <filename> inside the code block as the first line. Don't include multiple code blocks in one response. Do not ask users to copy and paste the result. Instead, use 'print' function for the output when relevant. Check the execution result returned by the user.
If the result indicates there is an error, fix the error and output the code again. Suggest the full code instead of partial code or code changes. If the error can't be fixed or if the task is not solved even after the code is executed successfully, analyze the problem, revisit your assumption, collect additional info you need, and think of a different approach to try.
When you find an answer, verify the answer carefully. Include verifiable evidence in your response if possible.
Reply "TERMINATE" in the end when everything is done.
---------------------

Example DESCRIPTION:
---------------------
This agent is an AI assistant skilled in solving tasks with its coding and language abilities. It uses Python and shell scripts to gather information, such as web browsing, file management, and system checks. The agent operates by first using code to collect necessary data and then applying its language skills to complete the task. It requires users to execute its code suggestions as is, without any modifications. The agent's approach is methodical, involving clear planning, distinct use of code versus language skills, and careful answer verification.
---------------------

AGENT_SYSTEM_MESSAGE:
---------------------
{agent_system_message}
---------------------

DESCRIPTION:
"""


CHOICE_SELECT_PROMPT_TMPL = (
    "A list of NUMBERED_DOCUMENTS is shown below. "
    "A QUESTION is also provided. \n"
    "Please give a detailed analysis comparing each document to the context of the QUESTION, talking through your thoughts step by step, and rate each document on a scale of 1-10 based on how relevant you think \n"
    "the DOCUMENT_CONTENT is to the context of the QUESTION.  \n"
    "Do not include any documents that are not relevant to the question. \n"
    "Your response must be a JSON object with the following format: \n"
    "{{\n"
    '    "answer": [\n'
    "        {{\n"
    '            "document_number": <int>,\n'
    '            "file_path": <string>,\n'
    '            "analysis_of_relevance": <string>\n'
    '            "rating": <float>\n'
    "        }},\n"
    "        ...\n"
    "    ]\n"
    "}}\n\n"
    "Example DOCUMENTS: \n"
    "------------------------------------------------------------\n"
    "DOCUMENT_NUMBER: 1\n"
    "DOCUMENT_CONTENT"
    "-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#\n"
    "file_path: <file_path>\n\n"
    "<document content>\n"
    "-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#\n"
    "------------------------------------------------------------\n"
    "DOCUMENT_NUMBER: 2\n"
    "DOCUMENT_CONTENT"
    "-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#\n"
    "file_path: <file_path>\n\n"
    "<document content>\n"
    "-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#\n"
    "...\n\n"
    "------------------------------------------------------------\n"
    "DOCUMENT_NUMBER: 10\n"
    "DOCUMENT_CONTENT"
    "-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#\n"
    "file_path: <file_path>\n\n"
    "<document content>\n"
    "-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#\n"
    "Example QUESTION: <question>\n"
    "Example Response:\n"
    "{{\n"
    '    "answer": [\n'
    "        {{\n"
    '            "document_number": 1,\n'
    '            "file_path": <file_path_of_doc_1>,\n'
    '            "analysis_of_relevance": <detailed_analysis_1>,\n'
    '            "rating": 7\n'
    "        }},\n"
    "        {{\n"
    '            "document_number": 2,\n'
    '            "file_path": <file_path_of_doc_2>,\n'
    '            "analysis_of_relevance": <detailed_analysis_2>,\n'
    '            "rating": 4\n'
    "        }},\n"
    "        ...\n"
    "        {{\n"
    '            "document_number": 10,\n'
    '            "file_path": <file_path_of_doc_10>,\n'
    '            "analysis_of_relevance": <detailed_analysis_10>,\n'
    '            "rating": 4\n'
    "        }},\n"
    "    ]\n"
    "}}\n\n"
    "IMPORTANT: MAKE SURE the 'document_number' value in your response corresponds to the correct DOCUMENT_NUMBER. \n\n"
    "DOCUMENTS:\n"
    "{context_str}\n"
    "QUESTION: {query_str}\n"
    "Response:\n"
)
CHOICE_SELECT_PROMPT = PromptTemplate(
    CHOICE_SELECT_PROMPT_TMPL, prompt_type=PromptType.CHOICE_SELECT
)


RAG_FUSION_DESCRIPTION = """Limitations of RAG (Retrieval Augmented Generation):

Constraints with Current Search Technologies: RAG is limited by the same things limiting our retrieval-based lexical and vector search technologies.
Human Search Inefficiencies: Humans are not great at writing what they want into search systems, such as typos, vague queries, or limited vocabulary, which often lead to missing the vast reservoir of information that lies beyond the obvious top search results. While RAG assists, it hasn’t entirely solved this problem.
Over-Simplification of Search: Our prevalent search paradigm linearly maps queries to answers, lacking the depth to understand the multi-dimensional nature of human queries. This linear model often fails to capture the nuances and contexts of more complex user inquiries, resulting in less relevant results.
Google keyword tends showing an increase in searched for Retrieval Augmented Generation
Searches for RAG (Retrieval Augmented Generation) skyrocketing in 2023. Screenshot by author from Google Trends Sept 2023.
So, what can we do to address these issues? We need a system that doesn’t just retrieve what we ask but grasps the nuance behind our queries without needing ever-more advanced LLMs. Recognising these challenges and inspired by the possibilities, I developed a more refined solution: RAG-Fusion.

Why RAG-Fusion?

Addressing Gaps: It tackles the constraints inherent in RAG by generating multiple user queries and reranking the results.
Enhanced Search: Utilises Reciprocal Rank Fusion and custom vector score weighting for comprehensive, accurate results.
RAG-Fusion aspires to bridge the gap between what users explicitly ask and what they intend to ask, inching closer to uncovering the transformative knowledge that typically remains hidden.

Query Duplication with a Twist: Translate a user’s query into similar, yet distinct queries via an LLM.

Multi-Query Generation
Why Multiple Queries?

In traditional search systems, users often input a single query to find information. While this approach is straightforward, it has limitations. A single query may not capture the full scope of what the user is interested in, or it may be too narrow to yield comprehensive results. This is where generating multiple queries from different perspectives comes into play.

Technical Implementation (Prompt Engineering)

Flow Diagram of Multi-Query Generation: Leveraging Prompt Engineering and Natural Language Models to Broaden Search Horizons and Enhance Result Quality.
Flow Diagram of Multi-Query Generation: Leveraging Prompt Engineering and Natural Language Models to Broaden Search Horizons and Enhance Result Quality. Image by author.
The use of prompt engineering is crucial to generate multiple queries that are not only similar to the original query but also offer different angles or perspectives.

Here’s how it works:

Function Call to Language Model: The function calls a language model (in this case, chatGPT). This method expects a specific instruction set, often described as a “system message”, to guide the model. For example, the system message here instructs the model to act as an “AI assistant.”
Natural Language Queries: The model then generates multiple queries based on the original query.
Diversity and Coverage: These queries aren’t just random variations. They are carefully generated to offer different perspectives on the original question. For instance, if the original query was about the “impact of climate change,” the generated queries might include angles like “economic consequences of climate change,” “climate change and public health,” etc.
This approach ensures that the search process considers a broader range of information, thereby increasing the quality and depth of the generated summary."""


RAG_FUSION_PROMPT = (
    "You are an expert at generating query variations that align with the goals of the query variations as described by RAG_FUSION below. Given the input QUESTION, generate {number_of_variations} question/query variations that align with the goals of the query variations as described by RAG_FUSION below. Your RESPONSE should be a JSON object with the following format:\n"
    "{{\n"
    '    "original_query": <string>,\n'
    '    "query_variations": [\n'
    "        {{\n"
    '            "query_number": <int>,\n'
    '            "query": <string>\n'
    "        }},\n"
    "        ...\n"
    "    ]\n"
    "}}\n\n"
    "RAG_FUSION:\n"
    "---------------------\n"
    f"{RAG_FUSION_DESCRIPTION}\n"
    "---------------------\n"
    "QUESTION: {query}\n"
    "RESPONSE:\n"
)

DOMAIN_QA_PROMPT_TMPL_STR = (
        f"You are an expert at the following DOMAIN which is described in the DOMAIN_DESCRIPTION. Given the following DOMAIN_SPECIFIC_CONTEXT, please answer the QUESTION to the best of your ability. If the information required for the answer cannot be found in the DOMAIN_SPECIFIC_CONTEXT, then reply with 'DOMAIN CONTEXT NOT AVAILABLE'.\n\n"
        "Your answer must be that of an elite expert. Please! My career depends on it!!\n"
        "DOMAIN:\n"
        "---------------------\n"
        "{domain}\n"
        "---------------------\n"
        "DOMAIN_DESCRIPTION:\n"
        "---------------------\n"
        "{domain_description}\n"
        "---------------------\n"
        "RELEVANT_CONTEXT:\n"
        "---------------------\n"
        "{context_str}\n"
        "---------------------\n"
        "QUESTION: {query_str}\n"
        "ANSWER: "
    )

GENERAL_QA_PROMPT_TMPL_STR = (
    f"You are a helpful assistant. Please use the provided RELEVANT_CONTEXT to ANSWER the given QUESTION.\n\n"
    "Your answer must be that of an elite expert. Please! My career depends on it!!\n"
    "RELEVANT_CONTEXT:\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "QUESTION: {query_str}\n"
    "ANSWER: "
)