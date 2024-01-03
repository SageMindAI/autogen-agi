# from guidance import models, user, assistant, system, gen
# import guidance
# gpt = models.OpenAI("gpt-3.5-turbo")


# @guidance
# def add(lm, input1, input2):
#     lm += f' = {int(input1) + int(input2)}'
#     return lm
# @guidance
# def subtract(lm, input1, input2):
#     lm += f' = {int(input1) - int(input2)}'
#     return lm
# @guidance
# def multiply(lm, input1, input2):
#     lm += f' = {float(input1) * float(input2)}'
#     return lm
# @guidance
# def divide(lm, input1, input2):
#     lm += f' = {float(input1) / float(input2)}'
#     return lm


# lm = None

# with system():
#     lm = gpt + "You are an expert at creating character profiles for an RPG."

# # with user():
# #     character_description = "A beserker who is a master of the sword."
# #     lm += f"""\
# #     The following is a character profile for an RPG game in JSON format.
# #     ```json
# #     {{
# #         "id": "{id}",
# #         "description": "{character_description}",
# #         "name": "{gen('name', stop='"')}",
# #         "age": {gen('age', regex='[0-9]+', stop=',')},
# #         "class": "{gen('class', stop='"')}",
# #         "mantra": "{gen('mantra', stop='"')}",
# #         "strength": {gen('strength', regex='[0-9]+', stop=',')},
# #         "items": ["{gen('item', list_append=True, stop='"')}", "{gen('item', list_append=True, stop='"')}", "{gen('item', list_append=True, stop='"')}"]
# #     }}```"""

# with assistant():
#     character_description = "A beserker who is a master of the sword."
#     lm += f"""\
#     The following is a character profile for an RPG game in JSON format.
#     ```json
#     {{
#         "id": "{id}",
#         "description": "{character_description}",
#         "name": "{gen('name', stop='"')}",
#         "age": {gen('age', regex='[0-9]+', stop=',')},
#         "class": "{gen('class', stop='"')}",
#         "mantra": "{gen('mantra', stop='"')}",
#         "strength": {gen('strength', regex='[0-9]+', stop=',')},
#         "items": ["{gen('item', list_append=True, stop='"')}", "{gen('item', list_append=True, stop='"')}", "{gen('item', list_append=True, stop='"')}"]
#     }}```"""
    
# print(lm)


from guidance import models, gen
import guidance
gpt = models.Transformers('microsoft/phi-2', trust_remote_code=True)
# gpt = models.Transformers('gpt2')
# gpt = models.Transformers('meta-llama/Llama-2-7b-chat-hf')
# gpt = models.Transformers('intfloat/e5-mistral-7b-instruct')
prompt = 'http:'
print("generating 1...")


@guidance
def uuid(lm):
    # generate a random uuid
    import uuid
    lm += str(uuid.uuid4())
    return lm

character_description = "A beserker who is a master of the sword."
# gpt += f"""\
# The following is a character profile for an RPG game in JSON format.
# ```json
# {{
#     "id": "{uuid()}",
#     "description": "{character_description}",
#     "name": "{gen('name', stop='"')}",
#     "age": {gen('age', regex='[0-9]+', stop=',')},
#     "class": "{gen('class', stop='"')}",
#     "mantra": "{gen('mantra', stop='"')}",
#     "strength": {gen('strength', regex='[0-9]+', stop=',')},
#     "items": ["{gen('item', list_append=True, stop='"')}", "{gen('item', list_append=True, stop='"')}", "{gen('item', list_append=True, stop='"')}"]
# }}```"""

# print("Answer 2: ", gpt)

# Generate a list of 3 documents with 3 sentences each
doc1 = """DOCUMENT 1:\n####\nI was born in 1990. I am 30 years old. I live in New York City.\n####\n"""
doc2 = """DOCUMENT 2:\n####\nMy dog is named Max. He is a golden retriever. He is 5 years old.\n####\n"""
doc3 = """DOCUMENT 3:\n####\nI like to play soccer. I play for the New York City soccer club. I am a striker.\n####\n"""

DOCUMENTS = [doc1, doc2, doc3]

QUESTION = "How old am I?"

prompt = """Instruct: Given the following QUESTION and DOCUMENTS below, please rate each document based on its relavance to the QUESTION. Your response should be a JSON object with the following properties:
- doc: the document number
- relevance_analysis: a detailed analysis of why the document is relevant or not relevant to the question. Be specific.
- relevance_score: a score from 0 to 9 indicating the relevance of the document to the question

DOCUMENTS:
---------------
{DOCUMENTS}
---------------

QUESTION:
---------------
{QUESTION}
---------------

Output:
"""

def format_doc_response(doc):
    # extract DOCUMENT # from doc string
    doc = doc.split("\n")[0].split(" ")[1]
    return f"""\
    {{
        "doc": "{doc}",
        "relevance_analysis": "{gen('relevance_analysis', stop='"')}",
        "relevance_score": {gen('relevance_score', regex='[0-9]+', stop=',')}
    }}"""

formatted_prompt = prompt.format(DOCUMENTS="\n".join(DOCUMENTS), QUESTION=QUESTION)

response_str = """\
```json
{{
    "response": [
        {formatted_doc_responses}
    ]
}}```""".format(formatted_doc_responses=", ".join([format_doc_response(doc) for doc in DOCUMENTS]))

gpt += formatted_prompt + response_str

print("Analysis: ", gpt)