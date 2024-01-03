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
# gpt = models.Transformers('mistralai/Mistral-7B-v0.1')
# gpt = models.Transformers('gpt2')
# gpt = models.Transformers('meta-llama/Llama-2-7b-chat-hf')
gpt = models.Transformers('intfloat/e5-mistral-7b-instruct')
prompt = 'http:'
print("generating 1...")
gpt = gpt + prompt + gen(max_tokens=10)

# print("Answer 1: ", gpt)

# gpt = None

# gpt = models.Transformers('mistralai/Mistral-7B-Instruct-v0.2')
# prompt = 'http:'
# print("generating 2...")
# gpt + prompt + gen(max_tokens=10)

print("Answer 2: ", gpt)