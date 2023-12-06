import autogen

from autogen.agentchat import GroupChat, GroupChatManager

from predifined_agents import (
    user_proxy,
    code_reviewer,
    agent_awareness_expert,
    python_expert,
    function_calling_agent,
    agi_gestalt_agent,
    code_execution_agent,
    creative_solution_agent,
    first_principles_thinker_agent,
    out_of_the_box_thinker_agent,
    strategic_planning_agent,
    project_manager_agent,
    efficiency_optimizer_agent,
    emotional_intelligence_expert_agent,
    task_history_review_agent,
    task_comprehension_agent,
)

from dotenv import load_dotenv

load_dotenv()

import os

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

llm_config4 = {
    "seed": 42,
    "config_list": config_list4,
    "temperature": 0.1,
}


AGENT_TEAM = [
    user_proxy,
    code_reviewer,
    agent_awareness_expert,
    python_expert,
    function_calling_agent,
    # agi_gestalt_agent,
    creative_solution_agent,
    first_principles_thinker_agent,
    # out_of_the_box_thinker_agent,
    # strategic_planning_agent,
    project_manager_agent,
    # efficiency_optimizer_agent,
    # emotional_intelligence_expert_agent,
    task_history_review_agent,
    task_comprehension_agent
]

groupchat = GroupChat(
    agents=AGENT_TEAM,
    messages=[],
    max_round=100,
)
manager = GroupChatManager(groupchat=groupchat, llm_config=llm_config4)

# message = """Please write a python script that prints 10 dad jokes and save it."""

# message = """Please execute the file to show that it works."""

# message = """Please review the code in the "code_to_improve" directory and improve it with your best judgement. This includes (but is not limited to) re-arranging the code and directory structure if needed, adding comments, and re-naming functions if needed. I don't think you will be able to run the code, but please do your best to make sure there are no syntax errors and it is complieable. Save your results in the "code_improved" directory."""

# message = """
# We are trying to automate a book writing process using AI. Give us 3 overarching things to consider to make it effective."""

# message = "Please show me a complete example of how to build automomous agents using the autogen python library."

# message = "Please generate 5 dad jokes and save them to a file. Then write a python script that outputs a random dad joke each time it is run."

# message = """I need a python function that takes in a list of overlapping chunks of text like the examples below and accurately stiches them together to form a cohesive single text. The trick is that the overlapping portions may have slightly different words or puncutation, so we will need to identify the largest string match in the overlapped portions and use the location of the match to stitch the texts together. You can assume that the overlapping portions will always be at the beginning and end of the text, and that the overlapping portions will always be the same length (say 500 characters). Below are two example chunks:

# Chunk 1:
# ```
# And I just want to say this here because, you know, we work with a lot of people, and they kind of have expectations, or they have heard stories about people, and they kind of looking for these shifts or bliss or whatever. And it's just so different for everybody. It's very different. Like every person's unique. So yeah, yeah.\n\nI think that's really good to bring that up and talk a little bit about your experience of bliss even because it's almost like people have this expectation that they should be like, you know, bliss out of their minds all day long, you know, sitting drooling on themselves, and you know, in this fuzzy state. Is that your experience?\n\nAnd so I started to experience bliss coming in and out in unity. Before I didn't, maybe just like in a meditation or very, very short periods. And then in unity, it was more sometimes you had a content. So when I say it's a content, it's like joy or love. But most of the time, it's empty. It's like so empty. Bliss is basically just the whole body vibrates. It's vibration in every cell of the body. And like right now, I experience it all the time, 24 hours, seven days a week. It's there all the time. It's just that vibration. It's without any content, but it's like it's just there. It's just the bliss of existence. I don't think about it. It's, you know, it's just.\n\nSo people are looking for the bliss, but the bliss is just the byproduct of the awakening process. So, and it depends on the system. Some systems, they have it. Some systems just don't, or a smaller amount. Yeah, yeah.\n\nSo I would describe that as the bliss of being. Just that's what you are, consciousness knowing itself as consciousness. It's just a very nice resting in brilliance and stillness. And it's a bliss, but it's not a bliss where, you know, it's not an intoxicating bliss. You can...
# ```

# Chunk 2:
# ```
# People are looking for the bliss, but the bliss is just the byproduct of the awakening process. So, and it depends on the system, some systems they have it, some systems just don't, or a smaller amount. Yeah, yeah. So I would describe that as the bliss of being, just that's what you are, consciousness knowing itself as consciousness. It's just a very nice resting in brilliance and stillness, and it's a bliss, but it's not a bliss where, you know, it's not an intoxicating bliss. You can be like, at least for me, I'm fully functional. I can like totally, you know, do all the human chores that need to be done. I can exercise, I can do all the things, and then come back down, sit here, open up my computer, talk with you, and then feel bliss. Yeah, yeah, it's just there all the time.\n\nBut like after, let's say one of your retreats, right? So we're under retreats, there's a lot of energy going on, a lot of peering and expanding and so on. Like a few days after your retreats, I felt a lot of bliss, for example. And even, okay, so I was sitting in the office, I tried to work, but I didn't because I just didn't care because the bliss felt so like... But this was only like, you know, therefore a few days. It's not that I was sitting in those bliss states for a long period of time.\n\nAnd just to add to that as well, because I did talk about, you know, having states of like extremely bliss, and that was the bliss of Kundalini, which it's a little bit different. It's, it almost feels like it's an external force that's like moving through your system, and you have no control over it. And as it's rising, it's like really, it is really intoxicating and blissful. But again, once the Kundalini rises and reaches the brain, it just becomes the bliss of being, the bliss of self.
# ```
# """

# message = """The function still cuts out big chunks of text. See the example chunks and example "stitched" text below:

# Chunk1:
# ```
# the whole body vibrates. It's vibration in every cell of the body. And like right now, I experience it all the time, 24 hours, seven days a week. It's there all the time. It's just that vibration. It's without any content, but it's like it's just there. It's just the bliss of existence. I don't think about it. It's, you know, it's just.\n\nSo people are looking for the bliss, but the bliss is just the byproduct of the awakening process. So, and it depends on the system. Some systems, they have it. Some systems just don't, or a smaller amount. Yeah, yeah.\n\nSo I would describe that as the bliss of being. Just that's what you are, consciousness knowing itself as consciousness. It's just a very nice resting in brilliance and stillness. And it's a bliss, but it's not a bliss where, you know, it's not an intoxicating bliss. You can...
# ```

# Chunk2:
# ```
# People are looking for the bliss, but the bliss is just the byproduct of the awakening process. So, and it depends on the system, some systems they have it, some systems just don't, or a smaller amount. Yeah, yeah. So I would describe that as the bliss of being, just that's what you are, consciousness knowing itself as consciousness. It's just a very nice resting in brilliance and stillness, and it's a bliss, but it's not a bliss where, you know, it's not an intoxicating bliss. You can be like, at least for me, I'm fully functional. I can like totally, you know, do all the human chores that need to be done. I can exercise, I can do all the things, and then come back down, sit here, open up my computer, talk with you, and then feel bliss. Yeah, yeah, it's just there all the time.\n\nBut like after, let's say one of your retreats, right?
# ```

# "stitched" text with missing text:
# ```
# It's vibration in every cell of the body. And like right now, I experience it all the time, 24 hours, seven days a week. It's there all the time. It's just that vibration. It's without any content, but it's like it's just there. It's just the bliss of existence. I don't think about it. It's, you know, it's just.

# Peopat least for me, I'm fully functional. I can like totally, you know, do all the human chores that need to be done. I can exercise, I can do all the things, and then come back down, sit here, open up my computer, talk with you, and then feel bliss. Yeah, yeah, it's just there all the time.
# ```
# """

message = """I'm interested in building autonomous agents using the autogen python library. Can you show me a complete example of how to do this? The example should show how to correctly configure and instantiate autogen automous agents. The request given to the agents will be: "Please write and then execute a python script that prints 10 dad jokes". I want the agents to run completely autonomously without any human intervention."""

user_proxy.initiate_chat(
    manager,
    clear_history=False,
    message=message,
)

# TODO: Add a function that allows injection of a new agent into the group chat.

# TODO: Add a function that allows spawning a new group chat with a new set of agents.
