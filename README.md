# AutoGen AGI

## Introduction
AutoGen AGI focuses on advancing the AutoGen framework for multi-agent conversational systems, with an eye towards characteristics of Artificial General Intelligence (AGI). This project introduces modifications to AutoGen, enhancing group chat dynamics among autonomous agents and increasing their proficiency in complex tasks. The aim is to explore and incrementally advance agent behaviors, aligning them more closely with elements reminiscent of AGI.


## Features
- **Enhanced Group Chat**: Modified AutoGen classes for advanced group chat functionalities.
- **Agent Council**: Utilizes a council of agents for decision-making and speaker/actor selection. Based on a prompting technique explored in [this blog post](https://www.prompthub.us/blog/exploring-multi-persona-prompting-for-better-outputs).
- **Conversation Continuity**: Supports loading and continuation of chat histories.
- **Agent Team Awareness**: Each agent is aware of its role and the roles of its peers, enhancing team-based problem-solving.
- **Advanced RAG**: Built in Retrieval Augmented Generation (RAG) leveraging RAG-fusion and llm re-ranking.
- **Domain Discovery**: Built in domain discovery for knowledge outside of llm training data.
- **Custom Agents**: A growing list of customized agents.


## Demo Transcript
For an example output see this post: https://www.reddit.com/r/AutoGenAI/comments/186cft9/autogen_executing_autogen/

## WARNING
This project leverages agents that have access to execute code locally. In addition it is based on the extended context window of gpt-4-turbo, which can be costly. Proceed at your own risk.

## Installation
- (optional) create a conda environment: `conda create --name autogen-agi python=3.11`
- install dependencies: `pip install -r requirements.txt`
- add environment variables
  - copy `.env.example` to `.env` and fill in your values
  - copy `OAI_CONFIG_LIST.json.example` to `OAI_CONFIG_LIST.json` and fill in your OPENAI_API_KEY (this will most likely be needed for the example task)
  
## Getting Started
- To attempt to reproduce the functionality seen [in the demo](https://www.prompthub.us/blog/exploring-multi-persona-prompting-for-better-outputs):
  ```bash
  python autogen_modified_group_chat.py
  ```
- If you would first like to see an example of the research/domain discovery functionality:
  ```bash
  python example_research.py
  ```
- If you want to see an example of the RAG functionality:
  ```bash
  python example_rag.py
  ```
- If you want to compare the demo functionality to standard autogen:
  ```bash
  python autogen_standard_group_chat.py
  ```


## Contributing
Contributions are welcome! Please read our contributing guidelines for instructions on how to make a contribution.

## License
[Your chosen License]
