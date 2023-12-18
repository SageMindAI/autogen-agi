<p align="center"><a><img width="367" height="367" src="https://github.com/metamind-ai/autogen-agi/assets/12631935/091b52b9-033c-48f6-ab61-1cff2e3f434f" alt="autogen-api logo"></a></p>



AutoGen AGI focuses on advancing the [AutoGen framework](https://github.com/microsoft/autogen) for multi-agent conversational systems, with an eye towards characteristics of Artificial General Intelligence (AGI). This project introduces modifications to AutoGen, enhancing group chat dynamics among autonomous agents and increasing their proficiency in robustly handling complex tasks. The aim is to explore and incrementally advance agent behaviors, aligning them more closely with elements reminiscent of AGI.


## Features
- **Enhanced Group Chat** 💬: Modified AutoGen classes for advanced group chat functionalities.
- **Agent Council** 🧙: Utilizes a council of agents for decision-making and speaker/actor selection. Based on a prompting technique explored in [this blog post](https://www.prompthub.us/blog/exploring-multi-persona-prompting-for-better-outputs).
- **Conversation Continuity** 🔄: Supports loading and continuation of chat histories.
- **Agent Team Awareness** 👥: Each agent is aware of its role and the roles of its peers, enhancing team-based problem-solving.
- **Advanced RAG** 📚: Built in Retrieval Augmented Generation (RAG) leveraging [RAG-fusion](https://towardsdatascience.com/forget-rag-the-future-is-rag-fusion-1147298d8ad1) and llm re-ranking implemented via [llama_index](https://www.llamaindex.ai/).
- **Domain Discovery** 🔍: Built in domain discovery for knowledge outside of llm training data.
- **Custom Agents** 🌟: A growing list of customized agents.


## Demo Transcript 📜
In the following link you can see some example output of the demo task, which is to get a team of agents to write and execute another team of autogen agents: 
- https://www.reddit.com/r/AutoGenAI/comments/186cft9/autogen_executing_autogen/


<p align="center"><a><img width="1080" height="1080" src="https://github.com/metamind-ai/autogen-agi/assets/12631935/9458b016-9ef3-4bc4-a9c1-127c5c250abc" alt="agent council demo"></a></p>
<p align="center">
<i>Example transcript of an "Agent Council" discussion 🧙</i>
</p>


## WARNING ⚠️
This project leverages agents that have access to execute code locally. In addition it is based on the extended context window of gpt-4-turbo, which can be costly. Proceed at your own risk.

## Installation 🛠️
- clone the project:
```bash
git clone git@github.com:metamind-ai/autogen-agi.git
cd autogen-agi
```
- (optional) create a conda environment:
```bash
conda create --name autogen-agi python=3.11
conda activate autogen-agi
```
- install dependencies
```bash
pip install -r requirements.txt
```
- add environment variables
  - copy `.env.example` to `.env` and fill in your values
    ```bash
    cp .env.example .env
    ```
  - copy `OAI_CONFIG_LIST.json.example` to `OAI_CONFIG_LIST.json` and fill in your OPENAI_API_KEY (this will most likely be needed for the example task)
    ```bash
    cp OAI_CONFIG_LIST.json.example OAI_CONFIG_LIST.json
    ```

All set!! 🎉✨
 
*NOTE*: 
- 🔴 visit [GitHub docs](https://docs.github.com/en/enterprise-server@3.9/authentication/keeping-your-account-and-data-secure/managing-your-personal-access-tokens) to get your GitHub personal access token (required)
- ✅ visit https://serpapi.com/ to get your own API key (optional)
- ✅ visit https://programmablesearchengine.google.com/controlpanel/create to get your own API key (optional)
  
## Getting Started 🚀
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

## Methodology 🔍
The evolution of this project has kept to a simple methodology so far. Mainly: 
1) Test increasingly complex tasks.
2) Observe the current limitations of the agents/framework.
3) Add specific agents/features to overcome those limitations.
4) Generalize features to be more scalable.

For an example of a future possible evolution: discover what team of agents seems most successful at accomplishing more and more complex tasks, then provide those agent prompts few-shot learning examples in a dynamic agent generation prompt.

## Contributing 🤝
Contributions are welcome! Please read our contributing guidelines for instructions on how to make a contribution.

## TODO 📝

- [ ] Expand research and discovery to support more resources (such as arxiv) and select the resource dynamically.
- [ ] Support chat history overflow. This would reflect a MemGPT like system where the overflow history would stay summarized in the context with relevant overflow data pulled in (via RAG) as needed.
- [ ] If possible, support smaller context windows and open source LLMs.
- [ ] Add ability to dynamically inject agents as needed.
- [ ] Add ability to spawn off agent teams as needed.
- [ ] Add support for communication and resource sharing between agent teams.


## Support ⭐
Love what we're building with AutoGen AGI? Star this project on GitHub! Your support not only motivates us, but each star brings more collaborators to this venture. More collaboration means accelerating our journey towards advanced AI and closer to AGI. Let's push the boundaries of AI together! ⭐

## News 📰
- [Check out our blog post for the project launch](https://medium.com/@headley.justin/from-autogpt-to-agi-the-evolutionary-journey-of-autogen-3fefee6d2cc0)!

## License

MIT License

Copyright (c) 2023 MetaMind Solutions

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

