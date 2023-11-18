# filename: agent_better_group_chat.py
import logging
import sys
import re
import time
import json
import os
from typing import Dict, List, Optional, Union

from autogen import Agent, GroupChat, ConversableAgent, GroupChatManager
from src.agent_prompts import (
    AGENT_SYSTEM_MESSAGE,
    AGENT_DESCRIPTION_SUMMARIZER,
    DEFAULT_COVERSATION_MANAGER_SYSTEM_PROMPT,
    PERSONA_DISCUSSION_FOR_NEXT_STEP_SYSTEM_PROMPT,
    PERSONA_DISCUSSION_FOR_NEXT_STEP_PROMPT,
    EXTRACT_NEXT_ACTOR_FROM_DISCUSSION_PROMPT,
)
from src.utils.misc import light_llm4_wrapper, extract_json_response

# Configure logging with color using the 'colored' package
from colored import fg, bg, attr

# Define colors for different log types
COLOR_AGENT_COUNCIL_RESPONSE = fg("yellow") + attr("bold")
COLOR_GET_NEXT_ACTOR_RESPONSE = fg("green") + attr("bold")
COLOR_NEXT_ACTOR = fg("blue") + attr("bold")
COLOR_INFO = fg("blue") + attr("bold")
RESET_COLOR = attr("reset")

logger = logging.getLogger(__name__)


class BetterGroupChat(GroupChat):
    def __init__(
        self,
        agents: List[Agent],
        group_name: str = "GroupChat",
        continue_chat: bool = False,
        messages: List[Dict] = [],
        max_round: int = 10,
        admin_name: str = "Admin",
        func_call_filter: bool = True,
        summarize_agent_descriptions: bool = False,
        persona_discussion: bool = False,
        inject_persona_discussion: bool = False,
    ):
        super().__init__(agents, messages, max_round, admin_name, func_call_filter)

        self.group_name = group_name
        self.continue_chat = continue_chat
        self.summarize_agent_descriptions = summarize_agent_descriptions
        self.persona_discussion = persona_discussion
        self.inject_persona_discussion = inject_persona_discussion
        self.agent_descriptions = []
        self.agent_team_description = ""
        self.manager = None

        # Set start time to current time formatted like "2021-08-31_15-00-00"
        self.start_time = time.time()
        self.start_time = time.strftime(
            "%Y-%m-%d_%H-%M-%S", time.localtime(self.start_time)
        )

        # Load in the chat history if continue_chat is True
        # if self.continue_chat:
        #     self.load_chat_history(file_path=None)

        # Generate agent descriptions based on configuration
        for agent in agents:
            description = (
                light_llm4_wrapper(
                    AGENT_DESCRIPTION_SUMMARIZER.format(
                        agent_system_message=agent.system_message
                    )
                ).text
                if self.summarize_agent_descriptions
                else agent.system_message
            )

            self.agent_descriptions.append(
                {
                    "name": agent.name,
                    "description": description,
                    "llm_config": agent.llm_config,
                }
            )

        # Create a formatted string of the agent team list
        self.agent_team_list = [
            f"{'*' * 20}\nAGENT_NAME: {agent['name']}\nAGENT_DESCRIPTION: {agent['description']}\n{self.describe_agent_actions(agent)}{'*' * 20}\n"
            for agent in self.agent_descriptions
        ]

        # Introduce the agent team
        # agent_team_description = AGENT_TEAM_DESCRIPTION.format(
        #     agent_team_list="\n".join(self.agent_team_list)
        # )

        # AGENT_SYSTEM_MESSAGE_WITH_TEAM = f"{AGENT_SYSTEM_MESSAGE}\n\n{agent_team_description}"

        # Update each agent's system message with the team preface
        for agent in agents:
            # Create the agent_team_list again, but without the agent's own description (just say: "THIS IS YOU")
            agent_specific_team_list = [
                ""
                if agent.name == agent_description["name"]
                else f"{'*' * 100}\nAGENT_NAME: {agent_description['name']}\nAGENT_DESCRIPTION: {agent_description['description']}\n{self.describe_agent_actions(agent_description)}{'*' * 100}\n"
                for agent_description in self.agent_descriptions
            ]

            # Get the agent_description for the current agent
            agent_description = [
                agent_description
                for agent_description in self.agent_descriptions
                if agent_description["name"] == agent.name
            ][0]

            # Agent system message with team info
            agent_system_message = AGENT_SYSTEM_MESSAGE.format(
                agent_team_list="\n".join(agent_specific_team_list),
                agent_name=agent.name,
                agent_description=agent.system_message,
                agent_function_list=self.describe_agent_actions(agent_description),
            )

            agent.update_system_message(agent_system_message)

        # display each agent's system message
        # for agent in agents:
        #     print(
        #         f"{COLOR_INFO}AGENT_SYSTEM_MESSAGE:{RESET_COLOR}\n{agent.system_message}\n\n\n\n"
        #     )

    def describe_agent_actions(self, agent: ConversableAgent):
        callable_functions = agent["llm_config"].get("functions", False)

        if callable_functions:
            AGENT_FUNCTION_LIST = "AGENT_REGISTERED_FUNCTIONS:"
            for function in callable_functions:
                AGENT_FUNCTION_LIST += f"""
----------------------------------------
FUNCTION_NAME: {function["name"]}
FUNCTION_DESCRIPTION: {function["description"]}
FUNCTION_ARGUMENTS: {function["parameters"]}
----------------------------------------\n"""
            return AGENT_FUNCTION_LIST

        return ""

    def select_speaker_msg(self, agents: List[Agent]):
        """Return the system message for selecting the next speaker."""
        agent_team = self._participant_roles()
        agent_names = [agent.name for agent in agents]

        if self.persona_discussion:
            all_agent_functions = []
            # loop through each agent and get their functions
            for agent in agents:
                agent_functions = self.describe_agent_actions(
                    {"llm_config": agent.llm_config}
                )
                if agent_functions:
                    all_agent_functions.append(agent_functions)

            agent_functions = "\n".join(all_agent_functions)
            # Remove all instances of "AGENT_REGISTERED_FUNCTIONS:" from the agent_functions string
            agent_functions = agent_functions.replace("AGENT_REGISTERED_FUNCTIONS:", "")

            return PERSONA_DISCUSSION_FOR_NEXT_STEP_SYSTEM_PROMPT.format(
                agent_functions=agent_functions,
            )
        else:
            return DEFAULT_COVERSATION_MANAGER_SYSTEM_PROMPT.format(
                agent_team=agent_team,
                agent_names=agent_names,
            )

    def _participant_roles(self):
        roles = []
        for agent in self.agent_descriptions:
            if agent["description"].strip() == "":
                logger.warning(
                    f"The agent '{agent['name']}' has an empty description, and may not work well with GroupChat."
                )
            roles.append(
                f"{'-' * 100}\n"
                + f"NAME: {agent['name']}\nDESCRIPTION: {agent['description']}"
                + f"\n{'-' * 100}"
            )
        return "\n".join(roles)

    def select_speaker(self, last_speaker: Agent, selector: ConversableAgent):
        """Select the next speaker."""
        if (
            self.func_call_filter
            and self.messages
            and "function_call" in self.messages[-1]
        ):
            # Find agents with the right function_map which contains the function name
            agents = [
                agent
                for agent in self.agents
                if agent.can_execute_function(
                    self.messages[-1]["function_call"]["name"]
                )
            ]
            if len(agents) == 1:
                # Only one agent can execute the function
                return agents[0]
            elif not agents:
                # Find all the agents with function_map
                agents = [agent for agent in self.agents if agent.function_map]
                if len(agents) == 1:
                    return agents[0]
                elif not agents:
                    raise ValueError(
                        f"No agent can execute the function {self.messages[-1]['name']}. "
                        "Please check the function_map of the agents."
                    )
        else:
            agents = self.agents
            # Warn if GroupChat is underpopulated
            n_agents = len(agents)
            if n_agents < 3:
                logger.warning(
                    f"GroupChat is underpopulated with {n_agents} agents. Direct communication would be more efficient."
                )

        selector.update_system_message(self.select_speaker_msg(agents))

        get_next_actor_message = ""

        if self.persona_discussion:
            get_next_actor_content = PERSONA_DISCUSSION_FOR_NEXT_STEP_PROMPT.format(
                task_goal=self.messages[0]["content"],
                agent_team=self._participant_roles(),
                conversation_history=self.messages,
            )
        else:
            get_next_actor_content = f"Read the above conversation. Then select the next agent from {[agent.name for agent in agents]} to speak. Only return the JSON object with your 'analysis' and chosen 'next_actor'."

        get_next_actor_message = self.messages + [
            {
                "role": "system",
                "content": get_next_actor_content,
            }
        ]

        final, response = selector.generate_oai_reply(get_next_actor_message)
        print(
            f"{COLOR_AGENT_COUNCIL_RESPONSE}AGENT_COUNCIL_RESPONSE:{RESET_COLOR}\n{response}\n"
        )
        if self.persona_discussion:
            if self.inject_persona_discussion:
                # Inject the persona discussion into the message history
                header = f"####\nSOURCE_AGENT: AGENT_COUNCIL\n####"
                response = f"{header}\n\n" + response
                self.messages.append({"role": "system", "content": response})
                # Send the persona discussion to all agents
                for agent in self.agents:
                    selector.send(response, agent, request_reply=False, silent=True)

            extracted_next_actor = light_llm4_wrapper(
                EXTRACT_NEXT_ACTOR_FROM_DISCUSSION_PROMPT.format(
                    actor_options=[agent.name for agent in agents],
                    discussion=response,
                ),
                kwargs={
                    "additional_kwargs": {"response_format": {"type": "json_object"}}
                },
            )
            response_json = extract_json_response(extracted_next_actor)
            print(
                f"{COLOR_GET_NEXT_ACTOR_RESPONSE}GET_NEXT_ACTOR_RESPONSE:{RESET_COLOR} \n{response_json['analysis']}"
            )
            name = response_json["next_actor"]
        else:
            response_json = extract_json_response(response)
            name = response_json["next_actor"]
        if not final:
            return self.next_agent(last_speaker, agents)
        try:
            return self.agent_by_name(name)
        except ValueError:
            logger.warning(
                f"GroupChat select_speaker failed to resolve the next speaker's name. Speaker selection will default to the UserProxy if it exists, otherwise we defer to next speaker in the list. This is because the speaker selection OAI call returned:\n{name}"
            )
            # Check if UserProxy exists in the agent list.
            for agent in agents:
                # Check for "User" or "UserProxy" in the agent name
                if agent.name == "User" or agent.name == "UserProxy":
                    return self.agent_by_name(agent.name)

            return self.next_agent(last_speaker, agents)

    def save_chat_history(self):
        """
        Saves the chat history to a file.
        """

        # Snake case the groupchat name
        groupchat_name = self.group_name.lower().replace(" ", "_")

        # Create the groupchat_name directory if it doesn't exist
        if not os.path.exists(groupchat_name):
            os.mkdir(groupchat_name)

        # Define the file path
        file_path = f"{groupchat_name}_chat_history_{self.start_time}.json"

        # Save the file to the groupchat_name directory
        with open(f"{groupchat_name}/{file_path}", "w") as f:
            # Convert the messages to a JSON string with indents
            messages = json.dumps(self.messages, indent=4)
            f.write(messages)

    def load_chat_history(self, file_path):
        """
        Loads the chat history from a file.
        """
        file_directory = self.group_name.lower().replace(" ", "_")

        if not file_path:
            # Load in the list of files in the groupchat_name directory
            try:
                file_list = os.listdir(file_directory)
            except FileNotFoundError:
                # Warn that no history was loaded
                logger.warning(f"No chat history was loaded for {self.group_name}.")
                return

            # Check if the file list is empty
            if not file_list:
                # Warn that no history was loaded
                logger.warning(f"No chat history was loaded for {self.group_name}.")
                return

            # Sort the list of files and grab the most recent
            file_list.sort()
            file_path = file_list[-1]
            file_path = f"{file_directory}/{file_path}"
        else:
            # Define the file path
            file_path = f"{file_directory}/{file_path}"

            # Check if the file exists
            if not os.path.exists(file_path):
                raise Exception(f"File {file_path} does not exist.")

        # Load the file from the groupchat_name directory
        with open(file_path, "r") as f:
            messages = json.load(f)

        self.messages = messages

        if not self.manager:
            raise Exception(f"No manager for group: {self.group_name}.")

        # Set the messages for each agent
        for agent in self.agents:
            agent._oai_messages[self.manager] = messages
            self.manager._oai_messages[agent] = messages

        print(f"\n{COLOR_INFO}Chat history loaded for {self.group_name}{COLOR_INFO}\n")

    def set_manager(self, manager: Agent):
        """
        Sets the manager for the groupchat.
        """
        self.manager = manager


class BetterGroupChatManager(GroupChatManager):
    def __init__(
        self,
        groupchat: BetterGroupChat,
        name: Optional[str] = "chat_manager",
        max_consecutive_auto_reply: Optional[int] = sys.maxsize,
        human_input_mode: Optional[str] = "NEVER",
        system_message: Optional[str] = "Group chat manager.",
        **kwargs,
    ):
        super().__init__(
            name=name,
            groupchat=groupchat,
            max_consecutive_auto_reply=max_consecutive_auto_reply,
            human_input_mode=human_input_mode,
            system_message=system_message,
            **kwargs,
        )

        groupchat.set_manager(self)

        if groupchat.continue_chat:
            # Load in the chat history
            groupchat.load_chat_history(file_path=None)

        # Empty the self._reply_func_list
        self._reply_func_list = []
        self.register_reply(
            Agent,
            BetterGroupChatManager.run_chat,
            config=groupchat,
            reset_config=BetterGroupChat.reset,
        )
        # Allow async chat if initiated using a_initiate_chat
        # self.register_reply(
        #     Agent,
        #     BetterGroupChatManager.a_run_chat,
        #     config=groupchat,
        #     reset_config=BetterGroupChat.reset,
        # )

    def run_chat(
        self,
        messages: Optional[List[Dict]] = None,
        sender: Optional[Agent] = None,
        config: Optional[BetterGroupChat] = None,
    ) -> Union[str, Dict, None]:
        """Run a group chat."""

        groupchat = config

        if messages is None:
            if groupchat.continue_chat:
                messages = groupchat.messages
            else:
                messages = self._oai_messages[sender]

        message = messages[-1]
        speaker = sender
        for i in range(groupchat.max_round):
            # Set the name to speaker's name if the role is not function
            if message["role"] != "function":
                message["name"] = speaker.name

            groupchat.messages.append(message)
            # Broadcast the message to all agents except the speaker
            for agent in groupchat.agents:
                if agent != speaker:
                    self.send(message, agent, request_reply=False, silent=True)
            if i == groupchat.max_round - 1:
                # The last round
                break
            try:
                # Select the next speaker
                speaker = groupchat.select_speaker(speaker, self)
                print(f"{COLOR_NEXT_ACTOR}NEXT_ACTOR:{RESET_COLOR} {speaker.name}\n")
                # Let the speaker speak
                reply = speaker.generate_reply(sender=self)
            except KeyboardInterrupt:
                # Let the admin agent speak if interrupted
                if groupchat.admin_name in groupchat.agent_names:
                    # Admin agent is one of the participants
                    speaker = groupchat.agent_by_name(groupchat.admin_name)
                    reply = speaker.generate_reply(sender=self)
                else:
                    # Admin agent is not found in the participants
                    raise
            if reply is None:
                break

            # Check if reply is a string
            if isinstance(reply, str):
                header = f"####\nSOURCE_AGENT: {speaker.name}\n####"
                reply = self.remove_agent_pattern(reply)
                reply = f"{header}\n\n" + reply
            # The speaker sends the message without requesting a reply

            speaker.send(reply, self, request_reply=False)

            # Save the chat history to file after each round
            groupchat.save_chat_history()
            message = self.last_message(speaker)
        return True, None

    def remove_agent_pattern(self, input_string):
        """
        Removes the pattern "####\nSOURCE_AGENT: <Agent Name>\n####" from the input string.
        `<Agent Name>` is a placeholder and can vary.
        """
        # Define the regular expression pattern to match the specified string
        pattern = r"####\nSOURCE_AGENT: .*\n####"

        # Use regular expression to substitute the pattern with an empty string
        modified_string = re.sub(pattern, "", input_string)

        return modified_string
