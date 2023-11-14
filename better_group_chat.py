

from typing import Dict, List, Optional, Union
from autogen import Agent, GroupChat, ConversableAgent
import logging

logger = logging.getLogger(__name__)

class BetterGroupChat(GroupChat):

    def select_speaker_msg(self, agents: List[Agent]):
        """Return the message for selecting the next speaker."""
        return f"""You are an expert at managine group conversations. You excel at following a conversation and determining who is best suited to be the next speaker. You are currently managing a conversation between a group of AGENTS. The current AGENTS and their role descriptions are as follows:

AGENTS:
{self._participant_roles()}

Read the following conversation.
Then select the next agent from {[agent.name for agent in agents]} to speak. Only return the agent."""
    

    def _participant_roles(self):
        roles = []
        for agent in self.agents:
            if agent.system_message.strip() == "":
                logger.warning(
                    f"The agent '{agent.name}' has an empty system_message, and may not work well with GroupChat."
                )
            roles.append(f"{'-' * 50}\n" + f"NAME: {agent.name}\nDESCRIPTION: {agent.system_message}" + f"\n{'-' * 50}")
        return "\n".join(roles)




    def select_speaker(self, last_speaker: Agent, selector: ConversableAgent):
        """Select the next speaker."""
        if self.func_call_filter and self.messages and "function_call" in self.messages[-1]:
            # find agents with the right function_map which contains the function name
            agents = [
                agent for agent in self.agents if agent.can_execute_function(self.messages[-1]["function_call"]["name"])
            ]
            if len(agents) == 1:
                # only one agent can execute the function
                return agents[0]
            elif not agents:
                # find all the agents with function_map
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

        print("SELECTOR: ", selector.name)
        print("SELECTOR_SYSTEM_MESSAGE: ", selector.system_message)
        print("MESSAGES: ", self.messages
            + [
                {
                    "role": "system",
                    "content": f"Read the above conversation. Then select the next agent from {[agent.name for agent in agents]} to play. Only return the agent.",
                }
            ])
        final, name = selector.generate_oai_reply(
            self.messages
            + [
                {
                    "role": "system",
                    "content": f"Read the above conversation. Then select the next agent from {[agent.name for agent in agents]} to play. Only return the agent.",
                }
            ]
        )
        print("FINAL: ", final)
        print("NAME: ", name)
        if not final:
            # i = self._random.randint(0, len(self._agent_names) - 1)  # randomly pick an id
            return self.next_agent(last_speaker, agents)
        try:
            return self.agent_by_name(name)
        except ValueError:
            logger.warning(
                f"GroupChat select_speaker failed to resolve the next speaker's name. Speaker selection will default to the next speaker in the list. This is because the speaker selection OAI call returned:\n{name}"
            )
            return self.next_agent(last_speaker, agents)
