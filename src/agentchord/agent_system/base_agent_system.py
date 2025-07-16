from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Self

from ..environment import BaseEnvironment, InnerEnvironment
from ..metadata import BaseMetaData
from ..sequence import BaseSequence
from ..utils import Action, Logger

if TYPE_CHECKING:
    from .base_agent import BaseAgent

import json


class BaseAgentSystem:
    def __init__(
            self,
            system_name: str,
            environment: BaseEnvironment,
            tools: Optional[List[str]] = None,
            maximum_loops: int = 50,
            log_name: str = "",
        ):
        self.subsystems = dict()
        self.on_start_actions = dict()
        self.on_completion_actions = dict()
        self.subsystem_sequence = BaseSequence()
        self.system_name = system_name
        self.maximum_loops = maximum_loops
        self.log_name = log_name
        self.logger = Logger(self.system_name, self.log_name)
        self.parent_number = 0
        self.tools = tools
        self.tool_descriptions = None
        self.environment = None
        self.inner_environment = None
        self.set_environment(environment)
        self.set_inner_environment(InnerEnvironment())

    def execution_loop(self, meta_data: BaseMetaData, loop: bool = False) -> BaseMetaData:
        loop_counter = 0
        while loop_counter < self.maximum_loops:
            self.subsystem_sequence.set_not_done()
            while not self.subsystem_sequence.is_done():
                # Move subsystem sequence forward
                current_subsystem_name = self.subsystem_sequence.get_current_subsystem_name()
                self.subsystem_sequence.update_next_subsystem()
                # Trigger on-start events
                if current_subsystem_name in self.on_start_actions:
                    meta_data = self.on_start_actions[current_subsystem_name](meta_data)
                # Trigger child completion event
                meta_data = self.subsystems[current_subsystem_name].execution_loop(meta_data)
                # Trigger on-completion event
                if current_subsystem_name in self.on_completion_actions:
                    meta_data = self.on_completion_actions[current_subsystem_name](meta_data)
                # Early termination if the environment is done
                if self.environment.is_done():
                    self.subsystem_sequence.set_done()
            # Only the root system controls the looping
            if self.parent_number > 0 or not loop or self.environment.is_done():
                break
            loop_counter += 1
        return meta_data
        
    def run(self, debug: bool = False, log_name: str = "", loop: bool = False) -> Any:
        self.log_initialization(debug, log_name)
        meta_data = self.on_initialization()
        meta_data = self.execution_loop(meta_data, loop)
        final_data = self.on_finalization(meta_data)
        self.log_finalization(log_name)
        return final_data

    def set_environment(self, environment: BaseEnvironment):
        self.environment = environment
        self.set_tool_descriptions()
        for subsystem_name in self.subsystem_sequence:
            subsystem = self.subsystems[subsystem_name]
            subsystem.set_environment(environment)

    def set_inner_environment(self, inner_environment: InnerEnvironment):
        self.inner_environment = inner_environment
        self.set_tool_descriptions()
        for subsystem_name in self.subsystem_sequence:
            subsystem = self.subsystems[subsystem_name]
            subsystem.set_inner_environment(inner_environment)

    def set_tool_descriptions(self):
        if self.tools is not None and self.environment is not None and not set(self.tools).issubset(set(self.environment.get_tool_descriptions().keys())):
            raise Exception(
                f"""Tools {", ".join(set(self.tools).difference(set(self.environment.get_tool_descriptions().keys())))} are not registered in the environment!"""
                f""" Available tools are {", ".join(self.environment.get_tool_descriptions().keys())}."""
            )
        if self.environment and self.environment.get_tool_descriptions() and self.inner_environment and self.inner_environment.get_tool_descriptions():
            if self.tools is not None:
                environment_tool_names = set(self.tools)
            else:
                environment_tool_names = set(self.environment.get_tool_descriptions().keys())
            inner_environment_tool_names = set(self.inner_environment.get_tool_descriptions().keys())
            common_tool_names = environment_tool_names.intersection(inner_environment_tool_names)
            if common_tool_names:
                raise Exception(f"""Tool names {", ".join(common_tool_names)} are duplicated!""")
        if self.environment and self.environment.get_tool_descriptions():
            if self.tools is not None:
                self.tool_descriptions = [self.environment.get_tool_descriptions()[tool_name] for tool_name in self.tools]
            else:
                self.tool_descriptions = list(self.environment.get_tool_descriptions().values())
            if self.inner_environment and self.inner_environment.get_tool_descriptions():
                self.tool_descriptions.extend(list(self.inner_environment.get_tool_descriptions().values()))
        elif self.inner_environment and self.inner_environment.get_tool_descriptions():
            self.tool_descriptions = list(self.inner_environment.get_tool_descriptions().values())

    def add_subsystem(self, subsystem: Self):
        subsystem_name = subsystem.system_name
        if subsystem_name in self.subsystems:
            raise Exception(f"Subsystem {subsystem_name} is already registered!")
        subsystem.parent_number += 1
        self.subsystems[subsystem_name] = subsystem
        self.subsystem_sequence.append(subsystem_name)
        subsystem.set_environment(self.environment)
        subsystem.set_inner_environment(self.inner_environment)

    def del_subsystem(self, subsystem_name: str):
        if subsystem_name not in self.subsystems:
            raise Exception(f"Subsystem {subsystem_name} has not been registered yet!")
        self.subsystems[subsystem_name].parent_number -= 1
        del self.subsystems[subsystem_name]
        self.subsystem_sequence.remove(subsystem_name)

    def get_subsystem(self, subsystem_name: str):
        if subsystem_name not in self.subsystems:
            raise Exception(f"Subsystem {subsystem_name} has not been registered yet!")
        return self.subsystems[subsystem_name]
    
    def add_on_start_action(self, subsystem_name: str, action_name: str, action: Callable):
        if subsystem_name not in self.subsystems:
            raise Exception(f"Subsystem {subsystem_name} has not been registered yet!")
        if subsystem_name in self.on_start_actions:
            raise Exception(f"Action on start of {subsystem_name} is already registered!")
        self.on_start_actions[subsystem_name] = Action(action_name, action)

    def del_on_start_action(self, subsystem_name: str):
        if subsystem_name not in self.on_start_actions:
            raise Exception(f"Action on start of {subsystem_name} has not been registered yet!")
        del self.on_start_actions[subsystem_name]

    def add_on_completion_action(self, subsystem_name: str, action_name: str, action: Callable):
        if subsystem_name not in self.subsystems:
            raise Exception(f"Subsystem {subsystem_name} has not been registered yet!")
        if subsystem_name in self.on_completion_actions:
            raise Exception(f"Action on completion of {subsystem_name} is already registered!")
        self.on_completion_actions[subsystem_name] = Action(action_name, action)

    def del_on_completion_action(self, subsystem_name: str):
        if subsystem_name not in self.on_completion_actions:
            raise Exception(f"Action on completion of {subsystem_name} has not been registered yet!")
        del self.on_completion_actions[subsystem_name]

    def on_initialization(self) -> BaseMetaData:
        meta_data = self.environment.get_initial_metadata()
        return meta_data
    
    def on_finalization(self, meta_data: BaseMetaData) -> Any:
        final_data = meta_data
        return final_data
    
    def log_initialization(self, debug: bool, log_name: str):
        self.set_debug_level(debug)
        if log_name:
            self.set_log_redirection(log_name)

    def log_finalization(self, log_name: str):
        self.set_debug_level(False)
        if log_name:
            self.set_log_redirection(self.log_name)

    def _get_pipeline_description_list(self) -> list:
        description_list = list()
        for subsystem_name in self.subsystem_sequence:
            subsystem = self.subsystems[subsystem_name]
            if subsystem_name in self.on_start_actions:
                on_start_action = self.on_start_actions[subsystem_name]
                description_list.append(f"On Start Action: {on_start_action.action_name} ({on_start_action.__class__})")
            description_list.append(f"Agent System: {subsystem_name} ({subsystem.__class__})")
            description_list.extend(subsystem._get_pipeline_description_list())
            if subsystem_name in self.on_completion_actions:
                on_completion_action = self.on_completion_actions[subsystem_name]
                description_list.append(f"On Completion Action: {on_completion_action.action_name} ({on_completion_action.__class__})")
        description_list = ["- " + desc for desc in description_list]
        return description_list
    
    def get_pipeline_description(self) -> str:
        description_list = self._get_pipeline_description_list()
        result_list = [f"Agent System: {self.system_name} ({self.__class__})"] + description_list
        return "\n".join(result_list)
    
    def set_debug_level(self, debug: bool):
        self.logger.set_debug_level(debug)
        for subsystem_name in self.subsystem_sequence:
            subsystem = self.subsystems[subsystem_name]
            subsystem.set_debug_level(debug)

    def set_log_redirection(self, file_name: str):
        self.logger.set_log_redirection(file_name)
        for subsystem_name in self.subsystem_sequence:
            subsystem = self.subsystems[subsystem_name]
            subsystem.set_log_redirection(file_name)

    def get_agents(self) -> Dict[str, "BaseAgent"]:
        agents = dict()
        for subsystem_name in self.subsystem_sequence:
            subsystem = self.subsystems[subsystem_name]
            agents.update(subsystem.get_agents())
        return agents
    
    def save_agents(self, file_name: str = "agents.json"):
        """
        Save the agents to a file.
        :param file_name: The name of the file to save the agents to.
        """
        agents = self.get_agents()
        prompts = {agent_name: agent.get_prompt() for agent_name, agent in agents.items()}
        with open(file_name, 'w') as file:
            json.dump(prompts, file, indent=2)

    def load_agents(self, file_name: str = "agents.json"):
        """
        Load the agents from a file.
        :param file_name: The name of the file to load the agents from.
        """
        with open(file_name, 'r') as file:
            prompts = json.load(file)
        for agent_name, prompt in prompts.items():
            if agent_name in self.get_agents():
                self.get_agents()[agent_name].set_prompt(prompt)
            else:
                raise Exception(f"Agent {agent_name} not found in the system.")

    def __repr__(self):
        return self.system_name