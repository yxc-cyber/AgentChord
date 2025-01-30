from typing import Any, Callable, Type

from ..environment import BaseEnvironment
from ..metadata import BaseMetaData
from ..utils import Action, Logger, SubsystemSequence


class BaseAgentSystem:
    def __init__(self, system_name: str, environment: BaseEnvironment, maximum_loops: int = 50, log_name: str = ""):
        self.subsystems = dict()
        self.on_start_actions = dict()
        self.on_completion_actions = dict()
        self.subsystem_sequence = SubsystemSequence()
        self.system_name = system_name
        self.environment = environment
        self.tool_descriptions = environment.get_tool_descriptions().values() if environment.get_tool_descriptions().values() else None
        self.maximum_loops = maximum_loops
        self.log_name = log_name
        self.logger = Logger(self.system_name, self.log_name)

    def completion_loop(self, meta_data: BaseMetaData) -> BaseMetaData:
        loop_counter = 0
        while not self.subsystem_sequence.is_done() and loop_counter < self.maximum_loops:
            current_subsystem_name = self.subsystem_sequence.get_current_subsystem_name()
            self.subsystem_sequence.update_next_subsystem()
            if current_subsystem_name in self.on_start_actions:
                meta_data = self.on_start_actions[current_subsystem_name](meta_data)
            self.subsystems[current_subsystem_name].completion_loop(meta_data)
            if current_subsystem_name in self.on_completion_actions:
                meta_data = self.on_completion_actions[current_subsystem_name](meta_data)
            if self.environment.is_done():
                self.subsystem_sequence.set_done()
            loop_counter += 1
        return meta_data
        
    def run(self, debug: bool = False, log_name: str = "") -> Any:
        self.log_initialization(debug, log_name)
        meta_data = self.on_initialization()
        meta_data = self.completion_loop(meta_data)
        final_data = self.on_finalization(meta_data)
        self.log_finalization(log_name)
        return final_data

    def add_subsystem(self, subsystem: Type["BaseAgentSystem"]):
        subsystem_name = subsystem.system_name
        if subsystem_name in self.subsystems:
            raise Exception(f"Subsystem {subsystem_name} is already registered!")
        self.subsystems[subsystem_name] = subsystem
        self.subsystem_sequence.append(subsystem_name)

    def del_subsystem(self, subsystem_name: str):
        if subsystem_name not in self.subsystems:
            raise Exception(f"Subsystem {subsystem_name} has not been registered yet!")
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
        meta_data = self.environment.get_initial_setup()
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

    def get_pipeline_description_list(self) -> list:
        description_list = list()
        for subsystem_name in self.subsystem_sequence:
            subsystem = self.subsystems[subsystem_name]
            if subsystem_name in self.on_start_actions:
                on_start_action = self.on_start_actions[subsystem_name]
                description_list.append(f"On Start Action: {on_start_action.action_name} ({on_start_action.__class__})")
            description_list.append(f"Agent System: {subsystem_name} ({subsystem.__class__})")
            description_list.extend(subsystem.get_pipeline_description_list())
            if subsystem_name in self.on_completion_actions:
                on_completion_action = self.on_completion_actions[subsystem_name]
                description_list.append(f"On Completion Action: {on_completion_action.action_name} ({on_completion_action.__class__})")
        description_list = ["- " + desc for desc in description_list]
        return description_list
    
    def get_pipeline_description(self) -> str:
        description_list = self.get_pipeline_description_list()
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