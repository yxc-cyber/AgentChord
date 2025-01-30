from typing import Any, Callable, Type

from ..environment import BaseEnvironment
from ..metadata import BaseMetaData
from ..utils import Action, SubsystemSequence


class BaseAgentSystem:
    def __init__(self, system_name: str, environment: BaseEnvironment):
        self.subsystems = dict()
        self.on_start_actions = dict()
        self.on_completion_actions = dict()
        self.subsystem_sequence = SubsystemSequence()
        self.system_name = system_name
        self.environment = environment
        self.tool_descriptions = environment.get_tool_descriptions().values() if environment.get_tool_descriptions().values() else None

    def completion_loop(self, meta_data: BaseMetaData) -> BaseMetaData:
        if not self.subsystem_sequence.is_done():
            current_subsystem_name = self.subsystem_sequence.get_current_subsystem_name()
            self.subsystem_sequence.update_next_subsystem()
            if current_subsystem_name in self.on_start_actions:
                meta_data = self.on_start_actions[current_subsystem_name](meta_data)
            self.subsystems[current_subsystem_name].completion_loop(meta_data)
            if current_subsystem_name in self.on_completion_actions:
                meta_data = self.on_completion_actions[current_subsystem_name](meta_data)
        return meta_data
        
    def start(self) -> Any:
        meta_data = self.on_initialization()
        meta_data = self.completion_loop(meta_data)
        final_data = self.on_finalization(meta_data)
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

    def get_pipeline_description_list(self, indent_level: int = 0) -> list:
        description_list = list()
        for subsystem_name, subsystem in self.subsystem_sequence.items():
            if subsystem_name in self.on_start_actions:
                description_list.append(f"On Start Action: {self.on_start_actions[subsystem_name].action_name}")
            description_list.append(f"Agent System: {subsystem_name} ({subsystem.__class__})")
            description_list.extend(subsystem.get_pipeline_description_list(indent_level+1))
            if subsystem_name in self.on_completion_actions:
                description_list.append(f"On Completion Action: {self.on_completion_actions[subsystem_name].action_name}")
        description_list = ["- "*indent_level + desc for desc in description_list]
        return description_list
    
    def get_pipeline_description(self) -> str:
        description_list = self.get_pipeline_description_list()
        return "\n".join(description_list)