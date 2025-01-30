from typing import Callable

from ..metadata import BaseMetaData


class BaseEnvironment:
    def __init__(self, initial_setup: BaseMetaData):
        self.tool_handlers = dict()
        self.tool_descriptions = dict()
        self.initial_setup = initial_setup
        self.done = False
        
    def get_tool_descriptions(self) -> dict:
        return self.tool_descriptions
    
    def get_initial_setup(self) -> BaseMetaData:
        return self.initial_setup
    
    def register_tool(self, tool_name: str, tool_description: str, tool_handler: Callable):
        if tool_name in self.tool_handlers or tool_name in self.tool_descriptions:
            raise Exception(f"Tool {tool_name} is already registered!")
        self.tool_handlers[tool_name] = tool_handler
        self.tool_descriptions[tool_name] = tool_description
    
    def apply_tool(self, tool_name: str, tool_arguments: dict):
        return self.tool_handlers[tool_name](**tool_arguments)
    
    def set_done(self):
        self.done = True

    def is_done(self) -> bool:
        return self.done