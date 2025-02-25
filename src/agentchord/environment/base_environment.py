from typing import Callable, Union

from ..metadata import BaseMetaData


class BaseEnvironment:
    @classmethod
    def iterate_test_cases(cls):
        pass

    def __init__(self, initial_metadata: Union[BaseMetaData, None] = None):
        self.tool_handlers = dict()
        self.tool_descriptions = dict()
        self.initial_metadata = initial_metadata if initial_metadata is not None else BaseMetaData()
        self.done = False
        
    def get_tool_descriptions(self) -> dict:
        return self.tool_descriptions
    
    def set_initial_metadata(self, initial_metadata: BaseMetaData):
        self.initial_metadata = initial_metadata
    
    def get_initial_metadata(self) -> BaseMetaData:
        return self.initial_metadata
    
    def register_tool(self, tool_name: str, tool_description: dict, tool_handler: Callable):
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
    
    def evaluate(self):
        pass