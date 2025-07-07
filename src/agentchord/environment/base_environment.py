import copy
import json
from typing import Callable, Optional

from ..metadata import BaseMetaData
from .utils import RESULT_TOOL_ARGS_ERROR, RESULT_TOOL_NAME_ERROR


class BaseEnvironment:
    evaluation_record = dict()
    pre_initialized = False

    @classmethod
    def pre_initialize(cls):
        """
        Pre-initializes the environment, setting up necessary configurations or resources.
        This method should be overridden by subclasses to provide specific pre-initialization logic.
        """
        if not cls.pre_initialized:
            cls.pre_initialized = True

    @classmethod
    def iterate_test_cases(cls):
        cls.pre_initialize()

    @classmethod
    def evaluate_test_cases(cls):
        cls.pre_initialize()

    def __init__(self, initial_metadata: Optional[BaseMetaData] = None):
        self.pre_initialize()
        self.tool_handlers = dict()
        self.tool_descriptions = dict()
        self.initial_metadata = initial_metadata if initial_metadata is not None else BaseMetaData()
        self.done = False
        
    def get_tool_descriptions(self) -> dict:
        return self.tool_descriptions
    
    def set_initial_metadata(self, initial_metadata: BaseMetaData):
        self.initial_metadata = initial_metadata
    
    def get_initial_metadata(self) -> BaseMetaData:
        return copy.deepcopy(self.initial_metadata)
    
    def register_tool(self, tool_name: str, tool_description: dict, tool_handler: Callable):
        if tool_name in self.tool_handlers or tool_name in self.tool_descriptions:
            raise Exception(f"Tool {tool_name} is already registered!")
        self.tool_handlers[tool_name] = tool_handler
        self.tool_descriptions[tool_name] = tool_description
    
    def apply_tool(self, tool_name: str, tool_arguments: dict) -> str:
        if tool_name not in self.tool_handlers:
            return json.dumps(RESULT_TOOL_NAME_ERROR)
        try:
            return self.tool_handlers[tool_name](**tool_arguments)
        except Exception as e:
            return json.dumps(RESULT_TOOL_ARGS_ERROR(e))
    
    def set_done(self):
        self.done = True

    def is_done(self) -> bool:
        return self.done
    
    def evaluate(self):
        pass