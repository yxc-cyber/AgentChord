from typing import Literal, Optional, Union

from litellm import Message

from ..gbc_object import GBCList, GBCStr

# Gradient strategy constants
FINEGRAINED = 0
SUM_SQUARES = 1
GRADIENT_STRATEGIES = {
    "finegrained": FINEGRAINED,
    "sum_squares": SUM_SQUARES,
}

# Connection strategy constants
MEAN_PRODUCT_INPUT = 0
MAX_PRODUCT_INPUT = 1
MEAN_L1_NORM = 2
MAX_L1_NORM = 3
CONNECTION_STRATEGIES = {
    "mean_product_input": MEAN_PRODUCT_INPUT,
    "max_product_input": MAX_PRODUCT_INPUT,
    "mean_l1_norm": MEAN_L1_NORM,
    "max_l1_norm": MAX_L1_NORM,
}

# Custom message class for handling GBC objects
class Message(Message):
    """
    Custom Message class that extends litellm.Message to handle GBC objects.
    It stores the GBC content in a separate attribute `gbc_content` for compatibility with litellm while storing the string representation in `content`.
    """
    """
        content: Optional[str] = None,
        role: Literal["assistant"] = "assistant",
        function_call=None,
        tool_calls: Optional[list] = None,
    """
    def __init__(
            self,
            content: Optional[Union[str, GBCStr]] = None,
            role: Literal["assistant"] = "assistant",
            function_call=None,
            tool_calls: Optional[list] = None,
            **kwargs,
        ):
        super().__init__(
            content=str(content) if isinstance(content, GBCStr) else content,
            role=role,
            function_call=function_call,
            tool_calls=list(tool_calls) if isinstance(tool_calls, GBCList) else tool_calls,
            **kwargs
        )
        self._gbc_content = content if isinstance(content, GBCStr) else None
        self._gbc_tool_calls = tool_calls if isinstance(tool_calls, GBCList) else None

    @property
    def gbc_content(self) -> Optional[GBCStr]:
        """
        Returns the GBC content if it exists, otherwise returns None.
        """
        return self._gbc_content
    
    @gbc_content.setter
    def gbc_content(self, value: Optional[GBCStr]) -> None:
        """
        Sets the GBC content.
        """
        if isinstance(value, GBCStr):
            self.content = str(value)
            self._gbc_content = value
        else:
            self._gbc_content = None

    @property
    def gbc_tool_calls(self) -> Optional[GBCList]:
        """
        Returns the GBC tool calls if they exist, otherwise returns None.
        """
        return self._gbc_tool_calls
    
    @gbc_tool_calls.setter
    def gbc_tool_calls(self, value: Optional[GBCList]) -> None:
        """
        Sets the GBC tool calls.
        """
        if isinstance(value, GBCList):
            self.tool_calls = list(value)
            self._gbc_tool_calls = value
        else:
            self._gbc_tool_calls = None



# Json parsing utilities
def parse_json_string(json_string: str) -> Optional[Union[dict, list]]:
    """
    Parse a JSON string and return the corresponding Python object.
    """
    import json
    import re
    def sanitize_invalid_escapes(s):
        # Replace \' with '
        s = re.sub(r"\\'", "'", s)
        # Replace all other invalid backslash escapes (except valid ones)
        s = re.sub(r'\\(?!["\\/bfnrtu])', r'\\\\', s)
        return s
    json_string = sanitize_invalid_escapes(json_string)
    def extract_from_markdown(json_string: str) -> str:
        """
        Extract JSON content from markdown format.
        """
        json_format = re.compile(r"```(json)?(.*)```", re.DOTALL)
        match = json_format.match(json_string)
        if match:
            return match.group(2).strip()
        return json_string
    json_string = extract_from_markdown(json_string)

    objects = list()
    start_idx = 0
    while start_idx < len(json_string):
        while start_idx < len(json_string) and json_string[start_idx] != '{' and json_string[start_idx] != '[':
            start_idx += 1
        if start_idx >= len(json_string):
            break
        try:
            obj, end_idx = json.JSONDecoder().raw_decode(json_string, idx=start_idx)
            objects.append(obj)
            start_idx = end_idx
        except json.JSONDecodeError:
            # If we can't decode, it might be due to multiple JSON objects or invalid format
            break
    if len(objects) == 1:
        return objects[0]
    elif len(objects) > 1:
        return objects
    else:
        return None
        

# Sanitize the output string by removing unwanted characters
def sanitize_output_string(output: str) -> str:
    """
    Clean the output string by removing unwanted characters.
    """
    import re

    # Remove leading/trailing whitespace and newlines
    cleaned = output.strip()
    # Remove <output> tags if present
    cleaned = re.sub(r"<\/?output>", "", cleaned)
    cleaned = cleaned.strip()
    return cleaned


# Singleton metaclass for ensuring a single instance of a class
class SingletonMeta(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        # Todo: control the behavior based on the configuration. Don't create a new instance if the configuration already exists.
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]