import json
from dataclasses import asdict, dataclass, field
from typing import List, Optional, Self, Union


@dataclass
class BaseMetaData:
    input: Union[str, List[str]] = ""
    note: Union[str, List[str]] = ""
    tool: list = field(default_factory=list)  # List of tool usage, each tool usage is a dict with keys: "tool_name", "tool_arguments" (dict), "tool_result"
    output: str = ""
    sample_id: int = 0

    def to_json(self, file_name: Optional[str]) -> dict:
        def convert_sets(obj):
            if isinstance(obj, set):
                return list(obj)
            raise TypeError(f"Type {type(obj)} not serializable")

        result_json = {key: value for key, value in asdict(self).items() if value}
        if file_name:
            with open(file_name, "w") as file:
                json.dump(result_json, file, indent=4, default=convert_sets)
        return result_json
    
    def load_non_basic_attributes(self, meta_data: Self) -> None:
        """
        Load non-basic attributes from another BaseMetaData instance.
        This is useful for copying attributes that are not part of the basic data structure.
        """
        for key, value in meta_data.__dict__.items():
            if key not in ["input", "note", "tool", "output", "sample_id"]:
                setattr(self, key, value)