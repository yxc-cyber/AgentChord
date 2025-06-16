import json
from dataclasses import asdict, dataclass, field
from typing import List, Optional, Union


@dataclass
class BaseMetaData:
    input: Union[str, List[str]] = ""
    note: Union[str, List[str]] = ""
    tool: list = field(default_factory=list)
    output: str = ""
    sample_id: int = 0

    def to_json(self, file_name: Optional[str]) -> dict:
        result_json = {key: value for key, value in asdict(self).items() if value}
        if file_name:
            with open(file_name, "w") as file:
                json.dump(result_json, file, indent=4)
        return result_json