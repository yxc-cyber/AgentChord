from dataclasses import dataclass, field


@dataclass
class BaseMetaData:
    input: str = ""
    tool: dict = field(default_factory=dict)
    output: str = ""