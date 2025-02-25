from dataclasses import dataclass, field


@dataclass
class BaseMetaData:
    input: str = ""
    note: str = ""
    tool: dict = field(default_factory=list)
    output: str = ""