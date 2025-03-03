from dataclasses import dataclass, field


@dataclass
class BaseMetaData:
    input: str = ""
    note: str = ""
    tool: list = field(default_factory=list)
    output: str = ""
    sample_number: int = 0