from dataclasses import dataclass, field

from .base_metadata import BaseMetaData


@dataclass
class MultiWOZ24MetaData(BaseMetaData):
    dialogue_state: dict = field(default_factory=dict)
    system_response: str = ""
    jga: float = 0.0
    slot_recall: float = 0.0
    slot_precision: float = 0.0