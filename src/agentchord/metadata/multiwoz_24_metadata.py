from dataclasses import dataclass, field

from .base_metadata import BaseMetaData


@dataclass
class MultiWOZ24MetaData(BaseMetaData):
    dialogue_state: dict = field(default_factory=dict)