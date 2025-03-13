from dataclasses import dataclass, field

from .base_metadata import BaseMetaData


@dataclass
class MultiWOZ24MetaData(BaseMetaData):
    grounding_utterance: str = ""
    dialogue_state: dict = field(default_factory=dict)
    system_response: str = ""
    delixicalized_system_response: str = ""
    inform: dict = field(default_factory=dict)
    success: dict = field(default_factory=dict)
    matched_turns: float = 0.0
    total_turns: float = 0.0
    true_positive: float = 0.0
    false_positive: float = 0.0
    false_negative: float = 0.0
    joint_goal_accuracy: float = 0.0
    slot_recall: float = 0.0
    slot_precision: float = 0.0
    slot_f1: float = 0.0