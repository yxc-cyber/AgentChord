from dataclasses import dataclass, field
from typing import List, Union

from .base_metadata import BaseMetaData


@dataclass
class MultiWOZ24MetaData(BaseMetaData):
    grounding_utterance: str = ""
    groundtruth_dialogue_state: dict = field(default_factory=dict)
    dialogue_state: dict = field(default_factory=dict)
    system_response: Union[List[str], str] = ""
    delixicalized_system_response: str = ""
    inform: dict = field(default_factory=dict)
    inform_detail: dict = field(default_factory=dict)  # e.g., {"restaurant": {"requested": [{"name": "Pizza Place"}], "provided": [{"name": "Pizza Place"}]}}
    success: dict = field(default_factory=dict)
    success_detail: dict = field(default_factory=dict)  # e.g., {"restaurant": {"requested": ["name", "address"], "provided": ["name"]}}
    matched_turns: float = 0.0
    total_turns: float = 0.0
    true_positive: float = 0.0
    false_positive: float = 0.0
    false_negative: float = 0.0
    joint_goal_accuracy: float = 0.0
    joint_goal_accuracy_detail: dict = field(default_factory=dict)  # e.g., {"true_positive": {"restaurant-name": "Pizza Place"}, "false_positive": {restaurant-name": "Pizza Place"}, "false_negative": ["restaurant-name"]}
    slot_recall: float = 0.0
    slot_precision: float = 0.0
    slot_f1: float = 0.0