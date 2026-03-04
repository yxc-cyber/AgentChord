from dataclasses import dataclass, field
from typing import List, Literal, Optional, Union

from .base_metadata import BaseMetaData


@dataclass
class TaubenchMetaData(BaseMetaData):
    """Metadata for a single TauBench task evaluation."""

    responses: List[str] = field(default_factory=list)  # List of agent responses (for multi-turn tasks)
    user_responses: List[str] = field(default_factory=list)  # List of user responses (for multi-turn tasks)
    instruction: str = ""        # The user instruction / task description
    wiki: str = ""               # Wiki information for the task
    rules: str = ""              # Rules for the task
    reward: float = 0.0          # Final binary reward (0.0 or 1.0)
    reward_details: dict = field(default_factory=dict)  # Detailed reward information (e.g., {"groundtruth_actions":[], "actions": [], "action_match": True, "groundtruth_outputs": [], "responses": [], "output_match": False})
    mean_reward: float = 0.0     # Mean reward across tasks (used in evaluate_test_cases)
    total_tasks: int = 0         # Number of evaluated tasks (used in evaluate_test_cases)
