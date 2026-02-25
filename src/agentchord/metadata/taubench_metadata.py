from dataclasses import dataclass, field
from typing import List, Literal, Optional, Union

from .base_metadata import BaseMetaData


@dataclass
class TaubenchMetaData(BaseMetaData):
    """Metadata for a single TauBench task evaluation."""

    instruction: str = ""        # The user instruction / task description
    task_idx: int = 0            # Index of the task within the split
    domain: str = ""             # "airline" or "retail"
    task_split: str = "test"     # "test", "train", or "dev"
    reward: float = 0.0          # Final binary reward (0.0 or 1.0)
    mean_reward: float = 0.0     # Mean reward across tasks (used in evaluate_test_cases)
    total_tasks: int = 0         # Number of evaluated tasks (used in evaluate_test_cases)
