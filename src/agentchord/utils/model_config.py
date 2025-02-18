from dataclasses import dataclass, field
from typing import Any


@dataclass
class ModelConfig:
    client_config: dict = field(default_factory=dict)
    local_model: Any = None