from dataclasses import dataclass, field
from typing import Any


@dataclass
class ModelConfig:
    client_model: str = None
    local_model: Any = None