from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class ModelConfig:
    client_model: Optional[str] = None
    local_model: Optional[Any] = None