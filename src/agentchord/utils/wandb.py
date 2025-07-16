from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional, Sequence, Union


@dataclass
class WandBConfig:
    """
    Configuration for Weights & Biases logging.
    """
    project: str = ""
    entity: str = ""
    name: str = ""
    config: Optional[Union[Dict[str, Any], str]] = None
    tags: Optional[Sequence[str]] = None
    group: Optional[str] = None
    job_type: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the WandBConfig to a dictionary.
        """
        return asdict(self)