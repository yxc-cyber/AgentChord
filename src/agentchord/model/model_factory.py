from .base_model import BaseModel
from .model_config import ModelConfig


class ModelFactory:
    def __init__(self, config: ModelConfig):
        self.config = config

    def create_model(self) -> BaseModel:
        if self.config.client_model:
            return BaseModel(self.config)
        elif self.config.local_model:
            if self.config.local_model == "LlamaModel":
                from .llama_model import LlamaModel
                return LlamaModel(self.config)
            else:
                raise ValueError(f"Unsupported local model: {self.config.local_model}")
        else:
            raise ValueError("No model configuration provided. Please specify either client_model or local_model.")