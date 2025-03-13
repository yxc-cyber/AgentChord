from .base_model import BaseModel
from .model_config import ModelConfig


class ModelFactory:
    def __init__(self, config: ModelConfig):
        self.config = config

    def create_model(self) -> BaseModel:
        if self.config.client_model:
            return BaseModel(self.config)
        else:
            raise NotImplementedError