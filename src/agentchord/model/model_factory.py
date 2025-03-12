from ..utils import ModelConfig
from .base_model import BaseModel


class ModelFactory:
    def __init__(self, config: ModelConfig):
        self.config = config

    def create_model(self):
        if self.config.client_model:
            return BaseModel(self.config)
        else:
            raise NotImplementedError