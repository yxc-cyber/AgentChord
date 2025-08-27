from dataclasses import dataclass
from typing import Any, List, Literal, Optional, Type, Union

from litellm import (
    ChatCompletionAudioParam,
    ChatCompletionModality,
    ChatCompletionPredictionContentParam,
)
from pydantic import BaseModel
from transformers import BitsAndBytesConfig


@dataclass
class ModelConfig:
    # LiteLLM configuration
    client_model: Optional[str] = None
    # Optional OpenAI params: see https://platform.openai.com/docs/api-reference/chat/create
    timeout: Optional[Union[float, int]] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    n: Optional[int] = None
    stream: Optional[bool] = None
    stream_options: Optional[dict] = None
    max_tokens: Optional[int] = None
    max_completion_tokens: Optional[int] = None
    modalities: Optional[List[ChatCompletionModality]] = None
    prediction: Optional[ChatCompletionPredictionContentParam] = None
    audio: Optional[ChatCompletionAudioParam] = None
    presence_penalty: Optional[float] = None
    frequency_penalty: Optional[float] = None
    logit_bias: Optional[dict] = None
    user: Optional[str] = None
    # openai v1.0+ new params
    response_format: Optional[Union[dict, Type[BaseModel]]] = None
    seed: Optional[int] = None
    parallel_tool_calls: Optional[bool] = None
    logprobs: Optional[bool] = None
    top_logprobs: Optional[int] = None
    # set api_base, api_version, api_key
    base_url: Optional[str] = None
    api_version: Optional[str] = None
    api_key: Optional[str] = None
    model_list: Optional[list] = None  # pass in a list of api_base,keys, etc.
    extra_headers: Optional[dict] = None

    # Local model configuration
    local_model: Optional[Any] = None
    model_path: Optional[str] = None
    quantization_config: Optional[BitsAndBytesConfig] = None
    gradient_strategy: Optional[Literal["finegrained", "sum_squares", "product_probs"]] = None
    connection_strategy: Optional[Literal["mean_product_input", "max_product_input", "mean_l1_norm", "max_l1_norm"]] = None
    chat_template_path: Optional[str] = None
    max_length: Optional[int] = None
    max_new_tokens: Optional[int] = None
    do_sample: Optional[bool] = None
    top_k: Optional[int] = None
    # top_p: Optional[float] = None
    # temperature: Optional[float] = None
    num_return_sequences: Optional[int] = None

    def get_client_configuration(self) -> dict:
        return_dict = {
            "model": self.client_model,
            "timeout": self.timeout,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "n": self.n,
            "stream": self.stream,
            "stream_options": self.stream_options,
            "max_tokens": self.max_tokens,
            "max_completion_tokens": self.max_completion_tokens,
            "modalities": self.modalities,
            "prediction": self.prediction,
            "audio": self.audio,
            "presence_penalty": self.presence_penalty,
            "frequency_penalty": self.frequency_penalty,
            "logit_bias": self.logit_bias,
            "user": self.user,
            "response_format": self.response_format,
            "seed": self.seed,
            "parallel_tool_calls": self.parallel_tool_calls,
            "logprobs": self.logprobs,
            "top_logprobs": self.top_logprobs,
            "base_url": self.base_url,
            "api_version": self.api_version,
            "api_key": self.api_key,
            "model_list": self.model_list,
            "extra_headers": self.extra_headers,
        }
        return_dict = {k: v for k, v in return_dict.items() if v is not None}
        return return_dict
    
    def get_local_configuration(self, include_model: bool = True) -> dict:
        return_dict = {
            "max_length": self.max_length,
            "max_new_tokens": self.max_new_tokens,
            "do_sample": self.do_sample,
            "top_k": self.top_k,
            "top_p": self.top_p,
            "temperature": self.temperature,
            "num_return_sequences": self.num_return_sequences,
        }
        return_dict = {k: v for k, v in return_dict.items() if v is not None}
        if include_model:
            return_dict["local_model"] = self.local_model
            return_dict["model_path"] = self.model_path
            return_dict["quantization_config"] = self.quantization_config
        return return_dict