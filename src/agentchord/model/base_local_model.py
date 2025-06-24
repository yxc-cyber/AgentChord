import threading
from typing import List, Optional, Tuple, Union

import torch
from litellm import (
    ChatCompletionMessageToolCall,
    Choices,
    CustomStreamWrapper,
    ModelResponse,
)
from transformers import BitsAndBytesConfig
from transformers.tokenization_utils_base import BatchEncoding

from ..gbc_object import GBC, GBCBase
from ..utils import (
    INPUT_FOOTER,
    INPUT_HEADER,
    INPUT_SEPARATOR,
    TOOL_FOOTER,
    TOOL_HEADER,
)
from .base_model import BaseModel
from .model_config import ModelConfig
from .utils import (
    CONNECTION_STRATEGIES,
    FINEGRAINED,
    GRADIENT_STRATEGIES,
    MAX_L1_NORM,
    MAX_PRODUCT_INPUT,
    MEAN_L1_NORM,
    MEAN_PRODUCT_INPUT,
    SUM_SQUARES,
    Message,
    SingletonMeta,
    sanitize_output_string,
)


class BaseLocalModel(BaseModel, metaclass=SingletonMeta):
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.model = None
        self.tokenizer = None
        self.config = config
        self.quantization_config = self.config.quantization_config
        if not self.config.model_path:
            raise ValueError("Local model path is not configured. Please provide a valid local path in the configuration.")
        if not self.config.gradient_strategy:
            raise ValueError("Gradient strategy must be specified in the configuration.")
        if self.config.gradient_strategy not in GRADIENT_STRATEGIES:
            raise ValueError(f"Invalid gradient strategy: {self.config.gradient_strategy}. Must be one of {GRADIENT_STRATEGIES}.")
        self.gradient_strategy = GRADIENT_STRATEGIES[self.config.gradient_strategy]
        if not self.config.connection_strategy:
            raise ValueError("Connection strategy is not configured. Please provide a valid connection strategy in the configuration.")
        if self.config.connection_strategy not in CONNECTION_STRATEGIES:
            raise ValueError(f"Invalid connection strategy: {self.config.connection_strategy}. Must be one of {CONNECTION_STRATEGIES}.")
        self.connection_strategy = CONNECTION_STRATEGIES[self.config.connection_strategy]
        

    def _check_tools(
            self,
            tools: Optional[list] = None,
            tool_choice:  Optional[str] = None
        ):
        if tools:
            for tool in tools:
                if not isinstance(tool, dict):
                    raise ValueError(f"Tool {tool} is not a valid tool description. It should be a dictionary.")
                if not tool.get("type") == "function":
                    raise ValueError(f"Tool {tool} is not a valid function tool description. It should have 'type' set to 'function'.")
                if not tool.get("function"):
                    raise ValueError(f"Tool {tool} is not a valid function tool description. It should have 'function' key with function details.")
                if not tool.get("function").get("name"):
                    raise ValueError(f"Tool {tool} is not a valid function tool description. It should have 'function' key with 'name'.")
            if tool_choice and tool_choice != "auto":
                tools = [tool for tool in tools if tool.get("function").get("name") == tool_choice]
                if not tools:
                    raise ValueError(f"No tool found with name {tool_choice}. Available tools: {[tool.get('function').get('name') for tool in tools]}")
            
    def _apply_chat_template(
        self,
        messages: List[Union[dict, Message]],
        tools: Optional[list] = None,
    ) -> str:
        """
        Apply the chat template to the messages and return a list of processed conversations.
        Args:
            messages (List[Union[dict, Message]]): List of messages to process.
        Returns:
            List[str]: List of processed conversations as strings.
        """
        conversations_processed = self.tokenizer.apply_chat_template(
            conversation = messages,
            tools = tools,
            add_generation_prompt = True,
            tokenize = False,
        )
        if not conversations_processed:
            raise ValueError("No conversations processed. Please check the input messages.")
        return conversations_processed

    def _get_input_blocks_and_encodings(self, conversations_processed: List[str]) -> Tuple[List[List[Tuple[int, int]]], BatchEncoding]:
        """
        Process the conversations and return the input blocks and their encodings.
        Args:
            conversations_processed (List[str]): List of processed conversations.
        Returns:
            Tuple[List[List[Tuple[int, int]]], torch.Tensor]: A tuple containing:
                - input_blocks: A list of lists, where each inner list contains tuples of (block_start[included], block_end[included]).
                - encoding: The tokenized encoding of the processed conversations.
        """
        # Find indices of input header, input footer, input separators, tool header, and tool footer
        input_header_indices = list()
        input_footer_indices = list()
        input_separator_indices = list()
        tool_header_indices = list()
        tool_footer_indices = list()
        for messages_processed in conversations_processed:
            input_header_indices.append(messages_processed.find(INPUT_HEADER))
            input_footer_indices.append(messages_processed.find(INPUT_FOOTER))
            input_separator_indices_temp = []
            sep_pos = 0
            while True:
                sep_pos = messages_processed.find(INPUT_SEPARATOR, sep_pos)
                if sep_pos == -1:
                    break
                if sep_pos > input_header_indices[-1] and sep_pos < input_footer_indices[-1]:
                    input_separator_indices_temp.append(sep_pos)
                sep_pos += len(INPUT_SEPARATOR)
            input_separator_indices.append(input_separator_indices_temp)
            tool_header_indices_temp = list()
            tool_footer_indices_temp = list()
            sep_pos = 0
            while True:
                tool_header_idx = messages_processed.find(TOOL_HEADER, sep_pos)
                if tool_header_idx == -1:
                    break
                tool_header_indices_temp.append(tool_header_idx)
                tool_footer_idx = messages_processed.find(TOOL_FOOTER, tool_header_idx + len(TOOL_HEADER))
                if tool_footer_idx == -1:
                    raise ValueError(f"Tool footer not found for tool header at index {tool_header_idx}. Please check the input messages.")
                tool_footer_indices_temp.append(tool_footer_idx)
                sep_pos = tool_footer_idx + len(TOOL_FOOTER)
            tool_header_indices.append(tool_header_indices_temp)
            tool_footer_indices.append(tool_footer_indices_temp)

        # Tokenize the processed messages
        encoding = self.tokenizer(
            conversations_processed,
            return_tensors="pt",
            padding=True,
            add_special_tokens=False,
        ).to(self.model.device)

        input_blocks = list()
        # Find the indices of the input header, footer, and separators in the tokenized input_ids
        for conversation, input_header_idx, input_footer_idx, input_separator_idcs, tool_header_idcs, tool_footer_idcs in \
            zip(conversations_processed, input_header_indices, input_footer_indices, input_separator_indices, tool_header_indices, tool_footer_indices):
            input_blocks_temp = list()
            if input_header_idx != -1 and input_footer_idx != -1:
                block_start = len(self.tokenizer(
                    conversation[:input_header_idx+len(INPUT_HEADER)],
                    return_tensors="pt",
                )["input_ids"][0])
                for input_separator_idx in input_separator_idcs:
                    block_end = len(self.tokenizer(
                        conversation[:input_separator_idx],
                        return_tensors="pt",
                    )["input_ids"][0])
                    input_blocks_temp.append((block_start, block_end-1))  # (block_start[included], block_end[included])
                    block_start = len(self.tokenizer(
                        conversation[:input_separator_idx+len(INPUT_SEPARATOR)],
                        return_tensors="pt",
                    )["input_ids"][0])
                block_end = len(self.tokenizer(
                    conversation[:input_footer_idx],
                    return_tensors="pt",
                )["input_ids"][0])
                input_blocks_temp.append((block_start, block_end-1))  # (block_start[included], block_end[included])
            if tool_header_idcs and tool_footer_idcs:
                for tool_header_idx, tool_footer_idx in zip(tool_header_idcs, tool_footer_idcs):
                    if tool_header_idx != -1 and tool_footer_idx != -1:
                        block_start = len(self.tokenizer(
                            conversation[:tool_header_idx+len(TOOL_HEADER)],
                            return_tensors="pt",
                        )["input_ids"][0])
                        block_end = len(self.tokenizer(
                            conversation[:tool_footer_idx],
                            return_tensors="pt",
                        )["input_ids"][0])
                        input_blocks_temp.append((block_start, block_end-1))
            input_blocks.append(input_blocks_temp)

        # input_blocks is a list of lists, where each inner list contains tuples of (block_start[included], block_end[included])
        return input_blocks, encoding
    
    def _get_outputs(self, encodings: BatchEncoding, sequences: torch.Tensor) -> List[str]:
        inputs = self.tokenizer.batch_decode(
            encodings.input_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )
        inputs_outputs = self.tokenizer.batch_decode(
            sequences,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )
        outputs = list()
        for input, input_output in zip(inputs, inputs_outputs):
            assert input_output.startswith(input)
            output = input_output[len(input):].strip()
            outputs.append(output)
        return outputs
    
    def _get_tool_calls(self, output: str) -> Optional[List[ChatCompletionMessageToolCall]]:
        raise NotImplementedError("This method should be implemented in subclasses to handle tool calls extraction.")
    
    def _get_connection_weights(
            self,
            gradients: torch.Tensor,
            input_blocks: List[List[Tuple[int, int]]],
            embedings: Optional[torch.Tensor] = None
        ) -> List[List[float]]:
        if self.gradient_strategy == FINEGRAINED:
            # gradient: (batch_size, output_sequence_length, input_sequence_length, hidden_size)
            # input_blocks: (batch_size, blocl_num, 2)
            raise NotImplementedError("Connection weight for fine-grained gradient strategy is not implemented yet.")
        elif self.gradient_strategy == SUM_SQUARES:
            # gradient: (batch_size, input_sequence_length, hidden_size)
            # input_blocks: (batch_size, blocl_num, 2)
            # embedings: (batch_size, input_sequence_length, hidden_size)
            weights = list()
            for batch_idx, (local_gradient, local_input_blocks) in enumerate(zip(gradients, input_blocks)):
                # local_gradient: (input_sequence_length, hidden_size)
                # local_input_blocks: (block_num, 2)
                block_weights = list()
                for block_start, block_end in local_input_blocks:
                    # Get the gradient for the current block
                    block_gradient = local_gradient[block_start:block_end+1, :]
                    # Calculate the sum of squares for the block
                    if self.connection_strategy == MEAN_PRODUCT_INPUT:
                        # Calculate the product of the input embeddings for the block
                        if embedings is None:
                            raise ValueError("Embedings must be provided when using PRODUCT_INPUT connection strategy.")
                        block_embedding = embedings[batch_idx, block_start:block_end+1, :]
                        block_weight = torch.mean(torch.sum(block_gradient * block_embedding, dim=1)).item()
                    elif self.connection_strategy == MEAN_L1_NORM:
                        # Calculate the L1 norm of the block gradient
                        block_weight = torch.mean(torch.sum(torch.abs(block_gradient), dim=1)).item()
                    elif self.connection_strategy == MAX_PRODUCT_INPUT:
                        # Calculate the product of the input embeddings for the block
                        if embedings is None:
                            raise ValueError("Embedings must be provided when using PRODUCT_INPUT connection strategy.")
                        block_embedding = embedings[batch_idx, block_start:block_end+1, :]
                        block_weight = torch.max(torch.sum(block_gradient * block_embedding, dim=1)).item()
                    elif self.connection_strategy == MAX_L1_NORM:
                        # Calculate the L1 norm of the block gradient
                        block_weight = torch.max(torch.sum(torch.abs(block_gradient), dim=1)).item()
                    else:
                        raise ValueError(f"Invalid connection strategy: {self.connection_strategy}. Must be one of {CONNECTION_STRATEGIES}.")
                    block_weights.append(block_weight)
                weights.append(block_weights)
            return weights
        else:
            raise ValueError(f"Invalid gradient strategy: {self.gradient_strategy}. Must be one of {GRADIENT_STRATEGIES}.")

    def completion(
        self,
        messages: List[Union[dict, Message]],
        tools: Optional[list] = None,
        tool_choice: Optional[str] = None
    ) -> Union[ModelResponse, CustomStreamWrapper]:
        """
        Generate a completion for the given messages using the Llama model.
        Args:
            messages (List[Union[dict, Message]]): The messages to process.
            tools (Optional[list]): List of tools available for the model to use.
            tool_choice (Optional[str]): The tool choice strategy.
        Returns:
            Union[ModelResponse, CustomStreamWrapper]: The model's response.
        """
        # In this function, we assume that the batch size is 1, i.e., we only process one conversation at a time.

        # Check the tools and tool_choice parameters
        self._check_tools(tools, tool_choice)

        # Apply the chat template to the messages
        conversation_processed = self._apply_chat_template(messages, tools)

        # Get the input blocks and their encodings
        input_blocks, encoding = self._get_input_blocks_and_encodings([conversation_processed])
        # Input_blocks is a list of lists, where each inner list contains tuples of (block_start[included], block_end[included])

        # Generate the model's response
        outputs = self.model.generate(**encoding, **self.config.get_local_configuration(include_model=False))
        sequences = outputs.sequences  # (batch_size, total_sequence_length)
        gradients = outputs.gradients
        # If FINEGRAINED: (batch_size, output_sequence_length, input_sequence_length, hidden_size)
        # If SUM_SQUARES: (batch_size, input_sequence_length, hidden_size)
        embedings = outputs.embeds  # (batch_size, input_sequence_length, hidden_size)

        # Get connection weights based on the gradients and input blocks
        connection_weights = self._get_connection_weights(
            gradients=gradients,
            input_blocks=input_blocks,
            embedings=embedings,
        )

        # Get the outputs from the model
        outputs = self._get_outputs(encoding, sequences)
        outputs = [sanitize_output_string(output) for output in outputs]

        # Parse the outputs and check for tool calls
        reponses = list()
        for output, local_connection_weights in zip(outputs, connection_weights):
            if not isinstance(output, str):
                raise ValueError(f"Output {output} is not a valid string. Please check the model's output.")
            # Get the connections and weights for the current batch
            if isinstance(messages[-1].get("content", ""), GBCBase):
                connections = messages[-1].get("content", "").get_connections()
                weights = local_connection_weights
            else:
                connections = messages[-1].get("content", "")
                weights = 1.0
            # Get the tool calls from the outputs
            tool_calls = self._get_tool_calls(output)
            if tool_calls:
                if not isinstance(tool_calls, list):
                    raise ValueError(f"Tool calls {tool_calls} is not a valid list. Please check the implementation of _get_tool_calls.")
                tool_calls = GBC(
                    tool_calls,
                    connections=connections,
                    weights=weights,
                )
                reponse = Choices(
                    message=Message(
                        role="assistant",
                        content=None,  # The content will be set to None since we are using tool calls
                        tool_calls=tool_calls,
                    )
                )
            else:
                output = GBC(
                    output,
                    connections=connections,
                    weights=weights,
                )
                reponse = Choices(
                    message=Message(
                        role="assistant",
                        content=output,
                        tool_calls=None,  # No tool calls in this case
                    )
                )
            reponses.append(reponse)
        assert len(reponses) == 1, "Batch size must be 1 for local model completion."
        return ModelResponse(choices=reponses)