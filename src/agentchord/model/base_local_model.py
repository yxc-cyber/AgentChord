from typing import List, Optional, Tuple, Union

import torch
from litellm import CustomStreamWrapper, Message, ModelResponse, completion
from transformers.tokenization_utils_base import BatchEncoding

from ..gbc_object import GBC, GBCBase
from ..utils import INPUT_FOOTER, INPUT_HEADER, INPUT_SEPARATOR
from .base_model import BaseModel
from .model_config import ModelConfig
from .utils import FINEGRAINED, SUM_SQUARES


class BaseLocalModel(BaseModel):
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.model = None
        self.tokenizer = None

    def _check_tools(
            self,
            tools: Optional[list] = None,
            tool_choice:  Optional[str] = None
        ):
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
        # Find indices of input header, input footer, and input separators
        input_header_indices = list()
        input_footer_indices = list()
        input_separator_indices = list()
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

        # Tokenize the processed messages
        encoding = self.tokenizer(
            conversations_processed,
            return_tensors="pt",
            padding=True,
            add_special_tokens=False,
        ).to(self.model.device)

        input_blocks = list()
        # Find the indices of the input header, footer, and separators in the tokenized input_ids
        for conversation, input_header_idx, input_footer_idx, input_separator_idcs in zip(conversations_processed, input_header_indices, input_footer_indices, input_separator_indices):
            input_blocks_temp = list()
            block_start = len(self.tokenizer(
                conversation[:input_header_idx+len(INPUT_HEADER)],
                return_tensors=False,
            )["input_ids"][0])
            for input_separator_idx in input_separator_idcs:
                block_end = len(self.tokenizer(
                    conversation[:input_separator_idx],
                    return_tensors=False,
                )["input_ids"][0])
                input_blocks_temp.append((block_start + 1, block_end))  # (block_start[included], block_end[included])
                block_start = len(self.tokenizer(
                        conversation[:input_separator_idx+len(INPUT_SEPARATOR)],
                        return_tensors=False,
                    )["input_ids"][0])
            block_end = len(self.tokenizer(
                conversation[:input_footer_idx+len(INPUT_FOOTER)],
                return_tensors=False,
            )["input_ids"][0])
            input_blocks_temp.append((block_start + 1, block_end))
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
    
    def _get_tool_calls(self, output: str) -> List[dict]:
        raise NotImplementedError("This method should be implemented in subclasses to handle tool calls extraction.")
    
    def _get_connection_weights(self, gradients: torch.Tensor) -> List[float]:
        # Todo: Implement a method to extract connection weights from the gradients.

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

        # Get the outputs from the model
        outputs = self._get_outputs(encoding, sequences)

        # Parse the outputs and check for tool calls
        reponses = list()
        for output in outputs:
            if not isinstance(output, str):
                raise ValueError(f"Output {output} is not a valid string. Please check the model's output.")
            # Get the tool calls from the outputs
            tool_calls = self._get_tool_calls(output)
            if tool_calls:
                if not isinstance(tool_calls, list):
                    raise ValueError(f"Tool calls {tool_calls} is not a valid list. Please check the implementation of _get_tool_calls.")
                if isinstance(messages, GBCBase):
                    connections = messages.get_connections()
                    weights = messages
                else:
                    connections = messages
                    weights = [1.0]
                tool_calls = GBC(
                    tool_calls,
                    connections=connections,
                    weights=weights,
                )
                reponse = ModelResponse(
                    output=None,
                    tool_calls=tool_calls,
                )
            else:
                reponse = ModelResponse(
                    output=output,
                    tool_calls=None,
                )
            reponses.append(reponse)
            