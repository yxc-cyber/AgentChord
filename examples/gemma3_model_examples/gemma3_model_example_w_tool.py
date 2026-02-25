import torch
from transformers import BitsAndBytesConfig

from agentchord import GBC
from agentchord.gbc_object import GBCBase, visualize_gbc_tree
from agentchord.model import ModelConfig, ModelFactory
from agentchord.utils import (
    INPUT_FOOTER,
    INPUT_HEADER,
    INPUT_SEPARATOR,
    TOOL_FOOTER,
    TOOL_HEADER,
)

bnb_config = BitsAndBytesConfig(
    load_in_4bit = True,
    bnb_4bit_use_double_quant = True,
    bnb_4bit_quant_type = "nf4",
    bnb_4bit_compute_dtype = torch.bfloat16
)

config = ModelConfig(
    local_model="Gemma3Model",
    model_path="/work/hdd/bghs/xyang7/models/gemma-3-27b-it",
    quantization_config=bnb_config,
    max_new_tokens=1024,
    # temperature=0.0,
    do_sample=False,
    gradient_strategy="product_probs",
    connection_strategy="max_l1_norm",
    chat_template_path="src/agentchord/model/chat_templates/tool_chat_template_gemma3_json.jinja",
    enable_thinking=False,
)
model_factory = ModelFactory(config)
model = model_factory.create_model()
input_content = GBC(
    value = f"{INPUT_HEADER}Input 1: A football match will be held tomorrow.{INPUT_SEPARATOR}Input 2: Weather Condition: Isolated thunderstorms throughout the day.{INPUT_SEPARATOR}Input 3: A cat sat on a mat.{INPUT_FOOTER}",
    connections= [
        "Input 1: A football match will be held tomorrow.",
        "Input 2: Weather Condition: Isolated thunderstorms throughout the day.",
        "Input 3: A cat sat on a mat."
    ]
)
txt1 = GBC(
    value = "Input 1: A football match will be held tomorrow.",
    connections=[
        "Input 1: A football match will be held tomorrow."
    ],
    weights=[1.0]
)
txt2 = GBC(
    value = "Input 2: Weather Condition: Isolated thunderstorms throughout the day.",
    connections=[
        "Input 2: Weather Condition: Isolated thunderstorms throughout the day."
    ],
    weights=[1.0]
)
txt3 = GBC(
    value = "Input 3: A cat sat on a mat.",
    connections=[
        "Input 3: A cat sat on a mat.",
    ],
    weights=[1.0]
)
text4 = GBC(
    value = "Isolated thunderstorms throughout the day",
    connections=[txt1, txt2, txt3],
    weights=[1.0, 1.0, 1.0],
)
tool_result = GBC(
    value = f"{TOOL_HEADER}Isolated thunderstorms throughout the day. Please use the response tool now.{TOOL_FOOTER}",
    connections=[
        txt1,
        txt2,
        txt3,
        text4
    ],
    weights=[
        1.0,  # Connection to txt1
        1.0,  # Connection to txt2
        1.0,  # Connection to txt3
        1.0   # Connection to text4
    ]
)
messages = [
    {"role": "system", "content": "You are a helpful agent that can extract information about today's weather based on the input. Important: You have to use the response tool to respond. Put the response in the tool result."},
    {"role": "user", "content": input_content},
    {"role": "assistant", "content": None, 'tool_calls': [{'function': {'arguments': '{"city":"champaign"}', 'name': 'query_weather'}, 'id': 'call_xBGr1c7JqAnbRMbkzM2F68ob', 'type': 'function'}]},
    {'role': 'tool', 'tool_call_id': 'call_xBGr1c7JqAnbRMbkzM2F68ob', 'name': 'query_weather', 'content': tool_result}
]
tools = [
    {
        "type": "function",
        "function": {
            "name": "response",
            "description": "Respond based on the input.",
            "parameters": {
                "type": "object",
                "properties": {
                    "response": {
                        "type": "string",
                        "description": "The response to the user's input."
                    }
                },
                "required": ["response"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "query_weather",
            "description": "Query the weather based on the input.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "The city for which to query the weather."
                    }
                },
                "required": ["city"]
            }
        }
    }
]

response = model.completion(messages, tools=tools)
print(response)
try:
    output = response.choices[0].message.gbc_tool_calls
    print(f"Is the output a GBC object? {isinstance(output, GBCBase)}")
    print(f"Class of the output: {output.__class__.__name__}")
    visualize_gbc_tree(response.choices[0].message.gbc_tool_calls, save_path="examples/gemma3_model_examples/gemma3_model_example_w_tool.png")
except:
    output = response.choices[0].message.gbc_content
    print(f"Is the output a GBC object? {isinstance(output, GBCBase)}")
    print(f"Class of the output: {output.__class__.__name__}")
    visualize_gbc_tree(response.choices[0].message.gbc_content, save_path="examples/gemma3_model_examples/gemma3_model_example_w_tool.png")