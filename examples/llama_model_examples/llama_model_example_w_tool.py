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

config = ModelConfig(
    local_model="LlamaModel",
    model_path="/shared/storage-01/users/xy61/models/Llama3.1-8B-Instruct",
    max_new_tokens=1024,
    # temperature=0.0,
    do_sample=False,
    gradient_strategy="sum_squares",
    connection_strategy="mean_l1_norm",
    chat_template_path="src/agentchord/model/chat_templates/tool_chat_template_llama3.1_json.jinja"
)
model_factory = ModelFactory(config)
model = model_factory.create_model()
input_content = GBC(
    value = f"{INPUT_HEADER}I found a love for me. Oh, darlin', just dive right in and follow my lead.{INPUT_SEPARATOR}Weather Condition: Isolated thunderstorms throughout the day.{INPUT_SEPARATOR}Well, I found a girl, beautiful and sweet. Oh, I never knew you were the someone waitin' for me.{INPUT_FOOTER}",
    connections= [
        "I found a love for me. Oh, darlin', just dive right in and follow my lead.",
        "Weather Condition: Isolated thunderstorms throughout the day.",
        "Well, I found a girl, beautiful and sweet. Oh, I never knew you were the someone waitin' for me."
    ]
)
txt1 = GBC(
    value = "I found a love for me. Oh, darlin', just dive right in and follow my lead.",
    connections=[
        "I found a love for me. Oh, darlin', just dive right in and follow my lead."
    ],
    weights=[1.0]
)
txt2 = GBC(
    value = "Weather Condition: Isolated thunderstorms throughout the day.",
    connections=[
        "Weather Condition: Isolated thunderstorms throughout the day."
    ],
    weights=[1.0]
)
txt3 = GBC(
    value = "Well, I found a girl, beautiful and sweet. Oh, I never knew you were the someone waitin' for me.",
    connections=[
        "Well, I found a girl, beautiful and sweet. Oh, I never knew you were the someone waitin' for me.",
    ],
    weights=[1.0]
)
text4 = GBC(
    value = "Isolated thunderstorms throughout the day",
    connections=[txt1, txt2, txt3],
    weights=[1.0, 1.0, 1.0],
)
tool_result = GBC(
    value = f"{TOOL_HEADER}Isolated thunderstorms throughout the day{TOOL_FOOTER}",
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
    {"role": "system", "content": "You are a helpful agent that can output information about today's weather based on the input. You have to report the answer using the Response tool."},
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
    }
]

response = model.completion(messages, tools=tools)
print(response)
output = response.choices[0].message.gbc_tool_calls
print(f"Is the output a GBC object? {isinstance(output, GBCBase)}")
print(f"Class of the output: {output.__class__.__name__}")
visualize_gbc_tree(response.choices[0].message.gbc_tool_calls, save_path="examples/llama_model_examples/llama_model_example_w_tool.png")