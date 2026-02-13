import torch
from transformers import BitsAndBytesConfig

from agentchord import GBC
from agentchord.gbc_object import GBCBase, visualize_gbc_tree
from agentchord.model import ModelConfig, ModelFactory
from agentchord.utils import INPUT_FOOTER, INPUT_HEADER, INPUT_SEPARATOR

bnb_config = BitsAndBytesConfig(
    load_in_4bit = True,
    bnb_4bit_use_double_quant = True,
    bnb_4bit_quant_type = "nf4",
    bnb_4bit_compute_dtype = torch.bfloat16
)

config = ModelConfig(
    local_model="LlamaModel",
    model_path="/shared/storage-01/users/xy61/models/Llama-3.3-70B-Instruct",
    quantization_config=bnb_config,
    max_new_tokens=1024,
    # temperature=0.0,
    do_sample=False,
    gradient_strategy="product_probs",
    connection_strategy="max_l1_norm",
    chat_template_path="src/agentchord/model/chat_templates/tool_chat_template_llama3.3_json.jinja"
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
messages = [
    {"role": "system", "content": "You are a helpful agent that can extract information about today's weather based on the input."},
    {"role": "user", "content": input_content}
]

response = model.completion(messages)
print(response)
output = response.choices[0].message.gbc_content
print(f"Is the output a GBC object? {isinstance(output, GBCBase)}")
print(f"Class of the output: {output.__class__.__name__}")
visualize_gbc_tree(response.choices[0].message.gbc_content, save_path="examples/llama_model_examples/llama_model_example.png")