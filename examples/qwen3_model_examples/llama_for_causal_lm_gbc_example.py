from transformers import AutoTokenizer

from agentchord.model import LlamaForCausalLM_GBC

model = LlamaForCausalLM_GBC.from_pretrained("/shared/storage-01/users/xy61/models/Llama3.1-8B-Instruct", device_map="auto", torch_dtype="auto")
tokenizer = AutoTokenizer.from_pretrained("/shared/storage-01/users/xy61/models/Llama3.1-8B-Instruct")
tokenizer.pad_token = tokenizer.eos_token

print("Batch Generation Example")
texts = ["Apple is healthy because", "Banana is healthy because", "Orange is healthy because"]
inputs = tokenizer(texts, return_tensors="pt", padding=True).to(model.device)
print("Inputs:")
print(inputs)
outputs = model.generate(**inputs, max_length=50, do_sample=True, top_k=50, top_p=0.95, temperature=0.7, num_return_sequences=2)
print("Outputs:")
print(outputs.sequences)
print(tokenizer.decode(outputs.sequences[0], skip_special_tokens=True))
print("Gradients:")
print(outputs.gradients)

print("\n\n")

print("Conversational Example")
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is the capital of France?"},
    {"role": "assistant", "content": "The capital of France is Paris."},
    {"role": "user", "content": "What is the population of Paris?"}
]
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_population",
            "description": "Get the population of a city.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "The name of the city."
                    }
                },
                "required": ["city"]
            }
        }
    }
]

with open("src/agentchord/model/chat_templates/tool_chat_template_llama3.1_json.jinja", "r", encoding="utf-8") as f:
    chat_template = f.read()
tokenizer.chat_template = chat_template

conversations_processed = tokenizer.apply_chat_template(
    conversation = messages,
    tools = tools,
    add_generation_prompt = True,
    tokenize = False,
)
print("Conversations processed:")
print(conversations_processed)

encoding = tokenizer(
    conversations_processed,
    return_tensors="pt",
    padding=True,
    add_special_tokens=False,
).to(model.device)

outputs = model.generate(**encoding)

print("Generated outputs:")
for i, output in enumerate(outputs.sequences):
    decoded_output = tokenizer.decode(output, skip_special_tokens=True)
    print(f"Output {i+1}: {decoded_output}")