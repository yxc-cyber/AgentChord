from transformers import AutoTokenizer

from agentchord.model import LlamaForCausalLM_GBC

model = LlamaForCausalLM_GBC.from_pretrained("/scratch/xy61/models/Llama3.1-8B-Instruct", device_map="auto", torch_dtype="auto")
tokenizer = AutoTokenizer.from_pretrained("/scratch/xy61/models/Llama3.1-8B-Instruct")
tokenizer.pad_token = tokenizer.eos_token
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