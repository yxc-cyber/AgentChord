from agentchord.model import ModelConfig, ModelFactory

config = ModelConfig(
    local_model="LlamaModel",
    temperature=0.0
)
model_factory = ModelFactory(config)
model = model_factory.create_model()
messages = [
    {"role": "system", "content": "You are a helpful agent that can greet the user."},
    {"role": "user", "content": "Hello!"}
]
response = model.completion(messages)
print(response)