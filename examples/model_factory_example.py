from agentchord import ModelConfig
from agentchord.model.model_factory import ModelFactory

config = ModelConfig(
    client_model="openai/gpt-4o-mini"
)
model_factory = ModelFactory(config)
model = model_factory.create_model()
messages = [
    {"role": "system", "content": "You are a helpful agent that can greet the user."},
    {"role": "user", "content": "Hello!"}
]
response = model.completion(messages)
print(response)