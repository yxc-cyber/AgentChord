from agentchord import BaseAgent, BaseEnvironment, BaseMetaData, ModelConfig

prompt = "You are a helpful agent that can repeat what the user says."
model_config = ModelConfig(client_config={
    "model": "openai/gpt-4o"
})
inital_setup = BaseMetaData(input="Nice to meet you!")
base_environment = BaseEnvironment(initial_setup=inital_setup)
base_agent = BaseAgent(
    system_name="base_agent",
    prompt=prompt,
    model_config=model_config,
    environment=base_environment,
    log_name="simple_example1.log"
)

print("Round 1")
step_result = base_agent.run(debug=True)
print(step_result)

print("Round 2")
step_result = base_agent.run(log_name="simple_example1.log.0")
print(step_result)