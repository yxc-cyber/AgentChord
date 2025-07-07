from agentchord import GBC, BaseAgent, BaseEnvironment, BaseMetaData, ModelConfig
from agentchord.agent_system import Input
from agentchord.gbc_object import visualize_gbc_tree

prompt = "You are a helpful agent that can repeat what the user says."
model_config = ModelConfig(client_model="openai/gpt-4o-mini")
initial_input = GBC("Nice to meet you!", subject=Input())
inital_setup = BaseMetaData(input=initial_input)
base_environment = BaseEnvironment(initial_metadata=inital_setup)
agents = [BaseAgent(
    system_name=f"agent_{i}",
    prompt=prompt,
    model_config=model_config,
    environment=base_environment,
    log_name="gbc_backward_example.log"
) for i in range(1, 8)]
connection_map = {
    1: ([0], [1.0]),
    2: ([0], [1.0]),
    3: ([1, 2], [0.1, 0.9]),
    4: ([1, 2], [0.9, 0.1]),
    5: ([3, 4], [0.9, 0.1]),
    6: ([3, 4], [0.1, 0.9]),
    7: ([5, 6], [0.1, 0.9]),   
}
outputs = [
    f"Output from agent_{i}" for i in range(1, 8)  
]
connected_outputs = []
for i, agent in enumerate(agents):
    connections, weights = connection_map.get(i + 1, ([], []))
    processed_connections = [connected_outputs[j-1] if j != 0 else initial_input for j in connections]
    connected_outputs.append(GBC(outputs[i], subject=agent, connections=processed_connections, weights=weights))
final_output = connected_outputs[-1]
visualize_gbc_tree(final_output, "examples/gbc_object_examples/gbc_backward_example.png")
final_output.backward(bandwidth=1)
print("Trajectory:")
print(f"Input: {Input().get_optimization_info()}")
for agent in agents:
    print(f"{agent.system_name}: {agent.get_optimization_info()}")