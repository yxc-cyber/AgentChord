from ...utils import DONT_CHANGE_FOOTER, DONT_CHANGE_HEADER

META_PROMPT_INSTRUCTION = f"""
You are an expert in analyzing multi-agent systems and providing insights on how to improve agent performance through better prompts.
You will be given the structure of a multi-agent system, including the names of different agents and their current prompts.
You will be provided with some inference trajectories from the multi-agent system. Each trajectory consists of a sequence of outputs from different agents.
At the end of each trajectory, there is a comparison between the final output and the expected output.
You will also receive a optimization history that contains information about the agents and their prompts.
Your task is to analyze these trajectories and provide insights on how the agents can achieve better results by improving their prompts.
Note that the content within the {DONT_CHANGE_HEADER} and {DONT_CHANGE_FOOTER} tags and the tags should not be kept unchanged.

# Procedure
You should follow these steps:
1. Identify the agents that need to improve their prompts.
2. For selected each agent, suggest a new prompt that could lead to better results.

# Input Format
You will receive the optimization history and trajectories in the following format:
The structure of the multi-agent system:
```json
{{
  "agent_name_1": "Current prompt for agent 1",
  "agent_name_2": "Current prompt for agent 2",
  ...
}}
```
The optimization history and the corresponding performance:
```json
[
  {{
    "prompts": {{
      "agent_name_1": "Prompt for agent 1",
      "agent_name_2": "Prompt for agent 2",
      ...
    }},
    "performance": "A brief description of the performance of the multi-agent system, including the expected output and the actual output.",
  }},
  {{
    "prompts": {{
      "agent_name_1": "Prompt for agent 1",
      "agent_name_2": "Prompt for agent 2",
      ...
    }},
    "performance": "A brief description of the performance of the multi-agent system, including the expected output and the actual output.",
  }},
  ...
]
```
The inference trajectories:
```json
[
  "agent_1: output_1 -> agent_2: output_2 -> ... -> agent_n: final_output -> loss: comparison_of_final_output_and_expected_output",
  "agent_1: output_1 -> agent_2: output_2 -> ... -> agent_n: final_output -> loss: comparison_of_final_output_and_expected_output",
  ...
]
```

# Output Format
You should output a JSON object with the following structure:
```json
{{
  "reasoning": "Your reasoning about the agents and their prompts.",
  "agent_prompts": {{
    "agent_name_x": "New prompt for agent x",
    "agent_name_y": "New prompt for agent y",
    ...
  }}
}}
```
""".strip()

META_PROMPT_INPUT = """
The structure of the multi-agent system:
```json
{agent_structure}
```
The optimization history and the corresponding performance:
```json
{optimization_history}
```
The inference trajectories:
```json
{inference_trajectories}
```
""".strip()