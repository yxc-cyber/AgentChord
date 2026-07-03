<div align="center"  id="banner">
<img src="assets/Banner.svg" alt="logo"  margin="10px"></img>
</div>

# AgentChord
  
AgentChord is a flexible framework for building, running, and optimizing multi-agent systems (MAS). It provides core abstractions and utilities to help you design, implement, and experiment with MAS efficiently.

## Features

- Gradient-Based Connections (GBC) for automataed multi-agent system optimization.
- Handy visualization of Gradient-Based Connections (GBC).
- Connected to [WandB](https://wandb.ai/site/sdk) to easily monitor the optimization process.
- Based on [LiteLLM](https://github.com/BerriAI/litellm) to support APIs from various providers.

## Getting Started

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/AgentChord.git
    cd AgentChord
    ```

2. Make sure you have Astral UV installed in your environment. Please refer to [the official installation instruaction](https://docs.astral.sh/uv/getting-started/installation/) if you have not yet.

3. Synchronize the virtual environment using UV:
    ```bash
    uv sync
    ```

4. Configure the LiteLLM environment variables in `.env`.

5. Explore the `examples/` directory for ready-to-run demonstrations.

## Instruction

### 1. Define Your Environment

- Create an environment instance (e.g., `Multiwoz24Environment`) that provides the context and tool descriptions for your agents.

### 2. Configure Models

- Use the `ModelConfig` class to specify model parameters, such as:
    - `local_model` or `client_model` (e.g., `"LlamaModel"` or `"openai/gpt-4o-mini"`)
    - `model_path` for local models
    - Quantization and optimization configs if needed (e.g., `BitsAndBytesConfig`)
    - Generation parameters: `max_new_tokens`, `temperature`, `do_sample`, etc.
    - Connection and gradient strategies if a local model is used: `connection_strategy`, `gradient_strategy`
    - Chat template path if a local model is used

Example:
```python
config = ModelConfig(
        local_model="LlamaModel",
        model_path="/path/to/model",
        quantization_config=BitsAndBytesConfig(
            load_in_4bit = True,
            bnb_4bit_use_double_quant = True,
            bnb_4bit_quant_type = "nf4",
            bnb_4bit_compute_dtype = torch.bfloat16
        ),
        max_new_tokens=128,
        do_sample=False,
        gradient_strategy="sum_squares",
        connection_strategy="mean_l1_norm",
        chat_template_path="/path/to/template"
)
```

### 3. Build Agent Systems

- Compose your system using `BaseAgentSystem` objects (e.g. ``BaseAgentSystem``, `GBCAgent`, `ParallelBlock`):
    - Divide one task into subtasks and design corresponding subsystems for the top-level system.
    - Add subsystems to the top-level system using `add_subsystem()`.
    - Define the initialization subroutine of the top-level system by overwriting `on_initialization()`.
    - Define subroutines before the execution of certain subsystems using `add_on_completion_action()`.
    - Define subroutines after the execution of certain subsystems using `add_on_start_action()`.
    - Define the finalizatin subroutine of the top-level system by overwriting `on_finalization()`.
    - Note that when defning the subroutines, you should take care of the Gradient-Based Connections (GBC) if you are using `GBCAgent`.

Example:
```python
class Multiwoz24System(BaseAgentSystem):
    def __init__(self, system_name: str, environment: Multiwoz24Environment, maximum_loops: int = 5, log_name: str = ""):
        super().__init__(
            system_name=system_name,
            environment=environment, 
            maximum_loops=maximum_loops,
            log_name=log_name
        )
        StateAgent = BaseAgent(
            system_name="state_agent",
            environment=environment,
            prompt=prompt_read_state,
            model_config=ModelConfig(client_model="openai/gpt-4o-mini", temperature=0.0),
            maximum_loops=maximum_loops,
            log_name=log_name
        )
        ResponseAgent = BaseAgent(
            system_name="response_agent",
            environment=environment,
            prompt=prompt_generate_response,
            model_config=ModelConfig(client_model="openai/gpt-4o-mini", temperature=0.0),
            maximum_loops=maximum_loops,
            log_name=log_name
        )

        self.add_subsystem(StateAgent)
        self.add_subsystem(ResponseAgent)
        self.add_on_start_action("response_agent", "read_state", self._read_state)

    def on_initialization(self) -> MultiWOZ24MetaData:
        metadata = self.environment.get_initial_metadata()
        grounding_utterance = metadata.grounding_utterance
        metadata.input = f"Dialogue History:\n{grounding_utterance}"
        return metadata

    def _read_state(self, metadata: MultiWOZ24MetaData) -> MultiWOZ24MetaData:
        """
        Read the state from the the output of the state agent.
        """
        grounding_utterance = metadata.grounding_utterance
        dialogue_state_raw = metadata.output.strip() or metadata.note.strip()
        try:
            dialogue_state = json.loads(dialogue_state_raw)
        except json.JSONDecodeError:
            _json_markdown_re = re.compile(r"```(json)?(.*)```", re.DOTALL)
            match = _json_markdown_re.search(dialogue_state_raw)
            if match:
                dialogue_state = json.loads(match.group(2))
            else:
                dialogue_state = dict()
        metadata.dialogue_state = dialogue_state
        metadata.input = f"Dialogue History:\n{grounding_utterance}\nDialogue State:\n{json.dumps(dialogue_state)}"
        return metadata
    
    def on_finalization(self, metadata: MultiWOZ24MetaData) -> MultiWOZ24MetaData:
        """
        Finalize the metadata after the response agent has generated the response.
        """
        metadata.system_response = metadata.output or metadata.note
        return metadata
```

### 4. Initialize Optimizer

- For optimization, use `OPROOptimizer` with your agents and a model config for the optimizer. You can monitor and log results with WandB.
    - Note that optimization only works for `GBCAgent` systems.

Example:
```python
optimizer = OPROOptimizer(
        agents=multiwoz_24_system.get_agents(),
        model_config=ModelConfig(client_model="openai/gpt-4o-mini"),
        log_name="example.log",
        wandb_config=WandBConfig(
                project="AgentChord",
                config={...}
        )
)
```

### 5. Run

- Instantiate your top-level system and run it

Example:
```python
multiwoz_24_environment = Multiwoz24Environment(mode, dialogue_idx, turn_idx)
multiwoz_24_system = Multiwoz24System("multiwoz_24_system", multiwoz_24_environment, log_name="example.log")
result = multiwoz_24_system.run()
```

### 6. Evaluate and Optimize

- Use provided evaluation function `evaluate()` and loss objects (e.g., `MultiWOZ24Loss`) to compute and backpropagate losses.
    - Note that optimization only works for `GBCAgent` systems.
    - Optionally, visualize the GBC tree using `visualize_gbc_tree()`

Example:
```python
evaluation_result = multiwoz_24_environment.evaluate(result)
loss_fn = MultiWOZ24Loss()
loss = loss_fn.compute_loss(prediction=result, evaluation_result=evaluation_result, type="joint_goal_accuracy")
loss.backward(bandwidth=1)
optimizer.step(
    performance="...",
    performance_dict={...}
)
visualize_gbc_tree(loss, "example.png")
```

### 7. Save

- Save optimized agents for later use.
- Optionally, save the optimizer state for later use.

Example:
```python
multiwoz_24_system.save_agents(file_name="agents.json")
optimizer.save_optimizer_state(file_path="optimizer_state.json")
```

### 8. Explore Examples

- See `examples/multiwoz_24_examples/` for more details.

## License

This project is licensed under the MIT License.

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{yang2026gbcgradientbasedconnectionsoptimizing,
      title={GBC: Gradient-Based Connections for Optimizing Multi-Agent Systems}, 
      author={Xiaocheng Yang and Abdulrahman Alrabah and Dilek Hakkani-Tür and Gokhan Tur},
      year={2026},
      eprint={2606.28187},
      archivePrefix={arXiv},
      primaryClass={cs.MA},
      url={https://arxiv.org/abs/2606.28187}, 
}
```