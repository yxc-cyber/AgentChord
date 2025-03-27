import json
import re

from agentchord import (
    BaseAgent,
    BaseAgentSystem,
    ModelConfig,
    Multiwoz24Environment,
    MultiWOZ24MetaData,
)

prompt_read_state = "You are a helpful agent that can repeat what the user says."
prompt_generate_response = "You are a helpful agent that can generate a response to the user."

class Multiwoz24System(BaseAgentSystem):
    def __init__(self, system_name: str, environment: Multiwoz24Environment, maximum_loops: int = 50, log_name: str = ""):
        super().__init__(system_name, environment, maximum_loops, log_name)
        StateAgent = BaseAgent(
            system_name="state_agent",
            environment=environment,
            prompt=prompt_read_state,
            model_config=ModelConfig(client_model="openai/gpt-4o-mini"),
            log_name=log_name
        )
        ResponseAgent = BaseAgent(
            system_name="response_agent",
            environment=environment,
            prompt=prompt_generate_response,
            model_config=ModelConfig(client_model="openai/gpt-4o-mini"),
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
        dialogue_state_raw = metadata.output.strip()
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
    
multiwoz_24_system = Multiwoz24System("multiwoz_24_system", Multiwoz24Environment(), log_name="multiwoz_24_system.log")
print(multiwoz_24_system.get_pipeline_description())

for dialogue_idx, dialogue_case in enumerate(Multiwoz24Environment.iterate_test_cases(mode="test")):
    for turn_idx, turn_case in enumerate(dialogue_case.iterate_dialog_turns()):
        multiwoz_24_system.set_environment(environment=turn_case)
        result = multiwoz_24_system.run()
        evaluation_result = turn_case.evaluate(result)
        if turn_idx >= 5:
            break
    if dialogue_idx >= 5:
        break
evaluation_result = Multiwoz24Environment.evaluate_test_cases(mode="test").to_json("./examples/multiwoz_24_evaluation.json")
print(evaluation_result)