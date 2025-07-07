import json
import re

import torch
from transformers import BitsAndBytesConfig

from agentchord import (
    BaseAgentSystem,
    GBCAgent,
    ModelConfig,
    Multiwoz24Environment,
    MultiWOZ24MetaData,
)
from agentchord.gbc_object import GBC, GBCBase, visualize_gbc_tree

prompt_read_state = """
You are a helpful agent that can retrieve dialogue states from the user.
The keys are the names of the slots and the values are the values of the slots. The keys of the dialogue state are:
```json
{
    "taxi-arriveby": {
        "type": "string",
        "description": "The time by which the taxi should arrive in the format HH:MM."
    },
    "taxi-departure": {
        "type": "string",
        "description": "The departure location of the taxi."
    },
    "taxi-destination": {
        "type": "string",
        "description": "The destination location of the taxi."
    },
    "taxi-leaveat": {
        "type": "string",
        "description": "The time at which the taxi should leave in the format HH:MM."
    }
}
```
The entries of the dialogue state should be put in JSON format. One example of the dialogue state is:
```json
{
    "taxi-arriveby": "17:45",
    "taxi-departure": "Cambridge city center",
    "taxi-destination": "The Eagle pub"
}
```
Note that this is not a tool call, you should only output the JSON object.
""".strip()
prompt_generate_response = """
You are an advanced AI assistant specializing in conversational dialogues. You can interact with the database and provide service to assist users in completing complex tasks. 
Each task may involve multiple sub-tasks, such as finding restaurants, making reservations, booking hotels, locating attractions, and arranging transportation by checking for trains and buying train tickets.
You are given the dialogue history and the current dialogue state. You need to query the database and generate a response to the user based on the dialogue history and the database results.
The generated response will be directly sent to the user, so it should be proper for a human to read.
"""

class Multiwoz24System(BaseAgentSystem):
    def __init__(self, system_name: str, environment: Multiwoz24Environment, maximum_loops: int = 5, log_name: str = ""):
        super().__init__(
            system_name=system_name,
            environment=environment, 
            maximum_loops=maximum_loops,
            log_name=log_name
        )

        bnb_config = BitsAndBytesConfig(
            load_in_4bit = True,
            bnb_4bit_use_double_quant = True,
            bnb_4bit_quant_type = "nf4",
            bnb_4bit_compute_dtype = torch.bfloat16
        )
        config = ModelConfig(
            local_model="LlamaModel",
            model_path="/shared/storage-01/users/xy61/models/Llama3.1-8B-Instruct",
            quantization_config=bnb_config,
            max_new_tokens=1024,
            # temperature=0.0,
            do_sample=False,
            gradient_strategy="sum_squares",
            connection_strategy="mean_l1_norm",
            chat_template_path="src/agentchord/model/chat_templates/tool_chat_template_llama3.1_json.jinja"
        )

        StateAgent = GBCAgent(
            system_name="state_agent",
            environment=environment,
            tools=[],
            prompt=prompt_read_state,
            model_config=config,
            maximum_loops=maximum_loops,
            log_name=log_name,
        )
        ResponseAgent = GBCAgent(
            system_name="response_agent",
            environment=environment,
            tools=["book_taxi"],
            prompt=prompt_generate_response,
            model_config=config,
            maximum_loops=maximum_loops,
            log_name=log_name
        )

        self.add_subsystem(StateAgent)
        self.add_subsystem(ResponseAgent)
        self.add_on_start_action("response_agent", "read_state", self._read_state)

    def on_initialization(self) -> MultiWOZ24MetaData:
        metadata = self.environment.get_initial_metadata()
        grounding_utterance = metadata.grounding_utterance
        metadata.input = GBC(
            f"Dialogue History:\n{grounding_utterance}",
            connections=[grounding_utterance],
            weights=[1.0]
        )
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
        metadata.input = [
            GBC(
                f"Dialogue History:\n{grounding_utterance}",
                connections=[
                    metadata.grounding_utterance,
                ],
                weights=[1.0]
            ),
            GBC(
                f"Dialogue State:\n{json.dumps(dialogue_state)}",
                connections=metadata.output.get_connections(),
                weights=[1.0]
            )
        ]
        metadata.note =[
            "",
            metadata.note
        ]
        return metadata

    def on_finalization(self, metadata: MultiWOZ24MetaData) -> MultiWOZ24MetaData:
        """
        Finalize the metadata after the response agent has generated the response.
        """
        metadata.system_response = metadata.output.strip() or metadata.note.strip()
        return metadata
    
multiwoz_24_system = Multiwoz24System("multiwoz_24_system", Multiwoz24Environment(), log_name="multiwoz_24_gbc_example.log")
print(multiwoz_24_system.get_pipeline_description())

for dialogue_idx, dialogue_case in enumerate(Multiwoz24Environment.iterate_test_cases(mode="test")):
    for turn_idx, turn_case in enumerate(dialogue_case.iterate_dialog_turns()):
        multiwoz_24_system.set_environment(environment=turn_case)
        result = multiwoz_24_system.run()
        # if turn_idx >= 0:
        break
    # if dialogue_idx >= 0:
    break

assert isinstance(result, MultiWOZ24MetaData)
assert isinstance(result.output, str)
assert isinstance(result.output, GBCBase)
print("Final Response:", result.output)
visualize_gbc_tree(result.output, save_path="examples/multiwoz_24_examples/multiwoz_24_gbc_example.png")