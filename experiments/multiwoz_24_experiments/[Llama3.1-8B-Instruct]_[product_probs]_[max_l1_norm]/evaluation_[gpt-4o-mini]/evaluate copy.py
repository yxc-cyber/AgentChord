import json
import re
from typing import List

from agentchord import (
    BaseAgent,
    BaseAgentSystem,
    ModelConfig,
    Multiwoz24Environment,
    MultiWOZ24MetaData,
    ParallelBlock,
)

# Configuration for the model
config = ModelConfig(client_model="openai/gpt-4o-mini", temperature=0.0)


class Multiwoz24DomainUnit(BaseAgentSystem):
    def __init__(
            self,
            prompt_read_state: str,
            prompt_use_tool: str,
            tools: list,
            system_name: str,
            domain: str,
            environment: Multiwoz24Environment,
            maximum_loops: int = 5,
            log_name: str = ""
        ):
        super().__init__(
            system_name=system_name,
            environment=environment, 
            maximum_loops=maximum_loops,
            log_name=log_name
        )
        StateAgent = BaseAgent(
            system_name=f"{domain}_state_agent",
            environment=environment,
            tools=[],
            prompt=prompt_read_state,
            model_config=config,
            maximum_loops=maximum_loops,
            log_name=log_name,
        )
        ToolAgent = ParallelBlock(
            system_name=f"{domain}_tool_agent_block",
            environment=environment,
            subsystems=[
                BaseAgent(
                    system_name=f"{domain}_tool_agent({tool})",
                    environment=environment,
                    tools=[tool],
                    prompt=prompt_use_tool,
                    model_config=config,
                    maximum_loops=maximum_loops,
                    log_name=log_name
                ) for tool in tools
            ],
            maximum_loops=maximum_loops,
            log_name=log_name,
            return_list=False
        )
        self.add_subsystem(StateAgent)
        self.add_subsystem(ToolAgent)
        self.add_on_completion_action(f"{domain}_state_agent", "read_state", self._read_state)

    def _read_state(self, metadata: MultiWOZ24MetaData) -> MultiWOZ24MetaData:
        """
        Read the state from the the output of the state agent.
        """
        grounding_utterance = metadata.grounding_utterance
        dialogue_state_raw = metadata.output.strip() or metadata.note.strip()
        try:
            dialogue_state = json.loads(dialogue_state_raw)
            if dialogue_state is None:
                dialogue_state = dict()
        except json.JSONDecodeError:
            _json_markdown_re = re.compile(r"```(json)?(.*?)```", re.DOTALL)
            match = _json_markdown_re.search(dialogue_state_raw)
            if match:
                try:
                    dialogue_state = json.loads(match.group(2))
                except json.JSONDecodeError:
                    print(f"Failed to parse JSON from dialogue state: {dialogue_state_raw}")
                    dialogue_state = dict()
            else:
                dialogue_state = dict()
        if not isinstance(dialogue_state, dict):
            if isinstance(dialogue_state, list) and len(dialogue_state) > 0 and isinstance(dialogue_state[0], dict):
                dialogue_state_new = dict()
                for item in dialogue_state:
                    if isinstance(item, dict):
                        dialogue_state_new.update(item)
                dialogue_state = dialogue_state_new
            else:
                print(f"Dialogue state is not a dictionary: {dialogue_state}")
                dialogue_state = dict()
        dialogue_state = {slot: value for slot, value in dialogue_state.items() if len(slot.split('-')) == 2 and isinstance(value, str)}
        metadata.dialogue_state = dialogue_state
        metadata.input = [
            f"Dialogue History:\n{grounding_utterance}",
            f"Dialogue State:\n{json.dumps(dialogue_state)}",
        ]
        metadata.note =[
            "",
            metadata.note
        ]
        return metadata

class Multiwoz24System(BaseAgentSystem):
    def __init__(self, system_name: str, environment: Multiwoz24Environment, maximum_loops: int = 5, log_name: str = ""):
        super().__init__(
            system_name=system_name,
            environment=environment, 
            maximum_loops=maximum_loops,
            log_name=log_name
        )
        attraction_unit = Multiwoz24DomainUnit(
            prompt_read_state="",
            prompt_use_tool="",
            tools=["query_attraction"],
            system_name="attraction_unit",
            domain="attraction",
            environment=environment,
            maximum_loops=maximum_loops,
            log_name=log_name
        )
        hotel_unit = Multiwoz24DomainUnit(
            prompt_read_state="",
            prompt_use_tool="",
            tools=["query_hotel", "book_hotel"],
            system_name="hotel_unit",
            domain="hotel",
            environment=environment,
            maximum_loops=maximum_loops,
            log_name=log_name
        )
        restaurant_unit = Multiwoz24DomainUnit(
            prompt_read_state="",
            prompt_use_tool="",
            tools=["query_restaurant", "book_restaurant"],
            system_name="restaurant_unit",
            domain="restaurant",
            environment=environment,
            maximum_loops=maximum_loops,
            log_name=log_name
        )
        taxi_unit = Multiwoz24DomainUnit(
            prompt_read_state="",
            prompt_use_tool="",
            tools=["book_taxi"],
            system_name="taxi_unit",
            domain="taxi",
            environment=environment,
            maximum_loops=maximum_loops,
            log_name=log_name
        )
        train_unit = Multiwoz24DomainUnit(
            prompt_read_state="",
            prompt_use_tool="",
            tools=["query_train", "book_train"],
            system_name="train_unit",
            domain="train",
            environment=environment,
            maximum_loops=maximum_loops,
            log_name=log_name
        )
        units_block = ParallelBlock(
            system_name="multiwoz_24_units_block",
            environment=environment,
            subsystems=[attraction_unit, hotel_unit, restaurant_unit, taxi_unit, train_unit],
            maximum_loops=maximum_loops,
            log_name=log_name,
            return_list=True
        )
        response_agent = BaseAgent(
            system_name="response_agent",
            environment=environment,
            tools=[],
            prompt="",
            model_config=config,
            maximum_loops=maximum_loops,
            log_name=log_name
        )
        self.add_subsystem(units_block)
        self.add_on_completion_action("multiwoz_24_units_block", "units_aggregation", self._units_aggregation)
        self.add_subsystem(response_agent)

    def on_initialization(self) -> MultiWOZ24MetaData:
        metadata = self.environment.get_initial_metadata()
        grounding_utterance = metadata.grounding_utterance
        metadata.input = f"Dialogue History:\n{grounding_utterance}"
        return metadata
    
    def _units_aggregation(self, metadata: List[MultiWOZ24MetaData]) -> MultiWOZ24MetaData:
        """
        Aggregate the outputs from the domain units and prepare the input for the response agent.
        """
        assert isinstance(metadata, list)
        final_metadata = metadata[-1]
        final_input = [f"Dialogue History:\n{final_metadata.grounding_utterance}"]
        final_note = [""]
        final_state = dict()
        final_tool = list()
        for single_metadata in metadata:
            if isinstance(single_metadata.output, list):
                final_input.extend(single_metadata.output)
            else:
                final_input.append(single_metadata.output)
            if isinstance(single_metadata.note, list):
                final_note.extend(single_metadata.note)
            else:
                final_note.append(single_metadata.note)
            final_state.update(single_metadata.dialogue_state)
            # final_tool.extend(single_metadata.tool)
            final_tool = single_metadata.tool
        final_metadata.input = final_input
        final_metadata.note = final_note
        final_metadata.dialogue_state = final_state
        final_metadata.tool = final_tool
        return final_metadata

    def on_finalization(self, metadata: MultiWOZ24MetaData) -> MultiWOZ24MetaData:
        """
        Finalize the metadata after the response agent has generated the response.
        """
        metadata.system_response = metadata.output or metadata.note
        return metadata

multiwoz_24_system = Multiwoz24System("multiwoz_24_system", Multiwoz24Environment(), log_name="multiwoz_experiment_eval_product_probs_max_l1_norm.log")
print(multiwoz_24_system.get_pipeline_description())