import json
import os
import re
from typing import List

import torch
from transformers import BitsAndBytesConfig

from agentchord import (
    BaseAgentSystem,
    GBCAgent,
    ModelConfig,
    Multiwoz24Environment,
    MultiWOZ24MetaData,
    ParallelBlock,
)
from agentchord.gbc_object import GBC, GBCBase, visualize_gbc_tree
from agentchord.loss import MultiWOZ24Loss
from agentchord.optimizer import OPROOptimizer
from agentchord.utils import DONT_CHANGE_FOOTER, DONT_CHANGE_HEADER, WandBConfig

# Prompt templates for reading states
prompt_read_attraction_state = f"""
You are a helpful agent that can retrieve taxi domain dialogue states from the user.
{DONT_CHANGE_HEADER}
The keys are the names of the slots and the values are the values of the slots. The keys of the dialogue state are:
```json
{{
    "attraction-area": {{
        "type": "string",
        "description": "The area in which the attraction is located.",
        "enum": ["centre", "north", "south", "east", "west"]
    }},
    "attraction-name": {{
        "type": "string",
        "description": "The name of the attraction."
    }},
    "attraction-type": {{
        "type": "string",
        "description": "The type of the attraction.",
        "enum": ["museum", "swimmingpool", "architecture", "boat", "college", "nightclub", "entertainment", "cinema", "concerthall", "mutliple sports", "park", "theatre"]
    }}
}}
```
{DONT_CHANGE_FOOTER}
The entries of the dialogue state should be put in JSON format. One example of the dialogue state is:
```json
{{
    "attraction-area": "centre",
    "attraction-name": "The Fitzwilliam Museum",
    "attraction-type": "museum"
}}
```
Note that this is not a tool call, you should only output the JSON object.
When there is no relative information in the dialogue state, you should return an empty JSON object.
""".strip()
prompt_read_hotel_state = f"""
You are a helpful agent that can retrieve hotel domain dialogue states from the user.
{DONT_CHANGE_HEADER}
The keys are the names of the slots and the values are the values of the slots. The keys of the dialogue state are:
```json
{{
    "hotel-area": {{
        "type": "string",
        "description": "The area in which the hotel is located.",
        "enum": ["centre", "north", "south", "east", "west"]
    }},
    "hotel-book day": {{
        "type": "string",
        "description": "The day of the booking.",
        "enum": ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
    }},
    "hotel-book people": {{
        "type": "string",
        "description": "The number of people in the booking."
    }},
    "hotel-book stay": {{
        "type": "string",
        "description": "The number of days of the booking."
    }},
    "hotel-internet": {{
        "type": "string",
        "description": "Whether the hotel has internet.",
        "enum": ["yes", "no"]
    }},
    "hotel-name": {{
        "type": "string",
        "description": "The name of the hotel."
    }},
    "hotel-parking": {{
        "type": "string",
        "description": "Whether the hotel has parking.",
        "enum": ["yes", "no"]
    }},
    "hotel-pricerange": {{
        "type": "string",
        "description": "The price range of the hotel.",
        "enum": ["cheap", "moderate", "expensive"]
    }},
    "hotel-stars": {{
        "type": "string",
        "description": "The number of stars of the hotel.",
        "enum": ["0", "1", "2", "3", "4", "5"]
    }},
    "hotel-type": {{
        "type": "string",
        "description": "The type of the hotel.",
        "enum": ["bed and breakfast", "guesthouse", "hotel"]
    }}
}}
```
{DONT_CHANGE_FOOTER}
The entries of the dialogue state should be put in JSON format. One example of the dialogue state is:
```json
{{
    "hotel-area": "centre",
    "hotel-book day": "monday",
    "hotel-internet": "yes",
    "hotel-name": "The Cambridge Belfry",
    "hotel-parking": "yes",
    "hotel-pricerange": "moderate"
}}
```
Note that this is not a tool call, you should only output the JSON object.
When there is no relative information in the dialogue state, you should return an empty JSON object.
""".strip()
prompt_read_restaurant_state = f"""
You are a helpful agent that can retrieve restaurant domain dialogue states from the user.
{DONT_CHANGE_HEADER}
The keys are the names of the slots and the values are the values of the slots. The keys of the dialogue state are:
```json
{{
    "restaurant-area": {{
        "type": "string",
        "description": "The area in which the restaurant is located.",
        "enum": ["centre", "north", "south", "east", "west"]
    }},
    "restaurant-book day": {{
        "type": "string",
        "description": "The day of the booking.",
        "enum": ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
    }},
    "restaurant-book people": {{
        "type": "string",
        "description": "The number of people in the booking."
    }},
    "restaurant-book time": {{
        "type": "string",
        "description": "The time of the booking in the format HH:MM."
    }},
    "restaurant-food": {{
        "type": "string",
        "description": "The type of food served at the restaurant.",
        "enum": ["international", "indian", "mediterranean", "italian", "vietnamese", "lebanese", "african", "modern european", "french", "european", "portuguese", "japanese", "seafood", "chinese", "turkish", "gastropub", "british", "thai", "spanish", "korean", "north american", "mexican", "asian oriental"]
    }},
    "restaurant-name": {{
        "type": "string",
        "description": "The name of the restaurant."
    }},
    "restaurant-pricerange": {{
        "type": "string",
        "description": "The price range of the restaurant.",
        "enum": ["cheap", "moderate", "expensive"]
    }}
}}
```
{DONT_CHANGE_FOOTER}
The entries of the dialogue state should be put in JSON format. One example of the dialogue state is:
```json
{{
    "restaurant-area": "centre",
    "restaurant-book day": "monday",
    "restaurant-book people": "2",
    "restaurant-food": "italian",
    "restaurant-name": "Trattoria Da Paolo"
}}
```
Note that this is not a tool call, you should only output the JSON object.
When there is no relative information in the dialogue state, you should return an empty JSON object.
""".strip()
prompt_read_taxi_state = f"""
You are a helpful agent that can retrieve taxi domain dialogue states from the user.
{DONT_CHANGE_HEADER}
The keys are the names of the slots and the values are the values of the slots. The keys of the dialogue state are:
```json
{{
    "taxi-arriveby": {{
        "type": "string",
        "description": "The time by which the taxi should arrive in the format HH:MM."
    }},
    "taxi-departure": {{
        "type": "string",
        "description": "The departure location of the taxi."
    }},
    "taxi-destination": {{
        "type": "string",
        "description": "The destination location of the taxi."
    }},
    "taxi-leaveat": {{
        "type": "string",
        "description": "The time at which the taxi should leave in the format HH:MM."
    }}
}}
```
{DONT_CHANGE_FOOTER}
The entries of the dialogue state should be put in JSON format. One example of the dialogue state is:
```json
{{
    "taxi-arriveby": "17:45",
    "taxi-departure": "Cambridge city center",
    "taxi-destination": "The Eagle pub"
}}
```
Note that this is not a tool call, you should only output the JSON object.
When there is no relative information in the dialogue state, you should return an empty JSON object.
""".strip()
prompt_read_train_state = f"""
You are a helpful agent that can retrieve train domain dialogue states from the user.
{DONT_CHANGE_HEADER}
The keys are the names of the slots and the values are the values of the slots. The keys of the dialogue state are:
```json
{{
    "train-arriveby": {{
        "type": "string",
        "description": "The time by which the train arrives in the format HH:MM."
    }},
    "train-book people": {{
        "type": "string",
        "description": "The number of people in the booking."
    }},
    "train-day": {{
        "type": "string",
        "description": "The day on which the train departs.",
        "enum": ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
    }},
    "train-departure": {{
        "type": "string",
        "description": "The departure location of the train."
    }},
    "train-destination": {{
        "type": "string",
        "description": "The destination location of the train."
    }},
    "train-leaveat": {{
        "type": "string",
        "description": "The time at which the train leaves in the format HH:MM."
    }}
}}
```
{DONT_CHANGE_FOOTER}
The entries of the dialogue state should be put in JSON format. One example of the dialogue state is:
```json
{{
    "train-arriveby": "17:45",
    "train-book people": "2",
    "train-day": "monday",
    "train-departure": "Cambridge",
    "train-destination": "London"
}}
```
Note that this is not a tool call, you should only output the JSON object.
When there is no relative information in the dialogue state, you should return an empty JSON object.
""".strip()

# Prompt templates for using tools
prompt_use_attraction_tool = """
You are a helpful agent that can properly use the attraction domain tools to make queries or bookings for the user.
You will be given the dialogue state of the user, which is retreived from the previous agent and contains the information about the user's request.
Based on the dialogue state and the dialogue history, you should decide whether to use the tools or not.
If you decide to use the tools, you should indicate what operation you have performed and what the result is to the next agent using the terminate tool after you have used all the necessary tools.
If you decide not to use the tools, you should indicate that you have not used the tools and provide a reason for not using them.
""".strip()
prompt_use_hotel_tool = """
You are a helpful agent that can properly use the hotel domain tools to make queries or bookings for the user.
You will be given the dialogue state of the user, which is retreived from the previous agent and contains the information about the user's request.
Based on the dialogue state and the dialogue history, you should decide whether to use the domain tools or not.
If you decide to use the tools, you should indicate what operation you have performed and what the result is to the next agent using the terminate tool after you have used all the necessary tools.
If you decide not to use the tools, you should indicate that you have not used the tools and provide a reason for not using them.
""".strip()
prompt_use_restaurant_tool = """
You are a helpful agent that can properly use the restaurant domain tools to make queries or bookings for the user.
You will be given the dialogue state of the user, which is retreived from the previous agent and contains the information about the user's request.
Based on the dialogue state and the dialogue history, you should decide whether to use the domain tools or not.
If you decide to use the tools, you should indicate what operation you have performed and what the result is to the next agent using the terminate tool after you have used all the necessary tools.
If you decide not to use the tools, you should indicate that you have not used the tools and provide a reason for not using them.
""".strip()
prompt_use_taxi_tool = """
You are a helpful agent that can properly use the taxi domain tools to make queries or bookings for the user.
You will be given the dialogue state of the user, which is retreived from the previous agent and contains the information about the user's request.
Based on the dialogue state and the dialogue history, you should decide whether to use the domain tools or not.
If you decide to use the tools, you should indicate what operation you have performed and what the result is to the next agent using the terminate tool after you have used all the necessary tools.
If you decide not to use the tools, you should indicate that you have not used the tools and provide a reason for not using them.
""".strip()
prompt_use_train_tool = """
You are a helpful agent that can properly use the train domain tools to make queries or bookings for the user.
You will be given the dialogue state of the user, which is retreived from the previous agent and contains the information about the user's request.
Based on the dialogue state and the dialogue history, you should decide whether to use the domain tools or not.
If you decide to use the tools, you should indicate what operation you have performed and what the result is to the next agent using the terminate tool after you have used all the necessary tools.
If you decide not to use the tools, you should indicate that you have not used the tools and provide a reason for not using them.
""".strip()

# Prompt template for generating response
prompt_generate_response = """
You are a helpful agent that can generate a response based on the dialogue history and the actions performed by the previous agents.
The previous agents have performed queries or bookings according to the user's request.
Now you should generate the final response to the user according to the dialogue history and the actions performed by the previous agents.
Note that the response will be directly sent to the user, so it should be clear and concise.
""".strip()

# Configuration for the model
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
    max_new_tokens=128,
    # temperature=0.0,
    do_sample=False,
    gradient_strategy="product_probs",
    connection_strategy="mean_l1_norm",
    chat_template_path="src/agentchord/model/chat_templates/tool_chat_template_llama3.1_json.jinja"
)


class Multiwoz24DomainUnit(BaseAgentSystem):
    def __init__(
            self,
            prompt_read_state: str,
            prompt_use_tool: list,
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
        StateAgent = GBCAgent(
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
                GBCAgent(
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
        except json.JSONDecodeError:
            _json_markdown_re = re.compile(r"```(json)?(.*)```", re.DOTALL)
            match = _json_markdown_re.search(dialogue_state_raw)
            if match:
                try:
                    dialogue_state = json.loads(match.group(2))
                except json.JSONDecodeError:
                    print(f"Failed to parse JSON from dialogue state: {dialogue_state_raw}")
                    dialogue_state = dict()
            else:
                dialogue_state = dict()
        dialogue_state = {slot: value for slot, value in dialogue_state.items() if len(slot.split('-')) == 2 and isinstance(value, str)}
        if isinstance(metadata.output, GBCBase):
            metadata.dialogue_state = GBC(
                dialogue_state,
                connections=metadata.output.get_connections(),
                weights=[1.0]
            )
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
        else:
            print(f"Output: {metadata.output}")
            print(f"Output type: {type(metadata.output)}")
            metadata.dialogue_state = GBC(
                dialogue_state,
                connections=metadata.output,
                weights=[1.0]
            )
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
                    connections=metadata.output,
                    weights=[1.0]
                )
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
            prompt_read_state=prompt_read_attraction_state,
            prompt_use_tool=prompt_use_attraction_tool,
            tools=["query_attraction"],
            system_name="attraction_unit",
            domain="attraction",
            environment=environment,
            maximum_loops=maximum_loops,
            log_name=log_name
        )
        hotel_unit = Multiwoz24DomainUnit(
            prompt_read_state=prompt_read_hotel_state,
            prompt_use_tool=prompt_use_hotel_tool,
            tools=["query_hotel", "book_hotel"],
            system_name="hotel_unit",
            domain="hotel",
            environment=environment,
            maximum_loops=maximum_loops,
            log_name=log_name
        )
        restaurant_unit = Multiwoz24DomainUnit(
            prompt_read_state=prompt_read_restaurant_state,
            prompt_use_tool=prompt_use_restaurant_tool,
            tools=["query_restaurant", "book_restaurant"],
            system_name="restaurant_unit",
            domain="restaurant",
            environment=environment,
            maximum_loops=maximum_loops,
            log_name=log_name
        )
        taxi_unit = Multiwoz24DomainUnit(
            prompt_read_state=prompt_read_taxi_state,
            prompt_use_tool=prompt_use_taxi_tool,
            tools=["book_taxi"],
            system_name="taxi_unit",
            domain="taxi",
            environment=environment,
            maximum_loops=maximum_loops,
            log_name=log_name
        )
        train_unit = Multiwoz24DomainUnit(
            prompt_read_state=prompt_read_train_state,
            prompt_use_tool=prompt_use_train_tool,
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
        response_agent = GBCAgent(
            system_name="response_agent",
            environment=environment,
            tools=[],
            prompt=prompt_generate_response,
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
        metadata.input = GBC(
            f"Dialogue History:\n{grounding_utterance}",
            connections=[grounding_utterance],
            weights=[1.0]
        )
        return metadata
    
    def _units_aggregation(self, metadata: List[MultiWOZ24MetaData]) -> MultiWOZ24MetaData:
        """
        Aggregate the outputs from the domain units and prepare the input for the response agent.
        """
        assert isinstance(metadata, list)
        final_metadata = metadata[-1]
        final_input = [GBC(
                f"Dialogue History:\n{final_metadata.grounding_utterance}",
                connections=[
                    final_metadata.grounding_utterance,
                ],
                weights=[1.0]
            )]
        final_note = [""]
        final_state = dict()
        final_state_connections = list()
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
            final_state_connections.extend(single_metadata.dialogue_state.get_connections())
            final_tool.extend(single_metadata.tool)
        final_metadata.input = final_input
        final_metadata.note = final_note
        final_metadata.dialogue_state = GBC(
            final_state,
            connections=final_state_connections,
            weights=[1.0]*len(final_state_connections)
        )
        final_metadata.tool = final_tool
        return final_metadata

    def on_finalization(self, metadata: MultiWOZ24MetaData) -> MultiWOZ24MetaData:
        """
        Finalize the metadata after the response agent has generated the response.
        """
        metadata.system_response = metadata.output or metadata.note
        return metadata

multiwoz_24_system = Multiwoz24System("multiwoz_24_system", Multiwoz24Environment(), log_name="multiwoz_experiment_product_probs_mean_l1_norm.log")
print(multiwoz_24_system.get_pipeline_description())

optimizer = OPROOptimizer(
    agents=multiwoz_24_system.get_agents(),
    model_config=ModelConfig(
        client_model="openai/gpt-4.1-2025-04-14",
    ),
    log_name="multiwoz_experiment_product_probs_mean_l1_norm.log",
    wandb_config=WandBConfig(
        project="AgentChord",
        config={
            "dataset": "MultiWOZ-24",
            "system": "Multiwoz24System",
            "gradient_strategy": "product_probs",
            "connection_strategy": "mean_l1_norm",
            "model": "Llama3.1-8B-Instruct",
            "optimizer": "OPROOptimizer",
            "optimizer_model": "gpt-4.1-2025-04-14",
        }
    )
)
loss_fn = MultiWOZ24Loss()
optimization_steps = 0
dialogue_idx_pool = list()

multiwoz_24_system.load_agents(file_name="experiments/checkpoint-3/multiwoz_24_agents_29.json")
optimizer.load_optimizer_state(file_path="experiments/checkpoint-3/optimizer_state_29.json")
for dialogue_idx, dialogue_case in enumerate(Multiwoz24Environment.iterate_test_cases(mode="train")):
    print(f"{dialogue_idx} Dialogue: {dialogue_case.dialogue_idx}")
    if dialogue_idx < 30:
        continue
    dialogue_idx_pool.append(dialogue_case.dialogue_idx)
    for turn_idx, turn_case in enumerate(dialogue_case.iterate_dialog_turns()):
        print(f"{dialogue_idx} Dialogue: {dialogue_case.dialogue_idx},\tTurn: {turn_idx}")
        multiwoz_24_system.set_environment(environment=turn_case)
        result = multiwoz_24_system.run()
        evaluation_result = turn_case.evaluate(result)
        loss = loss_fn.compute_loss(prediction=result, evaluation_result=evaluation_result, type="joint_goal_accuracy")
        loss.backward(bandwidth=1)
    evaluation_result = Multiwoz24Environment.evaluate_test_cases(mode="train", dialogue_indices=dialogue_case.dialogue_idx)
    loss = loss_fn.compute_loss(prediction=result, evaluation_result=evaluation_result, type="inform_success")
    loss.backward(bandwidth=1)
    if (dialogue_idx + 1) % 10 == 0:
        evaluation_result = Multiwoz24Environment.evaluate_test_cases(mode="train", dialogue_indices=dialogue_idx_pool)
        optimizer.step(
            performance=f"Inform: {evaluation_result.inform['total']}; Success : {evaluation_result.success['total']}; Joint Goal Accuracy: {evaluation_result.joint_goal_accuracy}",
            performance_dict={
                "inform": evaluation_result.inform['total'],
                "success": evaluation_result.success['total'],
                "joint_goal_accuracy": evaluation_result.joint_goal_accuracy
            }
        )
        optimization_steps = (dialogue_idx + 1) // 10

        if not os.path.exists(f"experiments/checkpoint-{optimization_steps}"):
            os.makedirs(f"experiments/checkpoint-{optimization_steps}")
        multiwoz_24_system.save_agents(file_name=f"experiments/checkpoint-{optimization_steps}/multiwoz_24_agents_{dialogue_idx}.json")
        optimizer.save_optimizer_state(file_path=f"experiments/checkpoint-{optimization_steps}/optimizer_state_{dialogue_idx}.json")
        dialogue_idx_pool = list()
    
    if dialogue_idx >= 99:
        break

optimizer.finish_wandb()
