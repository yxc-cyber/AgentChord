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
from agentchord.gbc_object import GBC, GBCBase, save_gbc_tree
from agentchord.loss import MultiWOZ24Loss
from agentchord.optimizer import OPROOptimizer
from agentchord.utils import DONT_CHANGE_FOOTER, DONT_CHANGE_HEADER, WandBConfig

# training configuration
WORKING_DIR = "experiments/multiwoz_24_experiments/[Llama3.1-8B-Instruct]_[product_probs]_[mean_product_input]_v2/human_evaluation"
LOG_NAME = "multiwoz_experiment_human_product_probs_mean_product_input_v2.log"
GRADIENT_STRATEGY = "product_probs"
CONNECTION_STRATEGY = "mean_product_input"
MODEL = "Llama3.1-8B-Instruct"
# training resume configuration
RESUME_DIR = None  # "experiments/multiwoz_24_experiments/[Llama3.1-8B-Instruct]_[product_probs]_[mean_product_input]_v2/training/checkpoint-2"
SKIP_SAMPLES = 0  # 20
# model loading configuration
LOCAL_MODEL = "LlamaModel"
MODEL_PATH = "/shared/storage-01/users/xy61/models/Llama3.1-8B-Instruct"
CHAT_TEMPLATE_PATH = "src/agentchord/model/chat_templates/tool_chat_template_llama3.1_json.jinja"

prompt_read_state_attraction = f"""
You are a helpful agent that can retrieve dialogue states from the user. The dialogue state, essentially, is the intent of the user that have been shown in the dialogue history. Make sure you cover all the mentioned intents in the dialogue history.
The dialogue state should be formatted as a JSON object that contains the slots and their values. The keys of the dialogue state are the names of the intent slots, and the values are the values of the slots.
{DONT_CHANGE_HEADER}
The keys of the dialogue state are:
1. attraction-area
description: The area in which the attraction is located.
enum: ["centre", "north", "south", "east", "west"]
2. attraction-name
description: The name of the attraction.
3. attraction-type
description: The type of the attraction.
enum: ["museum", "swimmingpool", "architecture", "boat", "college", "nightclub", "entertainment", "cinema", "concerthall", "mutliple sports", "park", "theatre"]
{DONT_CHANGE_FOOTER}

The dialogue state should be put in JSON format. One example of the dialogue state is:
```json
{{
    "attraction-area": "centre",
    "attraction-name": "Theatre Royal",
    "attraction-type": "theatre"
}}
```
""".strip()

prompt_read_state_hotel = f"""
You are a helpful agent that can retrieve dialogue states from the user. The dialogue state, essentially, is the intent of the user that have been shown in the dialogue history. Make sure you cover all the mentioned intents in the dialogue history.
The dialogue state should be formatted as a JSON object that contains the slots and their values. The keys of the dialogue state are the names of the intent slots, and the values are the values of the slots.
{DONT_CHANGE_HEADER}
The keys of the dialogue state are:
1. hotel-area
description: The area in which the hotel is located.
enum: ["centre", "north", "south", "east", "west"]
2. hotel-book day
description: The day of the booking.
enum: ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
3. hotel-book people
description: The number of people in the booking.
4. hotel-book stay
description: The number of days of the booking.
5. hotel-internet
description: Whether the hotel has internet.
enum: ["yes", "no"]
6. hotel-name
description: The name of the hotel.
7. hotel-parking
description: Whether the hotel has parking.
enum: ["yes", "no"]
8. hotel-pricerange
description: The price range of the hotel.
enum: ["cheap", "moderate", "expensive"]
9. hotel-stars
description: The number of stars of the hotel.
enum: ["0", "1", "2", "3", "4", "5"]
10. hotel-type
description: The type of the hotel.
enum: ["bed and breakfast", "guesthouse", "hotel"]
{DONT_CHANGE_FOOTER}

The dialogue state should be put in JSON format. One example of the dialogue state is:
```json
{{
    "hotel-area": "centre",
    "hotel-book day": "monday",
    "hotel-book people": "2",
    "hotel-book stay": "3",
    "hotel-internet": "yes",
    "hotel-name": "Theatre Royal Hotel",
    "hotel-parking": "no",
    "hotel-pricerange": "moderate",
    "hotel-stars": "4",
    "hotel-type": "guesthouse"
}}
```
""".strip()

prompt_read_state_restaurant = f"""
You are a helpful agent that can retrieve dialogue states from the user. The dialogue state, essentially, is the intent of the user that have been shown in the dialogue history. Make sure you cover all the mentioned intents in the dialogue history.
The dialogue state should be formatted as a JSON object that contains the slots and their values. The keys of the dialogue state are the names of the intent slots, and the values are the values of the slots.
{DONT_CHANGE_HEADER}
The keys of the dialogue state are:
1. restaurant-area
description: The area in which the restaurant is located.
enum: ["centre", "north", "south", "east", "west"]
2. restaurant-book day
description: The day of the booking.
enum: ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
3. restaurant-book people
description: The number of people in the booking.
4. restaurant-book time
description: The time of the booking in the format HH:MM.
5. restaurant-food
description: The type of food served at the restaurant.
enum: ["international", "indian", "mediterranean", "italian", "vietnamese", "lebanese", "african", "modern european", "french", "european", "portuguese", "japanese", "seafood", "chinese", "turkish", "gastropub", "british", "thai", "spanish", "korean", "north american", "mexican", "asian oriental"]
6. restaurant-name
description: The name of the restaurant.
7. restaurant-pricerange
description: The price range of the restaurant.
enum: ["cheap", "moderate", "expensive"]
{DONT_CHANGE_FOOTER}

The dialogue state should be put in JSON format. One example of the dialogue state is:
```json
{{
    "restaurant-area": "centre",
    "restaurant-book day": "monday",
    "restaurant-book people": "2",
    "restaurant-book time": "19:00"
}}
```
""".strip()

prompt_read_state_taxi = f"""
You are a helpful agent that can retrieve dialogue states from the user. The dialogue state, essentially, is the intent of the user that have been shown in the dialogue history. Make sure you cover all the mentioned intents in the dialogue history.
The dialogue state should be formatted as a JSON object that contains the slots and their values. The keys of the dialogue state are the names of the intent slots, and the values are the values of the slots.
{DONT_CHANGE_HEADER}
The keys of the dialogue state are:
1. taxi-arriveby
description: The time by which the taxi should arrive in the format HH:MM.
2. taxi-departure
description: The departure location of the taxi.
3. taxi-destination
description: The destination location of the taxi.
4. taxi-leaveat
description: The time at which the taxi should leave in the format HH:MM.
{DONT_CHANGE_FOOTER}

The dialogue state should be put in JSON format. One example of the dialogue state is:
```json
{{
    "taxi-arriveby": "19:00",
    "taxi-departure": "Theatre Royal",
    "taxi-destination": "Cambridge City Centre",
    "taxi-leaveat": "18:30"
}}
```
""".strip()

prompt_read_state_train = f"""
You are a helpful agent that can retrieve dialogue states from the user. The dialogue state, essentially, is the intent of the user that have been shown in the dialogue history. Make sure you cover all the mentioned intents in the dialogue history.
The dialogue state should be formatted as a JSON object that contains the slots and their values. The keys of the dialogue state are the names of the intent slots, and the values are the values of the slots.
{DONT_CHANGE_HEADER}
The keys of the dialogue state are:
1. train-arriveby
description: The time by which the train arrives in the format HH:MM.
2. train-book people
description: The number of people in the booking.
3. train-day
description: The day on which the train departs.
enum: ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
4. train-departure
description: The departure location of the train.
5. train-destination
description: The destination location of the train.
6. train-leaveat
description: The time at which the train leaves in the format HH:MM.
{DONT_CHANGE_FOOTER}

The dialogue state should be put in JSON format. One example of the dialogue state is:
```json
{{
    "train-arriveby": "19:00",
    "train-book people": "2",
    "train-day": "monday",
    "train-departure": "Cambridge",
    "train-destination": "London",
    "train-leaveat": "18:00"
}}
```
""".strip()


prompt_use_tool = """
You are a helpful agent that can properly use the tools to make queries or bookings for the user.
You will be provided with a dialogue history and a dialogue state extracted by the previous agent. The dialogue state indicates the intention of the user in the current turn.
Based on the dialogue history and the dialogue state, you should decide whether to use the tools or not. You can use another tool after you have used one and received the tool result.
Finally, you should generate a summary of the actions you have taken and the results you have received.
""".strip()


prompt_generate_response = """
You are a helpful agent that can generate a response based on the dialogue history and the actions performed by the previous agents.
The previous agents have performed queries or bookings according to the user's request. You will receive the summaries of the actions in the inputs.
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
    local_model=LOCAL_MODEL,
    model_path=MODEL_PATH,
    quantization_config=bnb_config,
    max_new_tokens=128,
    # temperature=0.0,
    do_sample=False,
    gradient_strategy=GRADIENT_STRATEGY,
    connection_strategy=CONNECTION_STRATEGY,
    chat_template_path=CHAT_TEMPLATE_PATH
)


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
            if dialogue_state is None:
                dialogue_state = dict()
        except json.JSONDecodeError:
            _json_markdown_re = re.compile(r"```(json)?(.*?)```", re.DOTALL)
            match = _json_markdown_re.search(dialogue_state_raw)
            if match:
                try:
                    dialogue_state = json.loads(match.group(2))
                except json.JSONDecodeError:
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
            prompt_read_state=prompt_read_state_attraction,
            prompt_use_tool=prompt_use_tool,
            tools=["query_attraction"],
            system_name="attraction_unit",
            domain="attraction",
            environment=environment,
            maximum_loops=maximum_loops,
            log_name=log_name
        )
        hotel_unit = Multiwoz24DomainUnit(
            prompt_read_state=prompt_read_state_hotel,
            prompt_use_tool=prompt_use_tool,
            tools=["query_hotel", "book_hotel"],
            system_name="hotel_unit",
            domain="hotel",
            environment=environment,
            maximum_loops=maximum_loops,
            log_name=log_name
        )
        restaurant_unit = Multiwoz24DomainUnit(
            prompt_read_state=prompt_read_state_restaurant,
            prompt_use_tool=prompt_use_tool,
            tools=["query_restaurant", "book_restaurant"],
            system_name="restaurant_unit",
            domain="restaurant",
            environment=environment,
            maximum_loops=maximum_loops,
            log_name=log_name
        )
        taxi_unit = Multiwoz24DomainUnit(
            prompt_read_state=prompt_read_state_taxi,
            prompt_use_tool=prompt_use_tool,
            tools=["book_taxi"],
            system_name="taxi_unit",
            domain="taxi",
            environment=environment,
            maximum_loops=maximum_loops,
            log_name=log_name
        )
        train_unit = Multiwoz24DomainUnit(
            prompt_read_state=prompt_read_state_train,
            prompt_use_tool=prompt_use_tool,
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

multiwoz_24_system = Multiwoz24System("multiwoz_24_system", Multiwoz24Environment(), log_name=LOG_NAME)
print(multiwoz_24_system.get_pipeline_description())

loss_fn = MultiWOZ24Loss()

if RESUME_DIR is not None:
    multiwoz_24_system.load_agents(file_name=os.path.join(RESUME_DIR, "multiwoz_24_agents.json"))
for dialogue_idx, dialogue_case in enumerate(Multiwoz24Environment.iterate_test_cases(mode="train", random_seed=42, max_turns=4)):
    print(f"{dialogue_idx} Dialogue: {dialogue_case.dialogue_idx}")
    if dialogue_idx < SKIP_SAMPLES:
        continue
    print(f"Saving results to {WORKING_DIR}/dialogue_{dialogue_case.dialogue_idx}/")
    if not os.path.exists(os.path.join(WORKING_DIR, f"dialogue_{dialogue_case.dialogue_idx}")):
        os.makedirs(os.path.join(WORKING_DIR, f"dialogue_{dialogue_case.dialogue_idx}"))
    for turn_idx, turn_case in enumerate(dialogue_case.iterate_dialog_turns()):
        print(f"{dialogue_idx} Dialogue: {dialogue_case.dialogue_idx},\tTurn: {turn_idx}")
        multiwoz_24_system.set_environment(environment=turn_case)
        result = multiwoz_24_system.run()
        evaluation_result = turn_case.evaluate(result)
        loss = loss_fn.compute_loss(prediction=result, evaluation_result=evaluation_result, type="joint_goal_accuracy")
        save_gbc_tree(loss, os.path.join(WORKING_DIR, f"dialogue_{dialogue_case.dialogue_idx}", f"dialogue_{dialogue_case.dialogue_idx}-turn_{turn_idx}-joint_goal_accuracy.pkl"))
    evaluation_result = Multiwoz24Environment.evaluate_test_cases(mode="train", dialogue_indices=dialogue_case.dialogue_idx)
    loss = loss_fn.compute_loss(prediction=result, evaluation_result=evaluation_result, type="inform_success")
    save_gbc_tree(loss, os.path.join(WORKING_DIR, f"dialogue_{dialogue_case.dialogue_idx}", f"dialogue_{dialogue_case.dialogue_idx}-inform_success.pkl"))

    if dialogue_idx >= 0:
        break
