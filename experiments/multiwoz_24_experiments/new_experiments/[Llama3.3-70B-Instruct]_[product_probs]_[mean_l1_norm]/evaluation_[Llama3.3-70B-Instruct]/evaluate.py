import itertools
import json
import os
import re
import sys
from copy import deepcopy
from typing import List

import dotenv

from agentchord import (
    BaseAgent,
    BaseAgentSystem,
    ModelConfig,
    Multiwoz24Environment,
    MultiWOZ24MetaData,
    ParallelBlock,
)

# evaluation configuration
CHECKPOINT_DIR = "experiments/multiwoz_24_experiments/new_experiments/[Llama3.3-70B-Instruct]_[product_probs]_[mean_l1_norm]/training"
TARGET_DIR = "experiments/multiwoz_24_experiments/new_experiments/[Llama3.3-70B-Instruct]_[product_probs]_[mean_l1_norm]/evaluation_[Llama3.3-70B-Instruct]"
CLIENT_MODEL = "openai/Llama3.3-70B-Instruct"
LOG_NAME = "[MultiWOZ]_[Llama3.3-70B-Instruct]_[Eval]_[product_probs]_[mean_l1_norm].log"

# Configuration for the model
config = ModelConfig(
    client_model=CLIENT_MODEL,
    base_url=dotenv.get_key(dotenv.find_dotenv(), "Proxy_API_BASE"),
    api_key=dotenv.get_key(dotenv.find_dotenv(), "Proxy_API_KEY"),
    temperature=0.0,
    max_completion_tokens=1024,
    enable_thinking=False,
)


global_dialogue_state = dict()

class Multiwoz24System(BaseAgentSystem):
    def __init__(self, system_name: str, environment: Multiwoz24Environment, maximum_loops: int = 5, log_name: str = ""):
        super().__init__(
            system_name=system_name,
            environment=environment,
            maximum_loops=maximum_loops,
            log_name=log_name
        )

        ManagerAgent = BaseAgent(
            system_name="manager_agent",
            environment=environment,
            prompt="",
            tools=[],
            model_config=config,
            maximum_loops=maximum_loops,
            log_name=log_name
        )
        WorkerAgent_RESTAURANT = BaseAgent(
            system_name="restaurant_worker_agent",
            environment=environment,
            prompt="",
            tools=["query_restaurant", "book_restaurant"],
            model_config=config,
            maximum_loops=maximum_loops,
            log_name=log_name
        )
        WorkerAgent_HOTEL = BaseAgent(
            system_name="hotel_worker_agent",
            environment=environment,
            prompt="",
            tools=["query_hotel", "book_hotel"],
            model_config=config,
            maximum_loops=maximum_loops,
            log_name=log_name
        )
        WorkerAgent_ATTRACTION = BaseAgent(
            system_name="attraction_worker_agent",
            environment=environment,
            prompt="",
            tools=["query_attraction"],
            model_config=config,
            maximum_loops=maximum_loops,
            log_name=log_name
        )
        WorkerAgent_TAXI = BaseAgent(
            system_name="taxi_worker_agent",
            environment=environment,
            prompt="",
            tools=["book_taxi"],
            model_config=config,
            maximum_loops=maximum_loops,
            log_name=log_name
        )
        WorkerAgent_TRAIN = BaseAgent(
            system_name="train_worker_agent",
            environment=environment,
            prompt="",
            tools=["query_train", "book_train"],
            model_config=config,
            maximum_loops=maximum_loops,
            log_name=log_name
        )
        WorkerAgents = ParallelBlock(
            system_name="worker_agents",
            environment=environment,
            subsystems=[
                WorkerAgent_RESTAURANT,
                WorkerAgent_HOTEL,
                WorkerAgent_ATTRACTION,
                WorkerAgent_TAXI,
                WorkerAgent_TRAIN
            ],
            maximum_loops=maximum_loops,
            log_name=log_name
        )
        RespondorAgent = BaseAgent(
            system_name="respondor_agent",
            environment=environment,
            prompt="",
            tools=[],
            model_config=config,
            maximum_loops=maximum_loops,
            log_name=log_name
        )
        self.add_subsystem(ManagerAgent)
        self.add_subsystem(WorkerAgents)
        self.add_subsystem(RespondorAgent)

        self.add_on_completion_action("manager_agent", "manage", self._manage)
        self.add_on_completion_action("worker_agents", "work", self._work)

    def on_initialization(self) -> MultiWOZ24MetaData:
        metadata = self.environment.get_initial_metadata()
        grounding_utterance = metadata.grounding_utterance
        metadata.input = "Dialogue History:\n{grounding_utterance}"
        return metadata
    
    def _manage(self, metadata: MultiWOZ24MetaData) -> MultiWOZ24MetaData:
        """
        Finalize the metadata after the manager agent has generated the plan.
        """
        metadata.input = [
            f"Dialogue History:\n{metadata.grounding_utterance}",
            f"manager_agent:\n{metadata.output}",
        ]
        return metadata
    
    def _work(self, metadata: MultiWOZ24MetaData) -> MultiWOZ24MetaData:
        """
        Finalize the metadata after the worker agents have generated their outputs.
        Update the dialogue state in the metadata based on the tool usage.
        """
        metadata.input = [
            f"Dialogue History:\n{metadata.grounding_utterance}",
            f"restaurant_worker_agent:\n{metadata.output[0]}",
            f"hotel_worker_agent:\n{metadata.output[1]}",
            f"attraction_worker_agent:\n{metadata.output[2]}",
            f"taxi_worker_agent:\n{metadata.output[3]}",
            f"train_worker_agent:\n{metadata.output[4]}",
        ]

        tool_usage = metadata.tool
        valid_tools = [
            "query_restaurant",
            "query_hotel",
            "query_attraction",
            "query_train",
            "book_restaurant",
            "book_hotel",
            "book_taxi",
            "book_train"
        ]
        attraction_slots = ["attraction-name", "attraction-type", "attraction-area"]
        hotel_slots = ["hotel-name", "hotel-type", "hotel-parking", "hotel-area", "hotel-bookday", "hotel-bookstay", "hotel-internet", "hotel-bookpeople", "hotel-stars", "hotel-pricerange"]
        restaurant_slots = ["restaurant-name", "restaurant-food", "restaurant-area", "restaurant-bookday", "restaurant-booktime", "restaurant-bookpeople", "restaurant-pricerange"]
        taxi_slots = ["taxi-arriveby", "taxi-departure", "taxi-leaveat", "taxi-destination"]
        train_slots = ["train-arriveby", "train-day", "train-leaveat", "train-destination", "train-departure", "train-bookpeople"]
        valid_slots = list(itertools.chain.from_iterable([attraction_slots, hotel_slots, restaurant_slots, taxi_slots, train_slots]))
        dialogue_state = dict()
        for tool_record in tool_usage:
            if tool_record["tool_name"] in valid_tools:
                 for slot, value in tool_record["tool_arguments"].items():
                    domain = tool_record["tool_name"].split("_")[1]
                    if f"{domain}-{slot}".lower() in valid_slots:
                        dialogue_state[f"{domain}-{slot}"] = value
                    elif f"{domain}-book{slot}".lower() in valid_slots:
                        dialogue_state[f"{domain}-book{slot}"] = value
                    else:
                        print(f"Warning: Slot {slot} in tool {tool_record['tool_name']} is not a valid slot.")
        
        global global_dialogue_state
        global_dialogue_state.update(dialogue_state)
        metadata.dialogue_state = deepcopy(global_dialogue_state)

        return metadata
    
    def on_finalization(self, metadata: MultiWOZ24MetaData) -> MultiWOZ24MetaData:
        """
        Finalize the metadata after the response agent has generated the response.
        """
        metadata.system_response = metadata.output
        return metadata

multiwoz_24_system = Multiwoz24System("multiwoz_24_system", Multiwoz24Environment(), log_name=LOG_NAME)
print(multiwoz_24_system.get_pipeline_description())

def convert_sets(obj):
    if isinstance(obj, set):
        return list(obj)
    raise TypeError(f"Type {type(obj)} not serializable")

total_record = dict()
for checkpoint_idx in range(1, 11):
    multiwoz_24_system.load_agents(file_name=os.path.join(CHECKPOINT_DIR, f"checkpoint-{checkpoint_idx}/multiwoz_24_agents.json"))
    dialogue_idx_pool = list()
    for dialogue_idx, dialogue_case in enumerate(Multiwoz24Environment.iterate_test_cases(mode="test", random_seed=42)):
        global_dialogue_state = dict()
        dialogue_idx_pool.append(dialogue_case.dialogue_idx)
        for turn_idx, turn_case in enumerate(dialogue_case.iterate_dialog_turns()):
            print(f"Dialogue: {dialogue_idx} {dialogue_case.dialogue_idx},\tTurn: {turn_idx}")
            multiwoz_24_system.set_environment(environment=turn_case)
            result = multiwoz_24_system.run()
            evaluation_result = turn_case.evaluate(result)
            print(f"System Response: {evaluation_result.system_response}")
            print(f"Delixicalized System Response: {evaluation_result.delixicalized_system_response}")
            print(f"Dialogue State: {evaluation_result.dialogue_state}")
            print(f"Groundtruth Dialogue State: {evaluation_result.groundtruth_dialogue_state}")
            print(f"Joint Goal Accuracy: {evaluation_result.joint_goal_accuracy}")
            print(f"Joint Goal Accuracy Detail: {evaluation_result.joint_goal_accuracy_detail}")
            print(f"Inform detail: {evaluation_result.inform_detail}")
            print(f"Success detail: {evaluation_result.success_detail}")
            total_record[f"dialogue_{dialogue_case.dialogue_idx}_turn_{turn_idx}"] = {
                "system_response": evaluation_result.system_response,
                "delixicalized_system_response": evaluation_result.delixicalized_system_response,
                "dialogue_state": evaluation_result.dialogue_state,
                "groundtruth_dialogue_state": evaluation_result.groundtruth_dialogue_state,
                "joint_goal_accuracy": evaluation_result.joint_goal_accuracy,
                "joint_goal_accuracy_detail": evaluation_result.joint_goal_accuracy_detail,
                "inform_detail": evaluation_result.inform_detail,
                "success_detail": evaluation_result.success_detail
            }
        if dialogue_idx >= 99:  # Limit to 100 dialogues for testing
            break
    evaluation_result = Multiwoz24Environment.evaluate_test_cases(mode="test")
    evaluation_result.to_json(file_name=os.path.join(TARGET_DIR, f"checkpoint-{checkpoint_idx}_evaluation_result.json"))
    with open(os.path.join(TARGET_DIR, f"checkpoint-{checkpoint_idx}_total_record.json"), "w", encoding="utf-8") as f:
        json.dump(total_record, f, ensure_ascii=False, indent=2, default=convert_sets)