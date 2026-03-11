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
    TaubenchEnvironment,
    TaubenchMetaData,
    ParallelBlock,
)

# evaluation configuration
CLIENT_MODEL = "openai/gpt-4o-mini"
LOG_NAME = "taubench_evaluation_example.log"
TARGET_DIR = "examples/taubench_examples"

# Configuration for the models
config = ModelConfig(
    client_model=CLIENT_MODEL,
    temperature=0.0,
)
user_config = ModelConfig(
    client_model=CLIENT_MODEL,
    temperature=0.0,
)


class TaubenchRetailSystem(BaseAgentSystem):
    def __init__(self, system_name: str, environment: TaubenchEnvironment, maximum_loops: int = 10, log_name: str = ""):
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
        WorkerAgent_INFO = BaseAgent(
            system_name="worker_agent(info)",
            environment=environment,
            prompt="",
            tools=["list_all_product_types"],
            model_config=config,
            maximum_loops=maximum_loops,
            log_name=log_name
        )
        WorkerAgent_FIND_USER = BaseAgent(
            system_name="worker_agent(find_user)",
            environment=environment,
            prompt="",
            tools=["find_user_id_by_email", "find_user_id_by_name_zip"],
            model_config=config,
            maximum_loops=maximum_loops,
            log_name=log_name
        )
        WorkerAgent_GET_DETAILS = BaseAgent(
            system_name="worker_agent(get_details)",
            environment=environment,
            prompt="",
            tools=["get_order_details", "get_product_details", "get_user_details"],
            model_config=config,
            maximum_loops=maximum_loops,
            log_name=log_name
        )
        WorkerAgent_MODIFY_ORDER = BaseAgent(
            system_name="worker_agent(modify_order)",
            environment=environment,
            prompt="",
            tools=["modify_pending_order_address", "modify_pending_order_items", "modify_pending_order_payment"],
            model_config=config,
            maximum_loops=maximum_loops,
            log_name=log_name
        )
        WorkerAgent_CANCEL_ORDER = BaseAgent(
            system_name="worker_agent(cancel_order)",
            environment=environment,
            prompt="",
            tools=["cancel_pending_order"],
            model_config=config,
            maximum_loops=maximum_loops,
            log_name=log_name
        )
        WorkerAgent_RETURN_ORDER = BaseAgent(
            system_name="worker_agent(return_order)",
            environment=environment,
            prompt="",
            tools=["return_delivered_order_items", "exchange_delivered_order_items"],
            model_config=config,
            maximum_loops=maximum_loops,
            log_name=log_name
        )
        WorkerAgent_MODIFY_USER = BaseAgent(
            system_name="worker_agent(modify_user)",
            environment=environment,
            prompt="",
            tools=["modify_user_address"],
            model_config=config,
            maximum_loops=maximum_loops,
            log_name=log_name
        )
        WorkerAgents = ParallelBlock(
            system_name="worker_agents",
            environment=environment,
            subsystems=[
                WorkerAgent_FIND_USER,
                WorkerAgent_GET_DETAILS,
                WorkerAgent_MODIFY_ORDER,
                WorkerAgent_CANCEL_ORDER,
                WorkerAgent_RETURN_ORDER,
                WorkerAgent_MODIFY_USER
            ],
            maximum_loops=maximum_loops,
            log_name=log_name
        )
        ResponderAgent = BaseAgent(
            system_name="responder_agent",
            environment=environment,
            prompt="",
            tools=["transfer_to_human_agents"],
            model_config=config,
            maximum_loops=maximum_loops,
            log_name=log_name
        )
        self.add_subsystem(ManagerAgent)
        self.add_subsystem(WorkerAgent_INFO)
        self.add_subsystem(WorkerAgents)
        self.add_subsystem(ResponderAgent)

        self.add_on_completion_action("manager_agent", "manage", self._manage)
        self.add_on_completion_action("worker_agents", "work", self._work)

    def on_initialization(self) -> TaubenchMetaData:
        metadata = self.environment.get_initial_metadata()
        dialogue_history_list = list()
        system_responses_num = len(metadata.responses)
        assert system_responses_num == len(metadata.user_responses) - 1, "The number of system responses should be equal to the number of user responses minus one."
        for idx in range(system_responses_num):
            user_response = metadata.user_responses[idx]
            system_response = metadata.responses[idx]
            dialogue_history_list.append(f"User Response:\n{user_response}")
            dialogue_history_list.append(f"System Response:\n{system_response}")
        dialogue_history_list.append(f"User Response:\n{metadata.user_responses[-1]}")
        dialogue_history = "\n".join(dialogue_history_list)
        metadata.input = f"Dialogue History:\n{dialogue_history}"
        return metadata
    
    def _manage(self, metadata: TaubenchMetaData) -> TaubenchMetaData:
        """
        Finalize the metadata after the manager agent has generated the plan.
        """
        dialogue_history_list = list()
        system_responses_num = len(metadata.responses)
        assert system_responses_num == len(metadata.user_responses) - 1, "The number of system responses should be equal to the number of user responses minus one."
        for idx in range(system_responses_num):
            user_response = metadata.user_responses[idx]
            system_response = metadata.responses[idx]
            dialogue_history_list.append(f"User Response:\n{user_response}")
            dialogue_history_list.append(f"System Response:\n{system_response}")
        dialogue_history_list.append(f"User Response:\n{metadata.user_responses[-1]}")
        dialogue_history = "\n".join(dialogue_history_list)
        
        metadata.input = [
            f"Dialogue History:\n{dialogue_history}",
            f"manager_agent:\n{metadata.output}",
        ]
        return metadata
    
    def _work(self, metadata: TaubenchMetaData) -> TaubenchMetaData:
        """
        Finalize the metadata after the worker agents have generated their outputs.
        Update the dialogue state in the metadata based on the tool usage.
        """
        dialogue_history_list = list()
        system_responses_num = len(metadata.responses)
        assert system_responses_num == len(metadata.user_responses) - 1, "The number of system responses should be equal to the number of user responses minus one."
        for idx in range(system_responses_num):
            user_response = metadata.user_responses[idx]
            system_response = metadata.responses[idx]
            dialogue_history_list.append(f"User Response:\n{user_response}")
            dialogue_history_list.append(f"System Response:\n{system_response}")
        dialogue_history_list.append(f"User Response:\n{metadata.user_responses[-1]}")
        dialogue_history = "\n".join(dialogue_history_list)

        metadata.input = [
            f"Dialogue History:\n{dialogue_history}",
            f"worker_agent(find_user):\n{metadata.output[0]}",
            f"worker_agent(get_details):\n{metadata.output[1]}",
            f"worker_agent(modify_order):\n{metadata.output[2]}",
            f"worker_agent(cancel_order):\n{metadata.output[3]}",
            f"worker_agent(return_order):\n{metadata.output[4]}",
            f"worker_agent(modify_user):\n{metadata.output[5]}",
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
    
    def on_finalization(self, metadata: TaubenchMetaData) -> TaubenchMetaData:
        """
        Finalize the metadata after the response agent has generated the response.
        """
        metadata.system_response = metadata.output
        return metadata

def convert_sets(obj):
    if isinstance(obj, set):
        return list(obj)
    raise TypeError(f"Type {type(obj)} not serializable")


taubench_retail_system = TaubenchRetailSystem("taubench_retail_system", TaubenchEnvironment(), log_name=LOG_NAME)
print(taubench_retail_system.get_pipeline_description())


total_record = dict()
for dialogue_idx, dialogue_case in enumerate(TaubenchEnvironment.iterate_test_cases(domain="retail", task_split="test", random_seed=42)):
    print(f"Dialogue: {dialogue_idx} {dialogue_case.task_idx}")
    taubench_retail_system.set_environment(environment=dialogue_case)
    result = taubench_retail_system.run(loop=True)
    evaluation_result = dialogue_case.evaluate(result)
    print(f"Dialogue State: {evaluation_result.instruction}")
    system_responses_num = len(evaluation_result.responses)
    for idx in range(system_responses_num):
        user_response = evaluation_result.user_responses[idx]
        system_response = evaluation_result.responses[idx]
        print(f"User Response: {user_response}")
        print(f"System Response: {system_response}")
    print(f"Reward: {evaluation_result.reward}")
    print(f"Reward Details: {evaluation_result.reward_details}")
    total_record[f"dialogue_{dialogue_case.task_idx}"] = {
        "system_responses": evaluation_result.responses,
        "user_responses": evaluation_result.user_responses,
        "instruction": evaluation_result.instruction,
        "reward": evaluation_result.reward,
        "reward_details": evaluation_result.reward_details,
    }
    break

evaluation_result.to_json(file_name=os.path.join(TARGET_DIR, f"evaluation_result.json"))
with open(os.path.join(TARGET_DIR, f"total_record.json"), "w", encoding="utf-8") as f:
    json.dump(total_record, f, ensure_ascii=False, indent=2, default=convert_sets)