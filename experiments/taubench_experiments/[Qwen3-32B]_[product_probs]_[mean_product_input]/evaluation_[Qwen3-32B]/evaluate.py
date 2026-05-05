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
    ParallelBlock,
    TaubenchEnvironment,
    TaubenchMetaData,
)

# evaluation configuration
CHECKPOINT_DIR = "experiments/taubench_experiments/[Qwen3-32B]_[product_probs]_[mean_product_input]/training"
TARGET_DIR = "experiments/taubench_experiments/[Qwen3-32B]_[product_probs]_[mean_product_input]/evaluation_[Qwen3-32B]"
# CLIENT_MODEL = "openai/Qwen3-32B"
CLIENT_MODEL = "openrouter/qwen/qwen3-32b"
USER_CLIENT_MODEL = "openai/gpt-4o-mini"
LOG_NAME = "[TauBench]_[Qwen3-32B]_[Eval]_[product_probs]_[mean_product_input].log"

# Configuration for the model
config = ModelConfig(
    client_model=CLIENT_MODEL,
    # base_url=dotenv.get_key(dotenv.find_dotenv(), "Proxy_API_BASE"),
    # api_key=dotenv.get_key(dotenv.find_dotenv(), "Proxy_API_KEY"),
    temperature=0.0,
    max_completion_tokens=1024,
    enable_thinking=False,
)
user_config = ModelConfig(
    client_model=USER_CLIENT_MODEL,
    api_key=dotenv.get_key(dotenv.find_dotenv(), "OPENAI_API_KEY"),
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
            maximum_loops=4,
            log_name=log_name
        )
        WorkerAgent_USER_RESOLUTION = BaseAgent(
            system_name="worker_agent(user_resolution)",
            environment=environment,
            prompt="",
            tools=["find_user_id_by_email", "find_user_id_by_name_zip"],
            model_config=config,
            maximum_loops=4,
            log_name=log_name
        )
        WorkerAgent_RETRIEVAL = BaseAgent(
            system_name="worker_agent(retrieval)",
            environment=environment,
            prompt="",
            tools=["get_user_details", "get_order_details", "get_product_details", "list_all_product_types"],
            model_config=config,
            maximum_loops=4,
            log_name=log_name
        )
        WorkerAgent_POST_DELIVERY = BaseAgent(
            system_name="worker_agent(post_delivery)",
            environment=environment,
            prompt="",
            tools=["return_delivered_order_items", "exchange_delivered_order_items"],
            model_config=config,
            maximum_loops=4,
            log_name=log_name
        )
        WorkerAgent_ORDER_MODIFICATION = BaseAgent(
            system_name="worker_agent(order_modification)",
            environment=environment,
            prompt="",
            tools=["modify_pending_order_address", "modify_pending_order_items", "modify_pending_order_payment", "cancel_pending_order"],
            model_config=config,
            maximum_loops=4,
            log_name=log_name
        )
        WorkerAgent_USER_PROFILE = BaseAgent(
            system_name="worker_agent(user_profile)",
            environment=environment,
            prompt=BaseAgent,
            tools=["modify_user_address"],
            model_config=config,
            maximum_loops=4,
            log_name=log_name
        )
        WorkerAgents = ParallelBlock(
            system_name="worker_agents",
            environment=environment,
            subsystems=[
                WorkerAgent_POST_DELIVERY,
                WorkerAgent_ORDER_MODIFICATION,
                WorkerAgent_USER_PROFILE
            ],
            log_name=log_name
        )
        ResponderAgent = BaseAgent(
            system_name="responder_agent",
            environment=environment,
            prompt="",
            tools=[],
            model_config=config,
            maximum_loops=4,
            log_name=log_name
        )
        self.add_subsystem(ManagerAgent)
        self.add_subsystem(WorkerAgent_USER_RESOLUTION)
        self.add_subsystem(WorkerAgent_RETRIEVAL)
        self.add_subsystem(WorkerAgents)
        self.add_subsystem(ResponderAgent)

        self.add_on_completion_action("manager_agent", "manage", self._manage)
        self.add_on_completion_action("worker_agent(user_resolution)", "resolve_user", self._resolve_user)
        self.add_on_completion_action("worker_agent(retrieval)", "retrieve", self._retrieve)
        self.add_on_completion_action("worker_agents", "work", self._work)
        self.add_on_completion_action("responder_agent", "respond", self._respond)

        self.dialogue_history_string = ""
        self.manager_string = ""

    def on_initialization(self) -> TaubenchMetaData:
        metadata = self.environment.get_initial_metadata()
        dialogue_history_list = list()
        system_responses_num = len(metadata.responses)
        assert system_responses_num == len(metadata.user_responses) - 1, \
            f"The number of system responses should be equal to the number of user responses minus one. Currently, system responses: {system_responses_num}, user responses: {len(metadata.user_responses)}."
        for idx in range(system_responses_num):
            user_response = metadata.user_responses[idx]
            system_response = metadata.responses[idx]
            dialogue_history_list.append(f"User Response:\n{user_response}")
            dialogue_history_list.append(f"System Response:\n{system_response}")
        dialogue_history_list.append(f"User Response:\n{metadata.user_responses[-1]}")
        dialogue_history = "\n".join(dialogue_history_list)
        self.dialogue_history_string = dialogue_history
        metadata.input = f"Dialogue History:\n{self.dialogue_history_string}"
        return metadata

    def _manage(self, metadata: TaubenchMetaData) -> TaubenchMetaData:
        """
        Finalize the metadata after the manager agent has generated the plan.
        """
        dialogue_history = self.dialogue_history_string
        self.manager_string = metadata.output
        metadata.input = [
            f"Dialogue History:\n{dialogue_history}",
            f"manager_agent:\n{metadata.output}",
        ]
        return metadata
    
    def _resolve_user(self, metadata: TaubenchMetaData) -> TaubenchMetaData:
        """
        Finalize the metadata after the user resolution worker agent has generated the output.
        """
        dialogue_history = self.dialogue_history_string
        manager_string = self.manager_string
        metadata.input = [
            f"Dialogue History:\n{dialogue_history}",
            f"manager_agent:\n{manager_string}",
            f"worker_agent(user_resolution):\n{metadata.output}",
        ]
        return metadata
    
    def _retrieve(self, metadata: TaubenchMetaData) -> TaubenchMetaData:
        """
        Finalize the metadata after the retrieval worker agent has generated the output.
        """
        dialogue_history = self.dialogue_history_string
        manager_string = self.manager_string
        metadata.input = [
            f"Dialogue History:\n{dialogue_history}",
            f"manager_agent:\n{manager_string}",
            f"worker_agent(retrieval):\n{metadata.output}",
        ]
        return metadata
    
    def _work(self, metadata: TaubenchMetaData) -> TaubenchMetaData:
        """
        Finalize the metadata after the post-delivery/order-modification/user-profile worker agents have generated the output.
        """
        dialogue_history = self.dialogue_history_string
        manager_string = self.manager_string
        metadata.input = [
            f"Dialogue History:\n{dialogue_history}",
            f"manager_agent:\n{manager_string}",
            f"worker_agent(post_delivery):\n{metadata.output[0]}",
            f"worker_agent(order_modification):\n{metadata.output[1]}",
            f"worker_agent(user_profile):\n{metadata.output[2]}",
        ]
        return metadata

    def _respond(self, metadata: TaubenchMetaData) -> TaubenchMetaData:
        """
        Finalize the metadata after the responder agent has generated the response.
        """
        metadata.responses.append(metadata.output)
        metadata.user_responses.append(self.environment.user_response(metadata.output))

        print(f"System Response: {metadata.output}")
        print(f"User Response: {metadata.user_responses[-1]}")

        # Initialize all the agents for the next turn
        self.subsystems["manager_agent"].messages_simplification(remove_roles=["user"], remove_tools=["terminate"], keep_last_n=6)
        self.subsystems["worker_agent(user_resolution)"].messages_simplification(remove_roles=["user"], remove_tools=["terminate"], keep_last_n=6)
        self.subsystems["worker_agent(retrieval)"].messages_simplification(remove_roles=["user"], remove_tools=["terminate"], keep_last_n=6)
        for worker_agent in self.subsystems["worker_agents"].subsystems.values():
            worker_agent.messages_simplification(remove_roles=["user"], remove_tools=["terminate"], keep_last_n=6)
        self.subsystems["responder_agent"].messages_simplification(remove_roles=["user"], remove_tools=["terminate"], keep_last_n=6)

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
        self.dialogue_history_string = dialogue_history
        metadata.input = f"Dialogue History:\n{self.dialogue_history_string}"

        return metadata

taubench_retail_system = TaubenchRetailSystem("taubench_retail_system", TaubenchEnvironment(), log_name=LOG_NAME)
print(taubench_retail_system.get_pipeline_description())

def convert_sets(obj):
    if isinstance(obj, set):
        return list(obj)
    raise TypeError(f"Type {type(obj)} not serializable")

total_record = dict()
for checkpoint_idx in range(1, 2):
    taubench_retail_system.load_agents(file_name=os.path.join(CHECKPOINT_DIR, f"checkpoint-{checkpoint_idx}/taubench_retail_system_agents.json"))
    dialogue_idx_pool = list()
    for dialogue_idx, dialogue_case in enumerate(TaubenchEnvironment.iterate_test_cases(domain="retail", task_split="test", random_seed=42, user_model_config=user_config, user_log_name=LOG_NAME)):
        dialogue_idx_pool.append(dialogue_case.task_idx)
        taubench_retail_system.set_environment(environment=dialogue_case)
        result = taubench_retail_system.run(loop=True)
        evaluation_result = dialogue_case.evaluate(result)
        print(f"Reward detail: {evaluation_result.reward_details}")
        total_record[f"dialogue_{dialogue_case.task_idx}"] = {
            "system_response": evaluation_result.responses,
            "user_response": evaluation_result.user_responses,
            "instruction": evaluation_result.instruction,
            "reward": evaluation_result.reward,
            "reward_details": evaluation_result.reward_details,
        }
    evaluation_result = TaubenchEnvironment.evaluate_test_cases(domain="retail", task_split="test", task_indices=dialogue_idx_pool)
    evaluation_result.to_json(file_name=os.path.join(TARGET_DIR, f"checkpoint-{checkpoint_idx}_evaluation_result.json"))
    with open(os.path.join(TARGET_DIR, f"checkpoint-{checkpoint_idx}_total_record.json"), "w", encoding="utf-8") as f:
        json.dump(total_record, f, ensure_ascii=False, indent=2, default=convert_sets)