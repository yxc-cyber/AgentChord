import itertools
import json
import os
import re
import sys
from copy import deepcopy
from typing import List
import time

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
CLIENT_MODEL = "openai/Qwen3-32B-FP8"
USER_CLIENT_MODEL = "openai/gpt-4o-mini"
LOG_NAME = "taubench_evaluation_example.log"
TARGET_DIR = "examples/taubench_examples"

# Configuration for the models
config = ModelConfig(
    client_model=CLIENT_MODEL,
    base_url=dotenv.get_key(dotenv.find_dotenv(), "Proxy_API_BASE"),
    api_key=dotenv.get_key(dotenv.find_dotenv(), "Proxy_API_KEY"),
    temperature=0.0,
    enable_thinking=False,
)
user_config = ModelConfig(
    client_model=USER_CLIENT_MODEL,
    api_key=dotenv.get_key(dotenv.find_dotenv(), "OPENAI_API_KEY"),
    temperature=0.0,
)

prompt_manager = """
You are a helpful agent in a multi-agent system, playing the role of a manager agent. You can analyze the dialogue history, reason about the user's needs, and make plans accordingly.
You will not be acutally taking any actions, but your reasoning and analysis will be used the worker agents to decide which tools to use.
There are multiple worker agents repsonsible for different domains, including info agent, find_user agent, get_details agent, modify_order agent, cancel_order agent, return_order agent, and modify_user agent. You can delegate tasks to these worker agents by specifying which agent to use.
Info agent can use the tool "list_all_product_types" to list all product types.
Find_user agent can use the tools "find_user_id_by_email" and "find_user_id_by_name_zip" to find the user ID based on the user's email or the combination of user's name and zip code.
Get_details agent can use the tools "get_order_details", "get_product_details", and "get_user_details" to get the details of the order, product, and user.
Modify_order agent can use the tools "modify_pending_order_address", "modify_pending_order_items", and "modify_pending_order_payment" to modify the pending order's address, items, and payment method.
Cancel_order agent can use the tool "cancel_pending_order" to cancel the pending order.
Return_order agent can use the tools "return_delivered_order_items" and "exchange_delivered_order_items" to return or exchange the delivered order items.
Modify_user agent can use the tool "modify_user_address" to modify the user's address.
After you, info agent goes first, then the find_user agent, get_details agent, modify_order agent, cancel_order agent, return_order agent, and modify_user agent will work in parallel to help the user.
In the end, the responder agent will generate the final response to the user based on the outputs from all worker agents.
Therefore, your job is to analyze the dialogue history and make plans for the worker agents to help the user. You can specify which agent to use and what information to provide to them. You should also consider the order of the worker agents, since info agent goes first, then the find_user agent, get_details agent, modify_order agent, cancel_order agent, return_order agent, and modify_user agent will work in parallel.
Your output will not directly go to the user. Your are not talking to the user.
Your instructions to the worker agents will be vital for them.
""".strip()

prompt_info_agent = """
You are an info agent in a multi-agent system. Your task is to offer information about the products to other worker agents based on the manager agent's instruction.
Your output will not directly go to the user, but will be used by other worker agents.
Your output will be given to other worker agents including find_user agent, get_details agent, modify_order agent, cancel_order agent, return_order agent, and modify_user agent.
You should decide whether to use the tool or not.
No matter whether you use the tool or not, you should relay the information you have from the manager agent to other worker agents.
""".strip()

prompt_find_user_agent = """
You are a find_user agent in a multi-agent system. Your task is to find the user ID based on the manager agent's instruction. You can find the user ID based on the user's email or the combination of user's name and zip code.
You are in parallel with other worker agents including info agent, get_details agent, modify_order agent, cancel_order agent, return_order agent, and modify_user agent.
Your output will not directly go to the user, but will be given to the responder agent to generate the final response to the user.
You should decide whether to use the tools or not. You can use another tool after you have used one and received the tool result.
You should generate a summary of the actions you have taken and the results you have received if you choose tools as the output, so that the responder agent can understand your actions and results.
""".strip()

prompt_get_details_agent = """
You are a get_details agent in a multi-agent system. Your task is to get the details of the order, product, and user based on the manager agent's instruction. You can get the details of the order, product, and user.
You are in parallel with other worker agents including info agent, find_user agent, modify_order agent, cancel_order agent, return_order agent, and modify_user agent.
Your output will not directly go to the user, but will be given to the responder agent to generate the final response to the user.
You should decide whether to use the tools or not. You can use another tool after you have used one and received the tool result.
You should generate a summary of the actions you have taken and the results you have received if you choose tools as the output, so that the responder agent can understand your actions and results.
""".strip()

prompt_modify_order_agent = """
You are a modify_order agent in a multi-agent system. Your task is to modify the pending order's address, items, and payment method based on the manager agent's instruction. You can modify the pending order's address, items, and payment method.
You are in parallel with other worker agents including info agent, find_user agent, get_details agent, cancel_order agent, return_order agent, and modify_user agent.
Your output will not directly go to the user, but will be given to the responder agent to generate the final response to the user.
You should decide whether to use the tools or not. You can use another tool after you have used one and received the tool result.
You should generate a summary of the actions you have taken and the results you have received if you choose tools as the output, so that the responder agent can understand your actions and results.
""".strip()

prompt_cancel_order_agent = """
You are a cancel_order agent in a multi-agent system. Your task is to cancel the pending order based on the manager agent's instruction. You can cancel the pending order.
You are in parallel with other worker agents including info agent, find_user agent, get_details agent, modify_order agent, return_order agent, and modify_user agent.
Your output will not directly go to the user, but will be given to the responder agent to generate the final response to the user.
You should decide whether to use the tools or not. You can use another tool after you have used one and received the tool result.
You should generate a summary of the actions you have taken and the results you have received if you choose tools as the output, so that the responder agent can understand your actions and results.
""".strip()

prompt_return_order_agent = """
You are a return_order agent in a multi-agent system. Your task is to return or exchange the delivered order items based on the manager agent's instruction. You can return or exchange the delivered order items.
You are in parallel with other worker agents including info agent, find_user agent, get_details agent, modify_order agent, cancel_order agent, and modify_user agent.
Your output will not directly go to the user, but will be given to the responder agent to generate the final response to the user.
You should decide whether to use the tools or not. You can use another tool after you have used one and received the tool result.
You should generate a summary of the actions you have taken and the results you have received if you choose tools as the output, so that the responder agent can understand your actions and results.
""".strip()

prompt_modify_user_agent = """
You are a modify_user agent in a multi-agent system. Your task is to modify the user's address based on the manager agent's instruction. You can modify the user's address.
You are in parallel with other worker agents including info agent, find_user agent, get_details agent, modify_order agent, cancel_order agent, and return_order agent.
Your output will not directly go to the user, but will be given to the responder agent to generate the final response to the user.
You should decide whether to use the tools or not. You can use another tool after you have used one and received the tool result.
You should generate a summary of the actions you have taken and the results you have received if you choose tools as the output, so that the responder agent can understand your actions and results.
""".strip()

prompt_responder_agent = """
You are a responder agent in a multi-agent system. Your task is to generate the final response to the user based on the manager agent's instruction and the outputs from all worker agents.
""".strip()

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
            prompt=prompt_manager,
            tools=[],
            model_config=config,
            maximum_loops=maximum_loops,
            log_name=log_name
        )
        WorkerAgent_INFO = BaseAgent(
            system_name="worker_agent(info)",
            environment=environment,
            prompt=prompt_info_agent,
            tools=["list_all_product_types"],
            model_config=config,
            maximum_loops=maximum_loops,
            log_name=log_name
        )
        WorkerAgent_FIND_USER = BaseAgent(
            system_name="worker_agent(find_user)",
            environment=environment,
            prompt=prompt_find_user_agent,
            tools=["find_user_id_by_email", "find_user_id_by_name_zip"],
            model_config=config,
            maximum_loops=maximum_loops,
            log_name=log_name
        )
        WorkerAgent_GET_DETAILS = BaseAgent(
            system_name="worker_agent(get_details)",
            environment=environment,
            prompt=prompt_get_details_agent,
            tools=["get_order_details", "get_product_details", "get_user_details"],
            model_config=config,
            maximum_loops=maximum_loops,
            log_name=log_name
        )
        WorkerAgent_MODIFY_ORDER = BaseAgent(
            system_name="worker_agent(modify_order)",
            environment=environment,
            prompt=prompt_modify_order_agent,
            tools=["modify_pending_order_address", "modify_pending_order_items", "modify_pending_order_payment"],
            model_config=config,
            maximum_loops=maximum_loops,
            log_name=log_name
        )
        WorkerAgent_CANCEL_ORDER = BaseAgent(
            system_name="worker_agent(cancel_order)",
            environment=environment,
            prompt=prompt_cancel_order_agent,
            tools=["cancel_pending_order"],
            model_config=config,
            maximum_loops=maximum_loops,
            log_name=log_name
        )
        WorkerAgent_RETURN_ORDER = BaseAgent(
            system_name="worker_agent(return_order)",
            environment=environment,
            prompt=prompt_return_order_agent,
            tools=["return_delivered_order_items", "exchange_delivered_order_items"],
            model_config=config,
            maximum_loops=maximum_loops,
            log_name=log_name
        )
        WorkerAgent_MODIFY_USER = BaseAgent(
            system_name="worker_agent(modify_user)",
            environment=environment,
            prompt=prompt_modify_user_agent,
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
            prompt=prompt_responder_agent,
            tools=[],
            model_config=config,
            maximum_loops=maximum_loops,
            log_name=log_name
        )
        self.add_subsystem(ManagerAgent)
        self.add_subsystem(WorkerAgent_INFO)
        self.add_subsystem(WorkerAgents)
        self.add_subsystem(ResponderAgent)

        self.add_on_completion_action("manager_agent", "manage", self._manage)
        self.add_on_completion_action("worker_agent(info)", "info", self._info)
        self.add_on_completion_action("worker_agents", "work", self._work)
        self.add_on_completion_action("responder_agent", "respond", self._respond)

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

    def _info(self, metadata: TaubenchMetaData) -> TaubenchMetaData:
        """
        Finalize the metadata after the info agent has generated the output.
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
            f"info_agent:\n{metadata.output}",
        ]
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
        return metadata
    
    def _respond(self, metadata: TaubenchMetaData) -> TaubenchMetaData:
        """
        Finalize the metadata after the responder agent has generated the response.
        """
        metadata.responses.append(metadata.output)
        metadata.user_responses.append(self.environment.user_response(metadata.output))

        print(f"System Response: {metadata.output}")
        print(f"User Response: {metadata.user_responses[-1]}")
        time.sleep(5)

        # Initialize all the agents for the next turn
        self.subsystems["manager_agent"].messages_initialization()
        self.subsystems["worker_agent(info)"].messages_initialization()
        for worker_agent in self.subsystems["worker_agents"].subsystems.values():
            worker_agent.messages_initialization()
        self.subsystems["responder_agent"].messages_initialization()

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


def convert_sets(obj):
    if isinstance(obj, set):
        return list(obj)
    raise TypeError(f"Type {type(obj)} not serializable")


taubench_retail_system = TaubenchRetailSystem("taubench_retail_system", TaubenchEnvironment(), log_name=LOG_NAME)
print(taubench_retail_system.get_pipeline_description())


total_record = dict()
for dialogue_idx, dialogue_case in enumerate(TaubenchEnvironment.iterate_test_cases(domain="retail", task_split="test", random_seed=42, user_model_config=user_config, user_log_name=LOG_NAME)):
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