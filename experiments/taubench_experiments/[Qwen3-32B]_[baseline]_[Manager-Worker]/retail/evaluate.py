import json
import os
import pickle

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
CLIENT_MODEL = "openai/Qwen3-32B"
USER_CLIENT_MODEL = "openai/gpt-4o-mini"
LOG_NAME = "taubench_evaluation_example.log"

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
You will not be acutally taking any actions, but your reasoning and analysis will be used by the worker agents to decide which tools to use.
There are multiple worker agents repsonsible for different domains, including worker_agent(user_resolution), worker_agent(retrieval), worker_agent(post_delivery), worker_agent(order_modification), and worker_agent(user_profile).
The dataflow of the multi-agent system is: manager_agent -> worker_agent(user_resolution) -> worker_agent(retrieval) -> worker_agent(post_delivery/order_modification/user_profile) -> responder_agent.
The outputs of the worker agents will be used by a responder agent to generate the final response to the user.
So you should assign tasks to the worker agents based on the dialogue history and clarify what information the responsible worker agents should provide in their final output so that the respondor agent can generate a complete response to the user.
""".strip()

prompt_user_resolution_agent = """
You are a helpful agent in a multi-agent system, playing the role of a user resolution worker agent. Your task is to resolve the user identity based on the dialogue history and the instruction from the manager agent, and provide the user id to the retrieval worker agent for information retrieval.
If the user identity (user_id) has been clearly identified in the dialogue history, you can directly provide the user id. Otherwise, you can use the tools available to you to find the user id via email or via name + zip code. 
You should provide necessary information user identity for the retrieval worker agent to retrieve the correct information for the user.
""".strip()

prompt_retrieval_agent = """
You are a helpful agent in a multi-agent system, playing the role of a retrieval worker agent. Your task is to retrieve necessary information based on the dialogue history and the instruction from the manager agent, and provide the retrieved information to the post-delivery/order-modification/user-profile worker agents for order modification or return/exchange, or provide the retrieved information to the responder agent for response generation.
If there is no need to retrieve any information, you don't have to use any tools and just provide the user identity information from the user resolution worker agent.
Otherwise, you can use the tools available to you to retrieve user details, order details, product details, or list all product types. Then you should provide the retrieved information plus the user identity information from the user resolution worker agent to the post-delivery/order-modification/user-profile worker agents for order modification or return/exchange, or provide the retrieved information to the responder agent for response generation.
""".strip()

prompt_post_delivery_agent = """
You are a helpful agent in a multi-agent system, playing the role of a post-delivery worker agent. Your task is to handle user requests after the order has been delivered, based on the dialogue history and the instruction from the manager agent.
If the user has requests related to return or exchange of delivered orders, you can use the tools available to you to return or exchange delivered order items. Then you should provide the return or exchange details to the responder agent for response generation.
If the user has no requests related to return or exchange of delivered orders, you can just explain that the user has no requests related to return or exchange of delivered orders.
""".strip()

prompt_order_modification_agent = """
You are a helpful agent in a multi-agent system, playing the role of an order modification worker agent. Your task is to handle user requests for modification or cancellation of pending orders, based on the dialogue history and the instruction from the manager agent.
If the user has requests related to modification or cancellation of pending orders, you can use the tools available to you to modify or cancel pending orders. Then you should provide the modification or cancellation details to the responder agent for response generation.
If the user has no requests related to modification or cancellation of pending orders, you can just explain that the user has no requests related to modification or cancellation of pending orders.
""".strip()

prompt_user_profile_agent = """
You are a helpful agent in a multi-agent system, playing the role of a user profile worker agent. Your task is to handle user requests for modification of user profile information, based on the dialogue history and the instruction from the manager agent.
If the user has requests related to modification of user profile information, you can use the tools available to you to modify user profile information such as user address. Then you should provide the modification details to the responder agent for response generation.
If the user has no requests related to modification of user profile information, you can just explain that the user has no requests related to modification of user profile information.
""".strip()

prompt_responder_agent = """
You are a helpful agent in a multi-agent system, playing the role of a responder agent. Your task is to generate a complete and helpful response to the user based on the dialogue history, the instruction from the manager agent, and the outputs from the worker agents.
You should first analyze the dialogue history and the instruction from the manager agent, then review the outputs from the worker agents, and finally generate a complete response to the user that addresses the user's needs and requests.
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
            maximum_loops=3,
            log_name=log_name
        )
        WorkerAgent_USER_RESOLUTION = BaseAgent(
            system_name="worker_agent(user_resolution)",
            environment=environment,
            prompt=prompt_user_resolution_agent,
            tools=["find_user_id_by_email", "find_user_id_by_name_zip"],
            model_config=config,
            maximum_loops=3,
            log_name=log_name
        )
        WorkerAgent_RETRIEVAL = BaseAgent(
            system_name="worker_agent(retrieval)",
            environment=environment,
            prompt=prompt_retrieval_agent,
            tools=["get_user_details", "get_order_details", "get_product_details", "list_all_product_types"],
            model_config=config,
            maximum_loops=3,
            log_name=log_name
        )
        WorkerAgent_POST_DELIVERY = BaseAgent(
            system_name="worker_agent(post_delivery)",
            environment=environment,
            prompt=prompt_post_delivery_agent,
            tools=["return_delivered_order_items", "exchange_delivered_order_items"],
            model_config=config,
            maximum_loops=3,
            log_name=log_name
        )
        WorkerAgent_ORDER_MODIFICATION = BaseAgent(
            system_name="worker_agent(order_modification)",
            environment=environment,
            prompt=prompt_order_modification_agent,
            tools=["modify_pending_order_address", "modify_pending_order_items", "modify_pending_order_payment", "cancel_pending_order"],
            model_config=config,
            maximum_loops=3,
            log_name=log_name
        )
        WorkerAgent_USER_PROFILE = BaseAgent(
            system_name="worker_agent(user_profile)",
            environment=environment,
            prompt=prompt_user_profile_agent,
            tools=["modify_user_address"],
            model_config=config,
            maximum_loops=3,
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
            maximum_loops=3,
            log_name=log_name
        )
        ResponderAgent = BaseAgent(
            system_name="responder_agent",
            environment=environment,
            prompt=prompt_responder_agent,
            tools=[],
            model_config=config,
            maximum_loops=3,
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
        self.dialogue_history_string = dialogue_history
        metadata.input = f"Dialogue History:\n{self.dialogue_history_string}"
        return metadata

    def _manage(self, metadata: TaubenchMetaData) -> TaubenchMetaData:
        """
        Finalize the metadata after the manager agent has generated the plan.
        """
        dialogue_history = self.dialogue_history_string
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
        metadata.input = [
            f"Dialogue History:\n{dialogue_history}",
            f"worker_agent(user_resolution):\n{metadata.output}",
        ]
        return metadata
    
    def _retrieve(self, metadata: TaubenchMetaData) -> TaubenchMetaData:
        """
        Finalize the metadata after the retrieval worker agent has generated the output.
        """
        dialogue_history = self.dialogue_history_string
        metadata.input = [
            f"Dialogue History:\n{dialogue_history}",
            f"worker_agent(retrieval):\n{metadata.output}",
        ]
        return metadata
    
    def _work(self, metadata: TaubenchMetaData) -> TaubenchMetaData:
        """
        Finalize the metadata after the post-delivery/order-modification/user-profile worker agents have generated the output.
        """
        dialogue_history = self.dialogue_history_string
        metadata.input = [
            f"Dialogue History:\n{dialogue_history}",
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

        print(f"User Response: {metadata.user_responses[-1]}")
        print(f"System Response: {metadata.output}")

        # Initialize all the agents for the next turn
        self.subsystems["manager_agent"].messages_simplification(remove_roles=["user"], remove_tools=["terminate"], keep_last_n=8)
        self.subsystems["worker_agent(user_resolution)"].messages_simplification(remove_roles=["user"], remove_tools=["terminate"], keep_last_n=8)
        self.subsystems["worker_agent(retrieval)"].messages_simplification(remove_roles=["user"], remove_tools=["terminate"], keep_last_n=8)
        for worker_agent in self.subsystems["worker_agents"].subsystems.values():
            worker_agent.messages_simplification(remove_roles=["user"], remove_tools=["terminate"], keep_last_n=8)
        self.subsystems["responder_agent"].messages_simplification(remove_roles=["user"], remove_tools=["terminate"], keep_last_n=8)

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
    # if dialogue_idx == 99:
    #     break


evaluation_record = TaubenchEnvironment.evaluation_record
evaluation_result = TaubenchEnvironment.evaluate_test_cases(domain="retail", task_split="test")
evaluation_result.to_json(file_name=f"experiments/taubench_experiments/[Qwen3-32B]_[baseline]_[Manager-Worker]/retail/evaluation_result.json")
with open(f"experiments/taubench_experiments/[Qwen3-32B]_[baseline]_[Manager-Worker]/retail/total_record.json", "w", encoding="utf-8") as f:
    json.dump(total_record, f, ensure_ascii=False, indent=2, default=convert_sets)
with open(f"experiments/taubench_experiments/[Qwen3-32B]_[baseline]_[Manager-Worker]/retail/total_record.pkl", "wb") as f:
    pickle.dump(evaluation_record, f)