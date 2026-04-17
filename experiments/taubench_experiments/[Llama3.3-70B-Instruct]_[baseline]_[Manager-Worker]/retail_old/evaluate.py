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
CLIENT_MODEL = "openai/Llama3.3-70B-Instruct"
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

Example:
Dialogue History:
User: I want to return everything from my recent order except for a tablet. Can you help me with that?
Manager Agent:
Based on the dialogue history, the user has a request related to return of delivered orders.
However, the user identity has not been clearly identified in the dialogue history.
So I will first assign the task to worker_agent(user_resolution) to resolve the user identity and provide the user id to worker_agent(retrieval) for information retrieval.
The worker_agent(retrieval) should look for orders containing tablets and provide the order information including the order id, the item ids, corresponding item names, and payment_method_id.
Then I will assign the task to worker_agent(post_delivery) to return the delivered order items except for the tablet using the order id, the item ids, corresponding item names, and payment_method_id, and provide the return details to the responder agent for response generation.

Here are some basic rules about the retail domain:
## Domain basic
- At the beginning of the conversation, system has to authenticate the user identity by locating their user id via email, or via name + zip code.
- All times in the database are EST and 24 hour based. For example "02:30:00" means 2:30 AM EST.
- Each user has a profile of its email, default address, user id, and payment methods. Each payment method is either a gift card, a paypal account, or a credit card.
- Our retail store has 50 types of products. For each type of product, there are variant items of different options. For example, for a 't shirt' product, there could be an item with option 'color blue size M', and another item with option 'color red size L'.
- Each product has an unique product id, and each item has an unique item id. They have no relations and should not be confused.
- Each order can be in status 'pending', 'processed', 'delivered', or 'cancelled'. Generally, you can only take action on pending or delivered orders.
- Exchange or modify order tools can only be called once. Be sure that all items to be changed are collected into a list before making the tool call!!!
## Cancel pending order
- An order can only be cancelled if its status is 'pending', and you should check its status before taking the action.
- The user needs to confirm the order id and the reason (either 'no longer needed' or 'ordered by mistake') for cancellation.
- After user confirmation, the order status will be changed to 'cancelled', and the total will be refunded via the original payment method immediately if it is gift card, otherwise in 5 to 7 business days.
## Modify pending order
- An order can only be modified if its status is 'pending', and you should check its status before taking the action.
- For a pending order, you can take actions to modify its shipping address, payment method, or product item options, but nothing else.
### Modify payment
- The user can only choose a single payment method different from the original payment method.
- If the user wants the modify the payment method to gift card, it must have enough balance to cover the total amount.
- After user confirmation, the order status will be kept 'pending'. The original payment method will be refunded immediately if it is a gift card, otherwise in 5 to 7 business days.
### Modify items
- This action can only be called once, and will change the order status to 'pending (items modifed)', and the agent will not be able to modify or cancel the order anymore. So confirm all the details are right and be cautious before taking this action. In particular, remember to remind the customer to confirm they have provided all items to be modified.
- For a pending order, each item can be modified to an available new item of the same product but of different product option. There cannot be any change of product types, e.g. modify shirt to shoe.
- The user must provide a payment method to pay or receive refund of the price difference. If the user provides a gift card, it must have enough balance to cover the price difference.
## Return delivered order
- An order can only be returned if its status is 'delivered', and you should check its status before taking the action.
- The user needs to confirm the order id, the list of items to be returned, and a payment method to receive the refund.
- The refund must either go to the original payment method, or an existing gift card.
- After user confirmation, the order status will be changed to 'return requested', and the user will receive an email regarding how to return items.
## Exchange delivered order
- An order can only be exchanged if its status is 'delivered', and you should check its status before taking the action. In particular, remember to remind the customer to confirm they have provided all items to be exchanged.
- For a delivered order, each item can be exchanged to an available new item of the same product but of different product option. There cannot be any change of product types, e.g. modify shirt to shoe.
- The user must provide a payment method to pay or receive refund of the price difference. If the user provides a gift card, it must have enough balance to cover the price difference.
- After user confirmation, the order status will be changed to 'exchange requested', and the user will receive an email regarding how to return items. There is no need to place a new order.
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
Make sure you retrieve as much information as possible and provide the most relevant information for the post-delivery/order-modification/user-profile worker agents to perform the operations, or for the responder agent to generate a complete response to the user.
""".strip()

prompt_post_delivery_agent = """
You are a helpful agent in a multi-agent system, playing the role of a post-delivery worker agent. Your task is to handle user requests after the order has been delivered, based on the dialogue history and the instruction from the manager agent.
If the user has requests related to return or exchange of delivered orders, you can use the tools available to you to return or exchange delivered order items. Then you should provide the return or exchange details to the responder agent for response generation.
If the user has no requests related to return or exchange of delivered orders, or the manager agent did not assign any task related to you, you can just explain without performing any task.
If the user cannot provide clear instructions on which items to return or exchange, you can try performing the operations on each item until the operation succeeds.
""".strip()

prompt_order_modification_agent = """
You are a helpful agent in a multi-agent system, playing the role of an order modification worker agent. Your task is to handle user requests for modification or cancellation of pending orders, based on the dialogue history and the instruction from the manager agent.
If the user has requests related to modification or cancellation of pending orders, you can use the tools available to you to modify or cancel pending orders. Then you should provide the modification or cancellation details to the responder agent for response generation.
If the user has no requests related to modification or cancellation of pending orders, or the manager agent did not assign any task related to you, you can just explain without performing any task.
If the user cannot provide clear instructions on which items to return or exchange, you can try performing the operations on each item until the operation succeeds.
""".strip()

prompt_user_profile_agent = """
You are a helpful agent in a multi-agent system, playing the role of a user profile worker agent. Your task is to handle user requests for modification of user profile information, based on the dialogue history and the instruction from the manager agent.
If the user has requests related to modification of user profile information, you can use the tools available to you to modify user profile information such as user address. Then you should provide the modification details to the responder agent for response generation.
If the user has no requests related to modification of user profile information, or the manager agent did not assign any task related to you, you can just explain without performing any task.
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
            maximum_loops=5,
            log_name=log_name
        )
        WorkerAgent_USER_RESOLUTION = BaseAgent(
            system_name="worker_agent(user_resolution)",
            environment=environment,
            prompt=prompt_user_resolution_agent,
            tools=["find_user_id_by_email", "find_user_id_by_name_zip"],
            model_config=config,
            maximum_loops=5,
            log_name=log_name
        )
        WorkerAgent_RETRIEVAL = BaseAgent(
            system_name="worker_agent(retrieval)",
            environment=environment,
            prompt=prompt_retrieval_agent,
            tools=["get_user_details", "get_order_details", "get_product_details", "list_all_product_types"],
            model_config=config,
            maximum_loops=5,
            log_name=log_name
        )
        WorkerAgent_POST_DELIVERY = BaseAgent(
            system_name="worker_agent(post_delivery)",
            environment=environment,
            prompt=prompt_post_delivery_agent,
            tools=["return_delivered_order_items", "exchange_delivered_order_items"],
            model_config=config,
            maximum_loops=5,
            log_name=log_name
        )
        WorkerAgent_ORDER_MODIFICATION = BaseAgent(
            system_name="worker_agent(order_modification)",
            environment=environment,
            prompt=prompt_order_modification_agent,
            tools=["modify_pending_order_address", "modify_pending_order_items", "modify_pending_order_payment", "cancel_pending_order"],
            model_config=config,
            maximum_loops=5,
            log_name=log_name
        )
        WorkerAgent_USER_PROFILE = BaseAgent(
            system_name="worker_agent(user_profile)",
            environment=environment,
            prompt=prompt_user_profile_agent,
            tools=["modify_user_address"],
            model_config=config,
            maximum_loops=5,
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
            prompt=prompt_responder_agent,
            tools=[],
            model_config=config,
            maximum_loops=5,
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
evaluation_result.to_json(file_name=f"experiments/taubench_experiments/[Llama3.3-70B-Instruct]_[baseline]_[Manager-Worker]/retail/evaluation_result.json")
with open(f"experiments/taubench_experiments/[Llama3.3-70B-Instruct]_[baseline]_[Manager-Worker]/retail/total_record.json", "w", encoding="utf-8") as f:
    json.dump(total_record, f, ensure_ascii=False, indent=2, default=convert_sets)
with open(f"experiments/taubench_experiments/[Llama3.3-70B-Instruct]_[baseline]_[Manager-Worker]/retail/total_record.pkl", "wb") as f:
    pickle.dump(evaluation_record, f)