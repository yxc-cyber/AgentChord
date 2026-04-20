import itertools
import json
import os
import re
from copy import deepcopy
from typing import List

import dotenv
import torch
from transformers import BitsAndBytesConfig

from agentchord import (
    BaseAgentSystem,
    GBCAgent,
    Input,
    ModelConfig,
    ParallelBlock,
    TaubenchEnvironment,
    TaubenchMetaData,
)
from agentchord.gbc_object import GBC, GBCBase, visualize_gbc_tree
from agentchord.loss import TauBenchLoss
from agentchord.optimizer import OPROOptimizer
from agentchord.utils import WandBConfig

# training configuration
WORKING_DIR = "experiments/taubench_experiments/[Llama3.3-70B-Instruct]_[product_probs]_[max_product_input]/training"
LOG_NAME = "[TauBench]_[Llama3.3-70B-Instruct]_[product_probs]_[max_product_input].log"
OPTIMIZER_LOG_NAME = "[TauBench]_[Llama3.3-70B-Instruct]_[Optimizer]_[product_probs]_[max_product_input].log"
GRADIENT_STRATEGY = "product_probs"
CONNECTION_STRATEGY = "max_product_input"
MODEL = "Llama3.3-70B-Instruct"
USER_CLIENT_MODEL = "openai/gpt-4o-mini"
UPDATE_STEP = 1
TOTAL_TRAINING_SAMPLES = 10
# training resume configuration
RESUME_DIR = None
# RESUME_DIR = "experiments/taubench_experiments/[Llama3.3-70B-Instruct]_[product_probs]_[max_product_input]/training/checkpoint-1"
SKIP_SAMPLES = 0
# model loading configuration
LOCAL_MODEL = "LlamaModel"
MODEL_PATH = "/work/hdd/bghs/xyang7/models/Llama-3.3-70B-Instruct"
CHAT_TEMPLATE_PATH = "src/agentchord/model/chat_templates/tool_chat_template_llama3.3_json.jinja"

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
    chat_template_path=CHAT_TEMPLATE_PATH,
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
        ManagerAgent = GBCAgent(
            system_name="manager_agent",
            environment=environment,
            prompt=prompt_manager,
            tools=[],
            model_config=config,
            maximum_loops=4,
            log_name=log_name,
            dummy_weights=True
        )
        WorkerAgent_USER_RESOLUTION = GBCAgent(
            system_name="worker_agent(user_resolution)",
            environment=environment,
            prompt=prompt_user_resolution_agent,
            tools=["find_user_id_by_email", "find_user_id_by_name_zip"],
            model_config=config,
            maximum_loops=4,
            log_name=log_name
        )
        WorkerAgent_RETRIEVAL = GBCAgent(
            system_name="worker_agent(retrieval)",
            environment=environment,
            prompt=prompt_retrieval_agent,
            tools=["get_user_details", "get_order_details", "get_product_details", "list_all_product_types"],
            model_config=config,
            maximum_loops=4,
            log_name=log_name
        )
        WorkerAgent_POST_DELIVERY = GBCAgent(
            system_name="worker_agent(post_delivery)",
            environment=environment,
            prompt=prompt_post_delivery_agent,
            tools=["return_delivered_order_items", "exchange_delivered_order_items"],
            model_config=config,
            maximum_loops=4,
            log_name=log_name
        )
        WorkerAgent_ORDER_MODIFICATION = GBCAgent(
            system_name="worker_agent(order_modification)",
            environment=environment,
            prompt=prompt_order_modification_agent,
            tools=["modify_pending_order_address", "modify_pending_order_items", "modify_pending_order_payment", "cancel_pending_order"],
            model_config=config,
            maximum_loops=4,
            log_name=log_name
        )
        WorkerAgent_USER_PROFILE = GBCAgent(
            system_name="worker_agent(user_profile)",
            environment=environment,
            prompt=prompt_user_profile_agent,
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
        ResponderAgent = GBCAgent(
            system_name="responder_agent",
            environment=environment,
            prompt=prompt_responder_agent,
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
        self.dialogue_history_string = GBC(dialogue_history, subject=Input())
        metadata.input = GBC(
            f"Dialogue History:\n{self.dialogue_history_string}",
            connections=[self.dialogue_history_string],
            weights=[1.0]
        )
        return metadata

    def _manage(self, metadata: TaubenchMetaData) -> TaubenchMetaData:
        """
        Finalize the metadata after the manager agent has generated the plan.
        """
        dialogue_history = self.dialogue_history_string
        metadata.input = [
            GBC(f"Dialogue History:\n{dialogue_history}", connections=[self.dialogue_history_string], weights=[1.0]),
            GBC(f"manager_agent:\n{metadata.output}", connections=metadata.output.get_connections(), weights=metadata.output.get_weights())
        ]
        return metadata
    
    def _resolve_user(self, metadata: TaubenchMetaData) -> TaubenchMetaData:
        """
        Finalize the metadata after the user resolution worker agent has generated the output.
        """
        dialogue_history = self.dialogue_history_string
        metadata.input = [
            GBC(f"Dialogue History:\n{dialogue_history}", connections=[self.dialogue_history_string], weights=[1.0]),
            GBC(f"worker_agent(user_resolution):\n{metadata.output}", connections=metadata.output.get_connections(), weights=metadata.output.get_weights())
        ]
        return metadata
    
    def _retrieve(self, metadata: TaubenchMetaData) -> TaubenchMetaData:
        """
        Finalize the metadata after the retrieval worker agent has generated the output.
        """
        dialogue_history = self.dialogue_history_string
        metadata.input = [
            GBC(f"Dialogue History:\n{dialogue_history}", connections=[self.dialogue_history_string], weights=[1.0]),
            GBC(f"worker_agent(retrieval):\n{metadata.output}", connections=metadata.output.get_connections(), weights=metadata.output.get_weights())
        ]
        return metadata
    
    def _work(self, metadata: TaubenchMetaData) -> TaubenchMetaData:
        """
        Finalize the metadata after the post-delivery/order-modification/user-profile worker agents have generated the output.
        """
        dialogue_history = self.dialogue_history_string
        metadata.input = [
            GBC(f"Dialogue History:\n{dialogue_history}", connections=[self.dialogue_history_string], weights=[1.0]),
            GBC(f"worker_agent(post_delivery):\n{metadata.output[0]}", connections=metadata.output[0].get_connections(), weights=metadata.output[0].get_weights()),
            GBC(f"worker_agent(order_modification):\n{metadata.output[1]}", connections=metadata.output[1].get_connections(), weights=metadata.output[1].get_weights()),
            GBC(f"worker_agent(user_profile):\n{metadata.output[2]}", connections=metadata.output[2].get_connections(), weights=metadata.output[2].get_weights())
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
        self.dialogue_history_string = GBC(dialogue_history, subject=Input())
        metadata.input = f"Dialogue History:\n{self.dialogue_history_string}"

        return metadata


taubench_retail_system = TaubenchRetailSystem("taubench_retail_system", TaubenchEnvironment(), log_name=LOG_NAME)
print(taubench_retail_system.get_pipeline_description())

optimizer = OPROOptimizer(
    agents=taubench_retail_system.get_agents(),
    model_config=ModelConfig(
        client_model="openai/gpt-4.1-2025-04-14",
    ),
    log_name=OPTIMIZER_LOG_NAME,
    wandb_config=WandBConfig(
        project="AgentChord",
        config={
            "dataset": "TauBench-Retail",
            "system": "TaubenchRetailSystem",
            "gradient_strategy": GRADIENT_STRATEGY,
            "connection_strategy": CONNECTION_STRATEGY,
            "model": MODEL,
            "optimizer": "OPROOptimizer",
            "optimizer_model": "gpt-4.1-2025-04-14",
        }
    )
)
loss_fn = TauBenchLoss()
optimization_steps = 0
dialogue_idx_pool = list()
missed_cases = 0


if not os.path.exists(os.path.join(WORKING_DIR, "checkpoint-0")):
    os.makedirs(os.path.join(WORKING_DIR, "checkpoint-0"))
taubench_retail_system.save_agents(file_name=os.path.join(WORKING_DIR, "checkpoint-0/taubench_retail_system_agents.json"))
optimizer.save_optimizer_state(file_path=os.path.join(WORKING_DIR, "checkpoint-0/optimizer_state.json"))
if RESUME_DIR is not None:
    taubench_retail_system.load_agents(file_name=os.path.join(RESUME_DIR, "taubench_retail_system_agents.json"))
    optimizer.load_optimizer_state(file_path=os.path.join(RESUME_DIR, "optimizer_state.json"))
for dialogue_idx, dialogue_case in enumerate(TaubenchEnvironment.iterate_test_cases(domain="retail", task_split="train", random_seed=42, user_model_config=user_config, user_log_name=LOG_NAME)):
    print(f"{dialogue_idx} Dialogue: {dialogue_case.task_idx}")
    if dialogue_idx < SKIP_SAMPLES:
        continue
    dialogue_idx_pool.append(dialogue_case.task_idx)
    taubench_retail_system.set_environment(environment=dialogue_case)
    try:
        result = taubench_retail_system.run(loop=True)
    except Exception as e:
        print(f"Error occurred during the execution of dialogue case {dialogue_case.task_idx}: {e}")
        missed_cases += 1
        continue
    evaluation_result = dialogue_case.evaluate(result)
    loss = loss_fn.compute_loss(evaluation_result=evaluation_result)
    loss.backward(bandwidth=1)
    if (dialogue_idx + 1 - missed_cases) % UPDATE_STEP == 0:
        evaluation_result = TaubenchEnvironment.evaluate_test_cases(domain="retail", task_split="train", task_indices=dialogue_idx_pool)
        optimizer.step(
            performance=f"Reward: {evaluation_result.mean_reward}",
            performance_dict={
                "reward": evaluation_result.mean_reward
            }
        )
        optimization_steps = (dialogue_idx + 1 - missed_cases) // UPDATE_STEP

        if not os.path.exists(os.path.join(WORKING_DIR, f"checkpoint-{optimization_steps}")):
            os.makedirs(os.path.join(WORKING_DIR, f"checkpoint-{optimization_steps}"))
        taubench_retail_system.save_agents(file_name=os.path.join(WORKING_DIR, f"checkpoint-{optimization_steps}/taubench_retail_system_agents.json"))
        optimizer.save_optimizer_state(file_path=os.path.join(WORKING_DIR, f"checkpoint-{optimization_steps}/optimizer_state.json"))
        dialogue_idx_pool = list()
    
    if dialogue_idx - missed_cases >= TOTAL_TRAINING_SAMPLES - 1:
        break

optimizer.finish_wandb()
visualize_gbc_tree(loss, "experiments/taubench_experiments/[Llama3.3-70B-Instruct]_[product_probs]_[max_product_input]/training/example.png")