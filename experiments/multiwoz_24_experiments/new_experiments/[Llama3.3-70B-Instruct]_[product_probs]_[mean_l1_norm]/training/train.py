import itertools
import json
import os
import re
from copy import deepcopy
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
from agentchord.utils import WandBConfig

# training configuration
WORKING_DIR = "experiments/multiwoz_24_experiments/new_experiments/[Llama3.3-70B-Instruct]_[product_probs]_[mean_l1_norm]/training"
LOG_NAME = "[MultiWOZ]_[Llama3.3-70B-Instruct]_[product_probs]_[mean_l1_norm].log"
OPTIMIZER_LOG_NAME = "[MultiWOZ]_[Llama3.3-70B-Instruct]_[Optimizer]_[product_probs]_[mean_l1_norm].log"
GRADIENT_STRATEGY = "product_probs"
CONNECTION_STRATEGY = "mean_l1_norm"
MODEL = "Llama3.3-70B-Instruct"
# training resume configuration
# RESUME_DIR = None
RESUME_DIR = "experiments/multiwoz_24_experiments/new_experiments/[Llama3.3-70B-Instruct]_[product_probs]_[mean_l1_norm]/training/checkpoint-9"
SKIP_SAMPLES = 27
# model loading configuration
LOCAL_MODEL = "LlamaModel"
MODEL_PATH = "/work/hdd/bghs/xyang7/models/Llama-3.3-70B-Instruct"
CHAT_TEMPLATE_PATH = "src/agentchord/model/chat_templates/tool_chat_template_llama3.3_json.jinja"

prompt_manager = """
You are a helpful agent in a multi-agent system, playing the role of a manager agent. You can analyze the dialogue history, reason about the user's needs, and make plans accordingly.
You will not be acutally taking any actions, but your reasoning and analysis will be used by the worker agents to decide which tools to use.
There are multiple worker agents repsonsible for different domains, including restaurant, hotel, attraction, taxi, and train. Each worker agent is reponsible for using the tools in their domain to make queries or bookings for the user.
The outputs of the worker agents will be used by a respondor agent to generate the final response to the user.
So you should assign tasks to the worker agents based on the dialogue history and clarify what information the responsible worker agents should provide in their final output so that the respondor agent can generate a complete response to the user.

Example:
Dialogue History:
User: I want to book a table at an Italian restaurant in the center of town for 7 pm tonight.
Manager Agent:
After analyzing the dialogue history, it is clear that the user wants to book a table at an Italian restaurant in the center of town for 7 pm tonight. 
The responsible worker agent is the restaurant worker agent. 
The restaurant worker agent should use the query_restaurant tool to find Italian restaurants in the center of town that have availability at 7 pm tonight. 
Once the restaurant worker agent has found a suitable restaurant, it should use the book_restaurant tool to make the booking for the user. 
In the final output, the restaurant worker agent should provide the name of the restaurant, the booking time, and any other relevant details.
""".strip()

prompt_worker_restaurant = """
You are a helpful agent in a multi-agent system, playing the role of a worker agent of domain restaurant. You can properly use the tools to make queries or bookings for the user.
The previous agent is a manager agent that provides you with analysis of the dialogue history and the assignment of tasks.
Based on the dialogue history and the task assignment, you should decide whether to use the tools or not. You can use another tool after you have used one and received the tool result.
Finally, you should generate a summary of the actions you have taken and the results you have received.
Your final output will be given to a respondor agent to generate the final response to the user. So you should pay attention to the message from the manager agent to make sure you have completed the task and provide all the necessary information.
""".strip()

prompt_worker_hotel = """
You are a helpful agent in a multi-agent system, playing the role of a worker agent of domain hotel. You can properly use the tools to make queries or bookings for the user.
The previous agent is a manager agent that provides you with analysis of the dialogue history and the assignment of tasks.
Based on the dialogue history and the task assignment, you should decide whether to use the tools or not. You can use another tool after you have used one and received the tool result.
Finally, you should generate a summary of the actions you have taken and the results you have received.
Your final output will be given to a respondor agent to generate the final response to the user. So you should pay attention to the message from the manager agent to make sure you have completed the task and provide all the necessary information.
""".strip()

prompt_worker_attraction = """
You are a helpful agent in a multi-agent system, playing the role of a worker agent of domain attraction. You can properly use the tools to make queries or bookings for the user.
The previous agent is a manager agent that provides you with analysis of the dialogue history and the assignment of tasks.
Based on the dialogue history and the task assignment, you should decide whether to use the tools or not. You can use another tool after you have used one and received the tool result.
Finally, you should generate a summary of the actions you have taken and the results you have received.
Your final output will be given to a respondor agent to generate the final response to the user. So you should pay attention to the message from the manager agent to make sure you have completed the task and provide all the necessary information.
""".strip()

prompt_worker_taxi = """
You are a helpful agent in a multi-agent system, playing the role of a worker agent of domain taxi. You can properly use the tools to make queries or bookings for the user.
The previous agent is a manager agent that provides you with analysis of the dialogue history and the assignment of tasks.
Based on the dialogue history and the task assignment, you should decide whether to use the tools or not. You can use another tool after you have used one and received the tool result.
Finally, you should generate a summary of the actions you have taken and the results you have received.
Your final output will be given to a respondor agent to generate the final response to the user. So you should pay attention to the message from the manager agent to make sure you have completed the task and provide all the necessary information.
""".strip()

prompt_worker_train = """
You are a helpful agent in a multi-agent system, playing the role of a worker agent of domain train. You can properly use the tools to make queries or bookings for the user.
The previous agent is a manager agent that provides you with analysis of the dialogue history and the assignment of tasks.
Based on the dialogue history and the task assignment, you should decide whether to use the tools or not. You can use another tool after you have used one and received the tool result.
Finally, you should generate a summary of the actions you have taken and the results you have received.
Your final output will be given to a respondor agent to generate the final response to the user. So you should pay attention to the message from the manager agent to make sure you have completed the task and provide all the necessary information.
""".strip()

prompt_respondor = """
You are a helpful agent in a multi-agent system, playing the role of a respondor agent. You can generate a complete and coherent response to the user based on the dialogue history and the outputs of the worker agents.
The previous agents are worker agents that have used various tools to make queries or bookings for the user after receiving task assignments from a manager agent.
Based on the dialogue history and the outputs of the worker agents, you should generate a final response to the user that addresses their needs and provides all the necessary information.

Example:
Dialogue History:
User: I want to book a table at an Italian restaurant in the center of town for 7 pm tonight.
Worker Agents Outputs:
restaurant_worker_agent:
I have used the query_restaurant tool to find Italian restaurants in the center of town that have availability at 7 pm tonight. I found "Luigi's Italian Bistro" which has a table available at that time. I have used the book_restaurant tool to make the booking for the user. The booking is confirmed for "Luigi's Italian Bistro" at 7 pm tonight.
Hotel Worker Agent:
No action taken.
Attraction Worker Agent:
No action taken.
Taxi Worker Agent:
No action taken.
Train Worker Agent:
No action taken.
Respondor Agent:
You have successfully booked a table at "Luigi's Italian Bistro", an Italian restaurant located in the center of town, for 7 pm tonight. Enjoy your meal!
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
    max_new_tokens=64,
    # temperature=0.0,
    do_sample=False,
    gradient_strategy=GRADIENT_STRATEGY,
    connection_strategy=CONNECTION_STRATEGY,
    chat_template_path=CHAT_TEMPLATE_PATH,
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

        ManagerAgent = GBCAgent(
            system_name="manager_agent",
            environment=environment,
            tools=[],
            prompt=prompt_manager,
            model_config=config,
            maximum_loops=maximum_loops,
            log_name=log_name
        )
        WorkerAgent_RESTAURANT = GBCAgent(
            system_name="restaurant_worker_agent",
            environment=environment,
            prompt=prompt_worker_restaurant,
            tools=["query_restaurant", "book_restaurant"],
            model_config=config,
            maximum_loops=maximum_loops,
            log_name=log_name
        )
        WorkerAgent_HOTEL = GBCAgent(
            system_name="hotel_worker_agent",
            environment=environment,
            prompt=prompt_worker_hotel,
            tools=["query_hotel", "book_hotel"],
            model_config=config,
            maximum_loops=maximum_loops,
            log_name=log_name
        )
        WorkerAgent_ATTRACTION = GBCAgent(
            system_name="attraction_worker_agent",
            environment=environment,
            prompt=prompt_worker_attraction,
            tools=["query_attraction"],
            model_config=config,
            maximum_loops=maximum_loops,
            log_name=log_name
        )
        WorkerAgent_TAXI = GBCAgent(
            system_name="taxi_worker_agent",
            environment=environment,
            prompt=prompt_worker_taxi,
            tools=["book_taxi"],
            model_config=config,
            maximum_loops=maximum_loops,
            log_name=log_name
        )
        WorkerAgent_TRAIN = GBCAgent(
            system_name="train_worker_agent",
            environment=environment,
            prompt=prompt_worker_train,
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
        RespondorAgent = GBCAgent(
            system_name="respondor_agent",
            environment=environment,
            tools=[],
            prompt=prompt_respondor,
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
        metadata.input = GBC(
            f"Dialogue History:\n{grounding_utterance}",
            connections=[grounding_utterance],
            weights=[1.0]
        )
        return metadata
    
    def _manage(self, metadata: MultiWOZ24MetaData) -> MultiWOZ24MetaData:
        """
        Finalize the metadata after the manager agent has generated the plan.
        """
        metadata.input = [
            GBC(
                f"Dialogue History:\n{metadata.grounding_utterance}",
                connections=[metadata.grounding_utterance],
                weights=[1.0]
            ),
            GBC(
                f"manager_agent:\n{metadata.output}",
                connections=metadata.output.get_connections(),
                weights=metadata.output.get_weights()
            )
        ]
        return metadata
    
    def _work(self, metadata: MultiWOZ24MetaData) -> MultiWOZ24MetaData:
        """
        Finalize the metadata after the worker agents have generated their outputs.
        Update the dialogue state in the metadata based on the tool usage.
        """
        metadata.input = [
            GBC(
                f"Dialogue History:\n{metadata.grounding_utterance}",
                connections=[metadata.grounding_utterance],
                weights=[1.0]
            ),
            GBC(
                f"restaurant_worker_agent:\n{metadata.output[0]}",
                connections=metadata.output[0].get_connections(),
                weights=metadata.output[0].get_weights()
            ),
            GBC(
                f"hotel_worker_agent:\n{metadata.output[1]}",
                connections=metadata.output[1].get_connections(),
                weights=metadata.output[1].get_weights()
            ),
            GBC(
                f"attraction_worker_agent:\n{metadata.output[2]}",
                connections=metadata.output[2].get_connections(),
                weights=metadata.output[2].get_weights()
            ),
            GBC(
                f"taxi_worker_agent:\n{metadata.output[3]}",
                connections=metadata.output[3].get_connections(),
                weights=metadata.output[3].get_weights()
            ),
            GBC(
                f"train_worker_agent:\n{metadata.output[4]}",
                connections=metadata.output[4].get_connections(),
                weights=metadata.output[4].get_weights()
            )
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
        metadata.dialogue_state = GBC(
            deepcopy(global_dialogue_state),
            connections=metadata.output,
            weights=[1.0] * len(metadata.output)
        )

        return metadata
    
    def on_finalization(self, metadata: MultiWOZ24MetaData) -> MultiWOZ24MetaData:
        """
        Finalize the metadata after the response agent has generated the response.
        """
        metadata.system_response = metadata.output
        return metadata


multiwoz_24_system = Multiwoz24System("multiwoz_24_system", Multiwoz24Environment(), log_name=LOG_NAME)
print(multiwoz_24_system.get_pipeline_description())

optimizer = OPROOptimizer(
    agents=multiwoz_24_system.get_agents(),
    model_config=ModelConfig(
        client_model="openai/gpt-4.1-2025-04-14",
    ),
    log_name=OPTIMIZER_LOG_NAME,
    wandb_config=WandBConfig(
        project="AgentChord",
        config={
            "dataset": "MultiWOZ-24",
            "system": "Multiwoz24System",
            "gradient_strategy": GRADIENT_STRATEGY,
            "connection_strategy": CONNECTION_STRATEGY,
            "model": MODEL,
            "optimizer": "OPROOptimizer",
            "optimizer_model": "gpt-4.1-2025-04-14",
        }
    )
)
loss_fn = MultiWOZ24Loss()
optimization_steps = 0
dialogue_idx_pool = list()

def check_inference_trajectories(optimizer):
    for trajectory_idx in range(len(optimizer.optimization_info)-1, -1, -1):
        trajectory = optimizer.optimization_info[trajectory_idx]
        if "The system has made the following user intention predictions" in trajectory[-1][1]:
            text = trajectory[-1][1]
            fp_match = re.search(r"The false positive predictions are:\s*(\{.*?\})", text, re.DOTALL)
            fn_match = re.search(r"The false negative predictions are:\s*(\[.*?\])", text, re.DOTALL)
            false_positive = json.loads(fp_match.group(1)) if fp_match else dict()
            false_negative = json.loads(fn_match.group(1)) if fn_match else dict()
            print("False Positive:", false_positive)
            print("False Negative:", false_negative)
            erroneous_domains = set([slot.split("-")[0] for slot in false_negative]) | set([slot.split("-")[0] for slot in false_positive.keys()])
            contain_erroneous_domain = False
            for subject, _ in trajectory:
                if "_worker_agent" in str(subject):
                    domain = str(subject).split("_worker_agent")[0]
                    if domain in erroneous_domains:
                        contain_erroneous_domain = True
                        break
            if not contain_erroneous_domain:
                print("Removing trajectory:", trajectory)
                optimizer.optimization_info.pop(trajectory_idx)

if not os.path.exists(os.path.join(WORKING_DIR, "checkpoint-0")):
    os.makedirs(os.path.join(WORKING_DIR, "checkpoint-0"))
multiwoz_24_system.save_agents(file_name=os.path.join(WORKING_DIR, "checkpoint-0/multiwoz_24_agents.json"))
optimizer.save_optimizer_state(file_path=os.path.join(WORKING_DIR, "checkpoint-0/optimizer_state.json"))
if RESUME_DIR is not None:
    multiwoz_24_system.load_agents(file_name=os.path.join(RESUME_DIR, "multiwoz_24_agents.json"))
    optimizer.load_optimizer_state(file_path=os.path.join(RESUME_DIR, "optimizer_state.json"))
for dialogue_idx, dialogue_case in enumerate(Multiwoz24Environment.iterate_test_cases(mode="train", random_seed=42, max_turns=10)):
    print(f"{dialogue_idx} Dialogue: {dialogue_case.dialogue_idx}")
    global_dialogue_state = dict()
    if dialogue_idx < SKIP_SAMPLES:
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
    if (dialogue_idx + 1) % 3 == 0:
        evaluation_result = Multiwoz24Environment.evaluate_test_cases(mode="train", dialogue_indices=dialogue_idx_pool)
        check_inference_trajectories(optimizer=optimizer)
        optimizer.step(
            performance=f"Inform: {evaluation_result.inform['total']}; Success : {evaluation_result.success['total']}; Joint Goal Accuracy: {evaluation_result.joint_goal_accuracy}",
            performance_dict={
                "inform": evaluation_result.inform['total'],
                "success": evaluation_result.success['total'],
                "joint_goal_accuracy": evaluation_result.joint_goal_accuracy
            }
        )
        optimization_steps = (dialogue_idx + 1) // 3

        if not os.path.exists(os.path.join(WORKING_DIR, f"checkpoint-{optimization_steps}")):
            os.makedirs(os.path.join(WORKING_DIR, f"checkpoint-{optimization_steps}"))
        multiwoz_24_system.save_agents(file_name=os.path.join(WORKING_DIR, f"checkpoint-{optimization_steps}/multiwoz_24_agents.json"))
        optimizer.save_optimizer_state(file_path=os.path.join(WORKING_DIR, f"checkpoint-{optimization_steps}/optimizer_state.json"))
        dialogue_idx_pool = list()
    
    if dialogue_idx >= 29:
        break

optimizer.finish_wandb()