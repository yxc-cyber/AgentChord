import itertools
import json
import re
from copy import deepcopy

from agentchord import (
    BaseAgent,
    BaseAgentSystem,
    ModelConfig,
    Multiwoz24Environment,
    MultiWOZ24MetaData,
    ParallelBlock,
)

global_dialogue_state = dict()

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
            tools=[],
            prompt=prompt_manager,
            model_config=ModelConfig(client_model="openai/Qwen3-32B-FP8", temperature=0.0),
            maximum_loops=maximum_loops,
            log_name=log_name
        )
        WorkerAgent_RESTAURANT = BaseAgent(
            system_name="restaurant_worker_agent",
            environment=environment,
            prompt=prompt_worker_restaurant,
            tools=["query_restaurant", "book_restaurant"],
            model_config=ModelConfig(client_model="openai/Qwen3-32B-FP8", temperature=0.0),
            maximum_loops=maximum_loops,
            log_name=log_name
        )
        WorkerAgent_HOTEL = BaseAgent(
            system_name="hotel_worker_agent",
            environment=environment,
            prompt=prompt_worker_hotel,
            tools=["query_hotel", "book_hotel"],
            model_config=ModelConfig(client_model="openai/Qwen3-32B-FP8", temperature=0.0),
            maximum_loops=maximum_loops,
            log_name=log_name
        )
        WorkerAgent_ATTRACTION = BaseAgent(
            system_name="attraction_worker_agent",
            environment=environment,
            prompt=prompt_worker_attraction,
            tools=["query_attraction"],
            model_config=ModelConfig(client_model="openai/Qwen3-32B-FP8", temperature=0.0),
            maximum_loops=maximum_loops,
            log_name=log_name
        )
        WorkerAgent_TAXI = BaseAgent(
            system_name="taxi_worker_agent",
            environment=environment,
            prompt=prompt_worker_taxi,
            tools=["book_taxi"],
            model_config=ModelConfig(client_model="openai/Qwen3-32B-FP8", temperature=0.0),
            maximum_loops=maximum_loops,
            log_name=log_name
        )
        WorkerAgent_TRAIN = BaseAgent(
            system_name="train_worker_agent",
            environment=environment,
            prompt=prompt_worker_train,
            tools=["query_train", "book_train"],
            model_config=ModelConfig(client_model="openai/Qwen3-32B-FP8", temperature=0.0),
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
            tools=[],
            prompt=prompt_respondor,
            model_config=ModelConfig(client_model="openai/Qwen3-32B-FP8", temperature=0.0),
            maximum_loops=maximum_loops,
            log_name=log_name
        )
        self.add_subsystem(ManagerAgent)
        self.add_subsystem(WorkerAgents)
        self.add_subsystem(RespondorAgent)

        self.add_on_completion_action("manager_agent", "manage", self._manage)
        self.add_on_completion_action("worker_agents", "work", self._work)
        self.add_on_completion_action("respondor_agent", "respond", self._respond)

    def on_initialization(self) -> MultiWOZ24MetaData:
        metadata = self.environment.get_initial_metadata()
        grounding_utterance = metadata.grounding_utterance
        metadata.input = f"Dialogue History:\n{grounding_utterance}"
        return metadata
    
    def _manage(self, metadata: MultiWOZ24MetaData) -> MultiWOZ24MetaData:
        """
        Finalize the metadata after the manager agent has generated the plan.
        """
        metadata.input = [
            f"Dialogue History:\n{metadata.grounding_utterance}",
            f"manager_agent:\n{metadata.output}"
        ]
        return metadata
    
    def _work(self, metadata: MultiWOZ24MetaData) -> MultiWOZ24MetaData:
        """
        Finalize the metadata after the worker agents have generated their outputs.
        """
        metadata.input = [
            f"Dialogue History:\n{metadata.grounding_utterance}",
            f"restaurant_worker_agent:\n{metadata.output[0]}",
            f"hotel_worker_agent:\n{metadata.output[1]}",
            f"attraction_worker_agent:\n{metadata.output[2]}",
            f"taxi_worker_agent:\n{metadata.output[3]}",
            f"train_worker_agent:\n{metadata.output[4]}",
        ]
        return metadata

    def _respond(self, metadata: MultiWOZ24MetaData) -> MultiWOZ24MetaData:
        """
        Update the dialogue state in the metadata based on the tool usage.
        """
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
    
multiwoz_24_system = Multiwoz24System("multiwoz_24_system", Multiwoz24Environment(), log_name="multiwoz_experiment_baseline_manager_worker_qwen3.log")
print(multiwoz_24_system.get_pipeline_description())

def convert_sets(obj):
    if isinstance(obj, set):
        return list(obj)
    raise TypeError(f"Type {type(obj)} not serializable")

total_record = dict()
for dialogue_idx, dialogue_case in enumerate(Multiwoz24Environment.iterate_test_cases(mode="test", random_seed=42)):
    global_dialogue_state = dict()
    for turn_idx, turn_case in enumerate(dialogue_case.iterate_dialog_turns()):
        print(f"Dialogue: {dialogue_idx} {dialogue_case.dialogue_idx},\tTurn: {turn_idx}")
        multiwoz_24_system.set_environment(environment=turn_case)
        result = multiwoz_24_system.run()
        evaluation_result = turn_case.evaluate(result)
        print(f"Dialogue State: {evaluation_result.dialogue_state}")
        print(f"Groundtruth Dialogue State: {evaluation_result.groundtruth_dialogue_state}")
        print(f"Joint Goal Accuracy: {evaluation_result.joint_goal_accuracy}")
        print(f"Joint Goal Accuracy Detail: {evaluation_result.joint_goal_accuracy_detail}")
        total_record[f"dialogue_{dialogue_case.dialogue_idx}_turn_{turn_idx}"] = {
            "dialogue_state": evaluation_result.dialogue_state,
            "groundtruth_dialogue_state": evaluation_result.groundtruth_dialogue_state,
            "joint_goal_accuracy": evaluation_result.joint_goal_accuracy,
            "joint_goal_accuracy_detail": evaluation_result.joint_goal_accuracy_detail,
        }
    if dialogue_idx >= 99:  # Limit to 100 dialogues for testing
        break
evaluation_result = Multiwoz24Environment.evaluate_test_cases(mode="test")
evaluation_result.to_json(file_name=f"experiments/multiwoz_24_experiments/new_experiments/[Qwen3-32B-FP8]_[baseline]_[Manager-Worker]/evaluation_result.json")
with open(f"experiments/multiwoz_24_experiments/new_experiments/[Qwen3-32B-FP8]_[baseline]_[Manager-Worker]/total_record.json", "w", encoding="utf-8") as f:
    json.dump(total_record, f, ensure_ascii=False, indent=2, default=convert_sets)