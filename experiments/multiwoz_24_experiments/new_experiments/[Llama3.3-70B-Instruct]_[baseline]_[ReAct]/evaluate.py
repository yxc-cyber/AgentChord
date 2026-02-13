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
)

global_dialogue_state = dict()

prompt_analyze = """
You are a helpful agent in a multi-agent system. You can analyze the dialogue history, reason about the user's needs, and make plans accordingly.
You will not be acutally taking any actions, but your reasoning and analysis will be used by another agent to decide which tools to use.
Your analysis and reasoning should be in texts only. You should put your output in the "output" field of the "terminate" tool.
""".strip()

prompt_use_tool = """
You are a helpful agent in a multi-agent system. You can properly use the tools to make queries or bookings for the user.
You will be provided with a dialogue history and an analysis by the previous agent.
Based on the dialogue history and the analysis, you should decide whether to use the tools or not. You can use another tool after you have used one and received the tool result.
Finally, you should generate the response to the user.
""".strip()

class Multiwoz24System(BaseAgentSystem):
    def __init__(self, system_name: str, environment: Multiwoz24Environment, maximum_loops: int = 5, log_name: str = ""):
        super().__init__(
            system_name=system_name,
            environment=environment,
            maximum_loops=maximum_loops,
            log_name=log_name
        )
        AnalyzeAgent = BaseAgent(
            system_name="analyze_agent",
            environment=environment,
            tools=[],
            prompt=prompt_analyze,
            model_config=ModelConfig(client_model="openai/Llama3.3-70B-Instruct", temperature=0.0),
            maximum_loops=maximum_loops,
            log_name=log_name
        )
        ToolAgent = BaseAgent(
            system_name="tool_agent",
            environment=environment,
            prompt=prompt_use_tool,
            model_config=ModelConfig(client_model="openai/Llama3.3-70B-Instruct", temperature=0.0),
            maximum_loops=maximum_loops,
            log_name=log_name
        )

        self.add_subsystem(AnalyzeAgent)
        self.add_on_completion_action("analyze_agent", "analyze", self._analyze)
        self.add_subsystem(ToolAgent)
        self.add_on_completion_action("tool_agent", "read_state", self._read_state)

    def on_initialization(self) -> MultiWOZ24MetaData:
        metadata = self.environment.get_initial_metadata()
        grounding_utterance = metadata.grounding_utterance
        metadata.input = f"Dialogue History:\n{grounding_utterance}"
        return metadata
    
    def _analyze(self, metadata: MultiWOZ24MetaData) -> MultiWOZ24MetaData:
        """
        Combine the analysis from the analyze agent into the metadata.
        """
        metadata.input = [
            f"Dialogue History:\n{metadata.grounding_utterance}",
            f"Analysis:\n{metadata.output}"
        ]
        return metadata

    def _read_state(self, metadata: MultiWOZ24MetaData) -> MultiWOZ24MetaData:
        """
        Read the state from the the output of the state agent.
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
    
multiwoz_24_system = Multiwoz24System("multiwoz_24_system", Multiwoz24Environment(), log_name="multiwoz_experiment_baseline_react.log")
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
evaluation_result.to_json(file_name=f"experiments/multiwoz_24_experiments/new_experiments/[Llama3.3-70B-Instruct]_[baseline]_[ReAct]/evaluation_result.json")
with open(f"experiments/multiwoz_24_experiments/new_experiments/[Llama3.3-70B-Instruct]_[baseline]_[ReAct]/total_record.json", "w", encoding="utf-8") as f:
    json.dump(total_record, f, ensure_ascii=False, indent=2, default=convert_sets)