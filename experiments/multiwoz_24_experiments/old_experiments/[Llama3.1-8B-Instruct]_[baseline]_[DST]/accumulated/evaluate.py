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

prompt_read_state = """
You are a helpful agent that can retrieve dialogue states from the user. The dialogue state, essentially, is the intent of the user that have been shown in the dialogue history. Make sure you cover all the mentioned intents in the dialogue history.
The dialogue state should be formatted as a JSON object that contains the slots and their values. The keys of the dialogue state are the names of the intent slots, and the values are the values of the slots.
The keys of the dialogue state are:
```json
{
    "attraction-area": {
        "type": "string",
        "description": "The area in which the attraction is located.",
        "enum": ["centre", "north", "south", "east", "west"]
    },
    "attraction-name": {
        "type": "string",
        "description": "The name of the attraction."
    },
    "attraction-type": {
        "type": "string",
        "description": "The type of the attraction.",
        "enum": ["museum", "swimmingpool", "architecture", "boat", "college", "nightclub", "entertainment", "cinema", "concerthall", "mutliple sports", "park", "theatre"]
    },
    "hotel-area": {
        "type": "string",
        "description": "The area in which the hotel is located.",
        "enum": ["centre", "north", "south", "east", "west"]
    },
    "hotel-book day": {
        "type": "string",
        "description": "The day of the booking.",
        "enum": ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
    },
    "hotel-book people": {
        "type": "string",
        "description": "The number of people in the booking."
    },
    "hotel-book stay": {
        "type": "string",
        "description": "The number of days of the booking."
    },
    "hotel-internet": {
        "type": "string",
        "description": "Whether the hotel has internet.",
        "enum": ["yes", "no"]
    },
    "hotel-name": {
        "type": "string",
        "description": "The name of the hotel."
    },
    "hotel-parking": {
        "type": "string",
        "description": "Whether the hotel has parking.",
        "enum": ["yes", "no"]
    },
    "hotel-pricerange": {
        "type": "string",
        "description": "The price range of the hotel.",
        "enum": ["cheap", "moderate", "expensive"]
    },
    "hotel-stars": {
        "type": "string",
        "description": "The number of stars of the hotel.",
        "enum": ["0", "1", "2", "3", "4", "5"]
    },
    "hotel-type": {
        "type": "string",
        "description": "The type of the hotel.",
        "enum": ["bed and breakfast", "guesthouse", "hotel"]
    },
    "restaurant-area": {
        "type": "string",
        "description": "The area in which the restaurant is located.",
        "enum": ["centre", "north", "south", "east", "west"]
    },
    "restaurant-book day": {
        "type": "string",
        "description": "The day of the booking.",
        "enum": ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
    },
    "restaurant-book people": {
        "type": "string",
        "description": "The number of people in the booking."
    },
    "restaurant-book time": {
        "type": "string",
        "description": "The time of the booking in the format HH:MM."
    },
    "restaurant-food": {
        "type": "string",
        "description": "The type of food served at the restaurant.",
        "enum": ["international", "indian", "mediterranean", "italian", "vietnamese", "lebanese", "african", "modern european", "french", "european", "portuguese", "japanese", "seafood", "chinese", "turkish", "gastropub", "british", "thai", "spanish", "korean", "north american", "mexican", "asian oriental"]
    },
    "restaurant-name": {
        "type": "string",
        "description": "The name of the restaurant."
    },
    "restaurant-pricerange": {
        "type": "string",
        "description": "The price range of the restaurant.",
        "enum": ["cheap", "moderate", "expensive"]
    },
    "taxi-arriveby": {
        "type": "string",
        "description": "The time by which the taxi should arrive in the format HH:MM."
    },
    "taxi-departure": {
        "type": "string",
        "description": "The departure location of the taxi."
    },
    "taxi-destination": {
        "type": "string",
        "description": "The destination location of the taxi."
    },
    "taxi-leaveat": {
        "type": "string",
        "description": "The time at which the taxi should leave in the format HH:MM."
    },
    "train-arriveby": {
        "type": "string",
        "description": "The time by which the train arrives in the format HH:MM."
    },
    "train-book people": {
        "type": "string",
        "description": "The number of people in the booking."
    },
    "train-day": {
        "type": "string",
        "description": "The day on which the train departs.",
        "enum": ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
    },
    "train-departure": {
        "type": "string",
        "description": "The departure location of the train."
    },
    "train-destination": {
        "type": "string",
        "description": "The destination location of the train."
    },
    "train-leaveat": {
        "type": "string",
        "description": "The time at which the train leaves in the format HH:MM."
    },
}
```
The dialogue state should be put in JSON format. One example of the dialogue state is:
```json
{
    "attraction-area": "centre",
    "attraction-name": "Theatre Royal",
    "attraction-type": "theatre",
    "hotel-area": "centre",
    "hotel-book day": "monday",
    "hotel-book people": "2",
    "hotel-book stay": "3",
    "hotel-internet": "yes",
    "hotel-name": "Theatre Royal Hotel",
    "hotel-parking": "no",
    "hotel-pricerange": "moderate",
    "hotel-stars": 4,
    "hotel-type": "guesthouse",
    "restaurant-area": "centre",
    "restaurant-book day": "monday",
    "restaurant-book people": 2,
    "restaurant-book time": 19:00
}
```
""".strip()

class Multiwoz24System(BaseAgentSystem):
    def __init__(self, system_name: str, environment: Multiwoz24Environment, maximum_loops: int = 5, log_name: str = ""):
        super().__init__(
            system_name=system_name,
            environment=environment,
            maximum_loops=maximum_loops,
            log_name=log_name
        )
        StateAgent = BaseAgent(
            system_name="state_agent",
            environment=environment,
            tools=[],
            prompt=prompt_read_state,
            model_config=ModelConfig(client_model="together_ai/meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo", temperature=0.0),
            maximum_loops=maximum_loops,
            log_name=log_name
        )

        self.add_subsystem(StateAgent)
        self.add_on_completion_action("state_agent", "read_state", self._read_state)

    def on_initialization(self) -> MultiWOZ24MetaData:
        metadata = self.environment.get_initial_metadata()
        grounding_utterance = metadata.grounding_utterance
        metadata.input = f"Dialogue History:\n{grounding_utterance}"
        return metadata

    def _read_state(self, metadata: MultiWOZ24MetaData) -> MultiWOZ24MetaData:
        """
        Read the state from the the output of the state agent.
        """
        dialogue_state_raw = metadata.output.strip() or metadata.note.strip()
        try:
            dialogue_state = json.loads(dialogue_state_raw)
            if dialogue_state is None:
                dialogue_state = dict()
        except json.JSONDecodeError:
            _json_markdown_re = re.compile(r"```(json)?(.*?)```", re.DOTALL)
            match = _json_markdown_re.search(dialogue_state_raw)
            if match:
                try:
                    dialogue_state = json.loads(match.group(2))
                except json.JSONDecodeError:
                    print(f"Failed to parse JSON from dialogue state: {dialogue_state_raw}")
                    dialogue_state = dict()
            else:
                dialogue_state = dict()
        if not isinstance(dialogue_state, dict):
            if isinstance(dialogue_state, list) and len(dialogue_state) > 0 and isinstance(dialogue_state[0], dict):
                dialogue_state_new = dict()
                for item in dialogue_state:
                    if isinstance(item, dict):
                        dialogue_state_new.update(item)
                dialogue_state = dialogue_state_new
            else:
                print(f"Dialogue state is not a dictionary: {dialogue_state}")
                dialogue_state = dict()
        dialogue_state = {slot: value for slot, value in dialogue_state.items() if len(slot.split('-')) == 2 and isinstance(value, str)}
        # Connected to the global dialogue state
        global global_dialogue_state
        global_dialogue_state.update(dialogue_state)
        dialogue_state = deepcopy(global_dialogue_state)

        metadata.dialogue_state = dialogue_state
        return metadata
    
    def on_finalization(self, metadata: MultiWOZ24MetaData) -> MultiWOZ24MetaData:
        """
        Finalize the metadata after the response agent has generated the response.
        """
        metadata.system_response = ""
        return metadata
    
multiwoz_24_system = Multiwoz24System("multiwoz_24_system", Multiwoz24Environment(), log_name="multiwoz_experiment_baseline_dst.log")
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
evaluation_result.to_json(file_name=f"experiments/multiwoz_24_experiments/[Llama3.1-8B-Instruct]_[baseline]_[DST]/accumulated/evaluation_result.json")
with open(f"experiments/multiwoz_24_experiments/[Llama3.1-8B-Instruct]_[baseline]_[DST]/accumulated/total_record.json", "w", encoding="utf-8") as f:
    json.dump(total_record, f, ensure_ascii=False, indent=2, default=convert_sets)