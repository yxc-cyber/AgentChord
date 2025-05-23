import json
import re

from agentchord import (
    BaseAgent,
    BaseAgentSystem,
    ModelConfig,
    Multiwoz24Environment,
    MultiWOZ24MetaData,
)

prompt_read_state = """
You are a helpful agent that can retrieve dialogue states from the user.
The keys are the names of the slots and the values are the values of the slots. The keys of the dialogue state are:
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
    "taxi-arriveBy": {
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
    "taxi-leaveAt": {
        "type": "string",
        "description": "The time at which the taxi should leave in the format HH:MM."
    },
    "train-arriveBy": {
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
    "train-leaveAt": {
        "type": "string",
        "description": "The time at which the train leaves in the format HH:MM."
    },
}
```
The entries of the dialogue state should be put in JSON format. One example of the dialogue state is:
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
prompt_generate_response = """
You are an advanced AI assistant specializing in conversational dialogues. You can interact with the database and provide service to assist users in completing complex tasks. 
Each task may involve multiple sub-tasks, such as finding restaurants, making reservations, booking hotels, locating attractions, and arranging transportation by checking for trains and buying train tickets.
You are given the dialogue history and the current dialogue state. You need to query the database and generate a response based on the dialogue history and the database results.
"""

class Multiwoz24System(BaseAgentSystem):
    def __init__(self, system_name: str, environment: Multiwoz24Environment, maximum_loops: int = 5, log_name: str = ""):
        super().__init__(system_name, environment, maximum_loops, log_name)
        StateAgent = BaseAgent(
            system_name="state_agent",
            environment=environment,
            prompt=prompt_read_state,
            model_config=ModelConfig(client_model="openai/gpt-4o-mini", temperature=0.0),
            maximum_loops=maximum_loops,
            log_name=log_name
        )
        ResponseAgent = BaseAgent(
            system_name="response_agent",
            environment=environment,
            prompt=prompt_generate_response,
            model_config=ModelConfig(client_model="openai/gpt-4o-mini", temperature=0.0),
            maximum_loops=maximum_loops,
            log_name=log_name
        )

        self.add_subsystem(StateAgent)
        self.add_subsystem(ResponseAgent)
        self.add_on_start_action("response_agent", "read_state", self._read_state)

    def on_initialization(self) -> MultiWOZ24MetaData:
        metadata = self.environment.get_initial_metadata()
        grounding_utterance = metadata.grounding_utterance
        metadata.input = f"Dialogue History:\n{grounding_utterance}"
        return metadata

    def _read_state(self, metadata: MultiWOZ24MetaData) -> MultiWOZ24MetaData:
        """
        Read the state from the the output of the state agent.
        """
        grounding_utterance = metadata.grounding_utterance
        dialogue_state_raw = metadata.output.strip()
        try:
            dialogue_state = json.loads(dialogue_state_raw)
        except json.JSONDecodeError:
            _json_markdown_re = re.compile(r"```(json)?(.*)```", re.DOTALL)
            match = _json_markdown_re.search(dialogue_state_raw)
            if match:
                dialogue_state = json.loads(match.group(2))
            else:
                dialogue_state = dict()
        metadata.dialogue_state = dialogue_state
        metadata.input = f"Dialogue History:\n{grounding_utterance}\nDialogue State:\n{json.dumps(dialogue_state)}"
        return metadata
    
multiwoz_24_system = Multiwoz24System("multiwoz_24_system", Multiwoz24Environment(), log_name="multiwoz_24_system.log")
print(multiwoz_24_system.get_pipeline_description())

for dialogue_idx, dialogue_case in enumerate(Multiwoz24Environment.iterate_test_cases(mode="test")):
    for turn_idx, turn_case in enumerate(dialogue_case.iterate_dialog_turns()):
        multiwoz_24_system.set_environment(environment=turn_case)
        result = multiwoz_24_system.run()
        evaluation_result = turn_case.evaluate(result)
        # if turn_idx > 2:
        #     break
    if dialogue_idx > 2:
        break
evaluation_result = Multiwoz24Environment.evaluate_test_cases(mode="test").to_json("./examples/multiwoz_24_evaluation.json")
print(evaluation_result)
# Save Multiwoz24Environment.evaluation_record to a json file
# with open("./examples/evaluation_record.json", "w") as f:
#     json.dump(Multiwoz24Environment.evaluation_record, f, indent=4)