import json
import os
import random
from typing import Iterator, Optional, Tuple, Type

from fuzzywuzzy import fuzz

from ...metadata import MultiWOZ24MetaData
from ..base_environment import BaseEnvironment
from .tool_descriptions import (
    BOOK_HOTEL_DESCRIPTION,
    BOOK_RESTAURANT_DESCRIPTION,
    BOOK_TAXI_DESCRIPTION,
    BOOK_TRAIN_DESCRIPTION,
    QUERY_ATTRACTION_DESCRIPTION,
    QUERY_HOTEL_DESCRIPTION,
    QUERY_RESTAURANT_DESCRIPTION,
    QUERY_TRAIN_DESCRIPTION,
)
from .utils import (
    ARRIVAL_TIME_KEY,
    CLEAN_DOMAINS,
    DEPARTURE_TIME_KEY,
    DOMAIN_FINITE_KEYS,
    ENV_PATH,
    FUZZY_KEYS,
    PRIMARY_KEYS,
    generate_reference_number,
    time_str_to_minutes,
)

# Fix the random seed for reproducibility
random.seed(0)

# Load the database
database = dict()
for domain in CLEAN_DOMAINS:
    with open(os.path.join(ENV_PATH, "MultiWOZ2.4/data/mwz24/MULTIWOZ2.4", f"{domain}_db.json"), "r") as f:
        if domain != "taxi":
            database[domain] = json.load(f)
        else:
            db_str = f.read()
            db_str = db_str.replace("]\n ", "],\n ").replace("\'", "\"").replace(" :", ":").replace("[\n ", "{\n ")[:-2]+"}"
            db_dict = json.loads(db_str)
            db_raw_keys = ["taxi_colors", "taxi_types", "taxi_phone"]
            database[domain] = list()
            number_pool = set()
            for color in db_dict[db_raw_keys[0]]:
                for type in db_dict[db_raw_keys[1]]:
                    for _ in range(10):
                        # Make sure the phone numbers are unique
                        phone = "".join(random.choices("0123456789", k=10))
                        while phone in number_pool:
                            phone = "".join(random.choices("0123456789", k=10))
                        number_pool.add(phone)
                        database[domain].append({"type": f"{color} {type}", "phone": phone})

# Load the dialogues
dialogues = dict()
with open(os.path.join(ENV_PATH, "MultiWOZ2.4/data/mwz2.4", "test_dials.json"), "r") as f:
    dialogues["test"] = json.load(f)
with open(os.path.join(ENV_PATH, "MultiWOZ2.4/data/mwz2.4", "train_dials.json"), "r") as f:
    dialogues["train"] = json.load(f)
with open(os.path.join(ENV_PATH, "MultiWOZ2.4/data/mwz2.4", "dev_dials.json"), "r") as f:
    dialogues["dev"] = json.load(f)
domains_indices, processed_dialogues = dict(), dict()
for mode, data in dialogues.items():
    for dialogue_content in data:
        for domain in dialogue_content["domains"]:
            if domain in CLEAN_DOMAINS:
                if mode not in domains_indices:
                    domains_indices[mode] = {domain: list() for domain in CLEAN_DOMAINS}
                domains_indices[mode][domain].append(dialogue_content["dialogue_idx"])
                if mode not in processed_dialogues:
                    processed_dialogues[mode] = dict()
                processed_dialogues[mode][dialogue_content["dialogue_idx"]] = dialogue_content


class Multiwoz24Environment(BaseEnvironment):
    database = database
    dialogues = processed_dialogues
    domains_indices = domains_indices
    evaluation_record = MultiWOZ24MetaData()

    @classmethod
    def iterate_test_cases(cls, mode: str) -> Iterator[Type["Multiwoz24Environment"]]:
        for dialogue_idx in cls.domains_indices[mode]:
            yield cls(mode=mode, dialogue_idx=dialogue_idx)

    @classmethod
    def evaluate_test_cases(cls) -> MultiWOZ24MetaData:
        return cls.evaluation_record

    def __init__(self, mode: str, dialogue_idx: str, turn_idx: int = 0):
        super().__init__()
        self.mode = mode
        self.dialogue_idx = dialogue_idx
        self.turn_idx = turn_idx
        self.register_tool(QUERY_HOTEL_DESCRIPTION["function"]["name"], QUERY_HOTEL_DESCRIPTION, self._query_hotel)
        self.register_tool(QUERY_RESTAURANT_DESCRIPTION["function"]["name"], QUERY_RESTAURANT_DESCRIPTION, self._query_restaurant)
        self.register_tool(QUERY_ATTRACTION_DESCRIPTION["function"]["name"], QUERY_ATTRACTION_DESCRIPTION, self._query_attraction)
        self.register_tool(QUERY_TRAIN_DESCRIPTION["function"]["name"], QUERY_TRAIN_DESCRIPTION, self._query_train)
        self.register_tool(BOOK_HOTEL_DESCRIPTION["function"]["name"], BOOK_HOTEL_DESCRIPTION, self._book_hotel)
        self.register_tool(BOOK_RESTAURANT_DESCRIPTION["function"]["name"], BOOK_RESTAURANT_DESCRIPTION, self._book_restaurant)
        self.register_tool(BOOK_TRAIN_DESCRIPTION["function"]["name"], BOOK_TRAIN_DESCRIPTION, self._book_train)
        self.register_tool(BOOK_TAXI_DESCRIPTION["function"]["name"], BOOK_TAXI_DESCRIPTION, self._book_taxi)
        self.grounding_utterance = self.prepare_grounding_utterance()
        self.target_utterance = self.dialogues[self.mode][self.dialogue_idx]["dialogue"][self.turn_idx+1]["system_transcript"]
        self.set_initial_metadata(MultiWOZ24MetaData(input=self.grounding_utterance))

    def prepare_grounding_utterance(self) -> str:
        grounding_utterance = list()
        for dialogue_turn in self.dialogues[self.mode][self.dialogue_idx]["dialogue"][:self.turn_idx+1]:
            if dialogue_turn["system_transcript"]:
                grounding_utterance.append(f"""[Assistant] {dialogue_turn["system_transcript"]}""")
            if dialogue_turn["transcript"]:
                grounding_utterance.append(f"""[User] {dialogue_turn["transcript"]}""")
        return "\n".join(grounding_utterance)

    def iterate_dialog_turns(self) -> Iterator[Type["Multiwoz24Environment"]]:
        for turn_idx in range(len(self.dialogues[self.mode][self.dialogue_idx]["dialogue"])-1):
            yield self.__class__(self.mode, self.dialogue_idx, turn_idx)

    def evaluate(self, metadata: MultiWOZ24MetaData) -> MultiWOZ24MetaData:
        dialogue_state = metadata.dialogue_state
        system_response = metadata.system_response
        jga, slot_recall, slot_precision = self.compute_dst(dialogue_state)
        # Todo

    def compute_dst(self, dialogue_state: dict) -> Tuple[float, float, float]:
        true_positive, false_negative, false_positive = 0, 0, 0
        # Todo

    def _query_basic(self, domain: str, max_retrieval: int = 10, fuzzy_ratio: int = 80, **query) -> dict:
        valid_items = []
        for database_item in database[domain]:
            valid = True
            for query_key, query_value in query.items():
                database_value = database_item.get(query_key, None)
                if database_value and query_value:
                    if (domain not in DOMAIN_FINITE_KEYS) or (query_key not in DOMAIN_FINITE_KEYS[domain]) or (query_value in DOMAIN_FINITE_KEYS[domain][query_key]):
                        if query_key in FUZZY_KEYS[domain] and (fuzz.partial_ratio(database_value, query_value) < fuzzy_ratio or fuzz.partial_ratio(query_value, database_value) < fuzzy_ratio):
                            valid = False
                        if query_key == ARRIVAL_TIME_KEY and time_str_to_minutes(database_value) > time_str_to_minutes(query_value):
                            valid = False
                        if query_key == DEPARTURE_TIME_KEY and time_str_to_minutes(database_value) < time_str_to_minutes(query_value):
                            valid = False
                        if query_key not in FUZZY_KEYS[domain] and query_key not in ["arriveBy", "leaveAt"] and database_value != query_value:
                            valid = False
                if not valid:
                    break
            if valid:
                if len(valid_items) == max_retrieval:
                    return json.dumps({"result": valid_items, "message": "Too many retrieved results! Please query more accurately!"})
                valid_items.append(database_item)
        return json.dumps({"result": valid_items})
    
    def _query_restaurant(
            self,
            area: Optional[str] = None,
            pricerange: Optional[str] = None,
            food: Optional[str] = None,
            name: Optional[str] = None,
        ) -> dict:
        return self._query_basic(
            domain = "restaurant",
            area = area,
            pricerange = pricerange,
            food = food,
            name = name,
        )

    def _query_hotel(
            self,
            area: Optional[str] = None,
            internet: Optional[str] = None,
            name: Optional[str] = None,
            parking: Optional[str] = None,
            pricerange: Optional[str] = None,
            stars: Optional[str] = None,
            type: Optional[str] = None,
        ) -> dict:
        return self._query_basic(
            domain = "hotel",
            area = area,
            internet = internet,
            name = name,
            parking = parking,
            pricerange = pricerange,
            stars = stars,
            type = type,
        )
    
    def _query_attraction(
            self,
            area: Optional[str] = None,
            name: Optional[str] = None,
            type: Optional[str] = None,
        ) -> dict:
        return self._query_basic(
            domain = "attraction",
            area = area,
            name = name,
            type = type,
        )
    
    def _query_train(
            self,
            day: Optional[str] = None,
            departure: Optional[str] = None,
            destination: Optional[str] = None,
            leaveAt: Optional[str] = None,
            arriveBy: Optional[str] = None,
            trainID: Optional[str] = None,
        ) -> dict:
        return self._query_basic(
            domain = "train",
            day = day,
            departure = departure,
            destination = destination,
            leaveAt = leaveAt,
            arriveBy = arriveBy,
            trainID = trainID,
        )
    
    def _book_basic(self, domain: str, fuzzy_ratio: int = 80, **query) -> dict:
        primary_key = PRIMARY_KEYS[domain]["primary_key"]
        other_keys = PRIMARY_KEYS[domain]["other_keys"]
        found = False
        if primary_key is None and (other_keys is None or set(other_keys).issubset(set(query.keys()))):
            found = True
        elif primary_key in query.keys() and (other_keys is None or set(other_keys).issubset(set(query.keys()))):
            for database_item in database[domain]:
                database_value = database_item.get(primary_key, None)
                query_value = query[primary_key]
                if (primary_key in FUZZY_KEYS[domain]) and (fuzz.partial_ratio(database_value, query_value) >= fuzzy_ratio) or (fuzz.partial_ratio(query_value, database_value) >= fuzzy_ratio):
                    found = True
                if primary_key not in FUZZY_KEYS[domain] and database_value == query_value:
                    found = True
        if found:
            return json.dumps({"result": {"reference": generate_reference_number(8)}, "message": "Success! Reference number is returned!"})
        elif other_keys is not None and not set(other_keys).issubset(set(query.keys())):
            remainders = other_keys - set(query.keys())
            return json.dumps({"result": None, "message": f"Failure! API keys are incomplete! The following keys are missing: {", ".join(remainders)}!"})
        else:
            return json.dumps({"result": None, "message": "Failure! No matched entity in database!"})

    def _book_restaurant(
            self,
            name: Optional[str] = None,
            people: Optional[str] = None,
            day: Optional[str] = None,
            time: Optional[str] = None,
        ) -> dict:
        return self._book_basic(
            domain = "restaurant",
            name = name,
            people = people,
            day = day,
            time = time,    
        )
    
    def _book_hotel(
            self,
            name: Optional[str] = None,
            people: Optional[str] = None,
            day: Optional[str] = None,
            stay: Optional[str] = None,
        ) -> dict:
        return self._book_basic(
            domain = "hotel",
            name = name,
            people = people,
            day = day,
            stay = stay,
        )
    
    def _book_train(
            self,
            trainID: Optional[str] = None,
            people: Optional[str] = None,
        ) -> dict:
        return self._book_basic(
            domain = "train",
            trainID = trainID,
            people = people,
        )
    
    def _book_taxi(
            self,
            departure: Optional[str] = None,
            destination: Optional[str] = None,
            arriveBy: Optional[str] = None,
            leaveAt: Optional[str] = None,
        ) -> dict:
        booking_result = self._book_basic(
            domain = "taxi",
            departure = departure,
            destination = destination,
            arriveBy = arriveBy,
            leaveAt = leaveAt,
        )
        if booking_result["result"]:
            entity = random.choice(database["taxi"])
            booking_result["result"]["phone"] = entity["phone"]
            booking_result["result"]["type"] = entity["type"]
        return booking_result
    
# Todo: Implement the evaluation function
