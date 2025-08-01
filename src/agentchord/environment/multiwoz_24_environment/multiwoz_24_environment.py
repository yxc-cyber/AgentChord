import json
import os
import random
from typing import Iterator, List, Optional, Self, Tuple, Union

from fuzzywuzzy import fuzz
from git import Repo

from ...metadata import MultiWOZ24MetaData
from ..base_environment import BaseEnvironment
from .normalization import normalize_data
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
    DATA_PATH,
    DEPARTURE_TIME_KEY,
    DOMAIN_FINITE_KEYS,
    ENV_PATH,
    FUZZY_KEYS,
    PRIMARY_KEYS,
    REPO_URL,
    delexicalize,
    generate_reference_number,
    prepareSlotValuesIndependent,
    time_str_to_minutes,
)


class Multiwoz24Environment(BaseEnvironment):
    evaluation_record = dict()

    @classmethod
    def pre_initialize(cls):
        """
        Pre-initializes the Multiwoz24Environment, setting up necessary configurations or resources.
        This method should be called before using the environment to ensure it is ready for use.
        """
        if not cls.pre_initialized:
            # Check if the MultiWOZ 2.4 dataset is already cloned, if not, clone it
            if not os.path.exists(DATA_PATH):
                Repo.clone_from(REPO_URL, DATA_PATH)
                print(f"Repository cloned to {DATA_PATH}")
                # Change the current working directory to the cloned repository
                original_dir = os.getcwd()
                os.chdir(DATA_PATH)
                print(f"Changed working directory to {os.getcwd()}")
                # Now run "python3 create_data.py"
                os.system("python3 create_data.py")
                print("Data created successfully.")
                # Change back to the original directory
                os.chdir(original_dir)
                print(f"Changed back to the original working directory: {original_dir}")
            
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

            # Prepare the delexicalization map
            delexicalization_map = prepareSlotValuesIndependent(database)

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
                    valid = True
                    for domain in dialogue_content["domains"]:
                        if domain in CLEAN_DOMAINS:
                            if mode not in domains_indices:
                                domains_indices[mode] = {domain: list() for domain in CLEAN_DOMAINS}
                            domains_indices[mode][domain].append(dialogue_content["dialogue_idx"])
                        else:
                            valid = False
                    if valid:
                        if mode not in processed_dialogues:
                            processed_dialogues[mode] = dict()
                        processed_dialogues[mode][dialogue_content["dialogue_idx"]] = dialogue_content

            # Load the goals
            goals = dict()
            with open(os.path.join(ENV_PATH, "goals.json"), "r") as f:
                goals_raw = json.load(f)
            for dialogue_idx, dialogue_content in goals_raw.items():
                goals[dialogue_idx.upper()+".json"] = dialogue_content

            # Load the booked domains
            booked_domains = dict()
            with open(os.path.join(ENV_PATH, "booked_domains.json"), "r") as f:
                booked_domains_raw = json.load(f)
            for dialogue_idx, dialogue_content in booked_domains_raw.items():
                booked_domains[dialogue_idx.upper()+".json"] = dialogue_content

            # Store the loaded data in the class attributes
            cls.database = database
            cls.delexicalization_map = delexicalization_map
            cls.domains_indices = domains_indices
            cls.dialogues = processed_dialogues
            cls.goals = goals
            cls.booked_domains = booked_domains
            cls.pre_initialized = True

    @classmethod
    def iterate_test_cases(cls, mode: str) -> Iterator[Self]:
        cls.pre_initialize()
        for dialogue_idx in cls.dialogues[mode]:
            if len(cls.dialogues[mode][dialogue_idx]["dialogue"]) >= 2:  # Ensure there is at least one user turn and one system turn
                yield cls(mode=mode, dialogue_idx=dialogue_idx)

    @classmethod
    def evaluate_test_cases(cls, mode: str, dialogue_indices: Optional[Union[List[str], str]] = None) -> MultiWOZ24MetaData:
        cls.pre_initialize()
        matched_turns, true_positive, false_positive, false_negative, total_turns = 0.0, 0.0, 0.0, 0.0, 0.0
        total_inform, total_success, total_dialogues = 0.0, 0.0, 0.0
        system_response = list()
        inform_detail = dict()
        success_detail = dict()
        for dialogue_idx, dialogue_eval_record in cls.evaluation_record[mode].items():
            if (isinstance(dialogue_indices, list) and dialogue_idx not in dialogue_indices) or \
                (isinstance(dialogue_indices, str) and dialogue_idx != dialogue_indices):
                continue
            inform, success = {"total": 0.0}, {"total": 0.0}
            requested_slots, provided_slots = dict(),  dict()
            requested_queries, provided_queries = dict(), dict()
            for turn_idx, turn_eval_record in dialogue_eval_record.items():
                system_response.append(turn_eval_record.system_response)
                matched_turns += turn_eval_record.matched_turns
                true_positive += turn_eval_record.true_positive
                false_positive += turn_eval_record.false_positive
                false_negative += turn_eval_record.false_negative
                total_turns += turn_eval_record.total_turns
                for domain, domain_inform in turn_eval_record.inform.items():
                    if domain not in inform:
                        inform[domain] = 0.0
                    if domain != "total" and domain_inform > 0.0:
                        inform[domain] = domain_inform
                inform["total"] = float(sum(inform.values()) >= len(inform.keys())-1)  # Exclude the "total" key from the count
                for domain, domain_inform_detail in turn_eval_record.inform_detail.items():
                    if domain not in requested_slots:
                        requested_queries[domain] = list()
                    if domain not in provided_slots:
                        provided_queries[domain] = list()
                    requested_queries[domain].extend(domain_inform_detail["requested"])
                    provided_queries[domain].extend(domain_inform_detail["provided"])
                for domain, domain_success_detail in turn_eval_record.success_detail.items():
                    if domain not in requested_slots:
                        requested_slots[domain] = set()
                    if domain not in provided_slots:
                        provided_slots[domain] = set()
                    requested_slots[domain].update(domain_success_detail["requested"])
                    provided_slots[domain].update(domain_success_detail["provided"])
            for domain in turn_eval_record.success_detail.keys():
                success[domain] = float(len(requested_slots[domain]) == len(provided_slots[domain]))
            if inform["total"]:
                success["total"] = float(sum(success.values()) >= len(success.keys())-1)  # Exclude the "total" key from the count
            else:
                success["total"] = 0.0
            total_inform += inform["total"]
            total_success += success["total"]
            total_dialogues += 1.0
            inform_detail = {domain: {"requested": requested_queries[domain], "provided": provided_queries[domain]} for domain in requested_queries}
            success_detail = {domain: {"requested": requested_slots[domain], "provided": provided_slots[domain]} for domain in requested_slots}
        joint_goal_accuracy = matched_turns / (total_turns + 1e-10)
        slot_recall = true_positive / (true_positive + false_negative + 1e-10)
        slot_precision = true_positive / (true_positive + false_positive + 1e-10)
        slot_f1 = 2 * slot_precision * slot_recall / (slot_precision + slot_recall + 1e-10)
        total_inform = total_inform / (total_dialogues + 1e-10)
        inform = {domain: inform[domain] for domain in inform if domain != "total"}
        inform["total"] = total_inform
        total_success = total_success / (total_dialogues + 1e-10)
        success = {domain: success[domain] for domain in success if domain != "total"}
        success["total"] = total_success
        return MultiWOZ24MetaData(
            system_response=system_response,
            matched_turns=matched_turns,
            total_turns=total_turns,
            true_positive=true_positive,
            false_positive=false_positive,
            false_negative=false_negative,
            inform=inform,
            inform_detail=inform_detail,
            success=success,
            success_detail=success_detail,
            joint_goal_accuracy=joint_goal_accuracy,
            slot_recall=slot_recall,
            slot_precision=slot_precision,
            slot_f1=slot_f1,
        )

    def __init__(self, mode: str = "test", dialogue_idx: str = "", turn_idx: int = 0, fuzzy_ratio: int = 80):
        from ...agent_system import Input
        from ...gbc_object import GBC

        super().__init__()
        if not dialogue_idx:
            dialogue_idx = list(self.dialogues[mode].keys())[0]
        self.mode = mode
        self.dialogue_idx = dialogue_idx
        self.turn_idx = turn_idx
        self.fuzzy_ratio = fuzzy_ratio
        self.register_tool(QUERY_HOTEL_DESCRIPTION["function"]["name"], QUERY_HOTEL_DESCRIPTION, self._query_hotel)
        self.register_tool(QUERY_RESTAURANT_DESCRIPTION["function"]["name"], QUERY_RESTAURANT_DESCRIPTION, self._query_restaurant)
        self.register_tool(QUERY_ATTRACTION_DESCRIPTION["function"]["name"], QUERY_ATTRACTION_DESCRIPTION, self._query_attraction)
        self.register_tool(QUERY_TRAIN_DESCRIPTION["function"]["name"], QUERY_TRAIN_DESCRIPTION, self._query_train)
        self.register_tool(BOOK_HOTEL_DESCRIPTION["function"]["name"], BOOK_HOTEL_DESCRIPTION, self._book_hotel)
        self.register_tool(BOOK_RESTAURANT_DESCRIPTION["function"]["name"], BOOK_RESTAURANT_DESCRIPTION, self._book_restaurant)
        self.register_tool(BOOK_TRAIN_DESCRIPTION["function"]["name"], BOOK_TRAIN_DESCRIPTION, self._book_train)
        self.register_tool(BOOK_TAXI_DESCRIPTION["function"]["name"], BOOK_TAXI_DESCRIPTION, self._book_taxi)
        self.grounding_utterance = self.prepare_grounding_utterance()
        self.grounding_utterance = GBC(self.grounding_utterance, subject=Input())
        self.target_utterance = self.dialogues[self.mode][self.dialogue_idx]["dialogue"][self.turn_idx+1]["system_transcript"]
        ground_truth_raw = self.dialogues[self.mode][self.dialogue_idx]["dialogue"][self.turn_idx]["belief_state"]
        ground_truth_dialogue_state = dict()
        for ground_truth in ground_truth_raw:
            ground_truth_dialogue_state[ground_truth["slots"][0][0]] = ground_truth["slots"][0][1]
        ground_truth_dialogue_state = normalize_data(ground_truth_dialogue_state, type="state")
        self.groundtruth_dialogue_state = ground_truth_dialogue_state
        self.set_initial_metadata(MultiWOZ24MetaData(input=self.grounding_utterance, grounding_utterance=self.grounding_utterance, groundtruth_dialogue_state=self.groundtruth_dialogue_state))

    def prepare_grounding_utterance(self) -> str:
        grounding_utterance = list()
        for dialogue_turn in self.dialogues[self.mode][self.dialogue_idx]["dialogue"][:self.turn_idx+1]:
            if dialogue_turn["system_transcript"]:
                grounding_utterance.append(f"""[Assistant] {dialogue_turn["system_transcript"]}""")
            if dialogue_turn["transcript"]:
                grounding_utterance.append(f"""[User] {dialogue_turn["transcript"]}""")
        return "\n".join(grounding_utterance)

    def iterate_dialog_turns(self) -> Iterator[Self]:
        for turn_idx in range(len(self.dialogues[self.mode][self.dialogue_idx]["dialogue"])-1):
            yield self.__class__(self.mode, self.dialogue_idx, turn_idx)

    def evaluate(self, metadata: MultiWOZ24MetaData) -> MultiWOZ24MetaData:
        self.init_evaluation_record()
        dialogue_state = metadata.dialogue_state
        system_response = metadata.system_response
        tool_usage = metadata.tool
        self.evaluation_record[self.mode][self.dialogue_idx][self.turn_idx].grounding_utterance = metadata.grounding_utterance
        self.evaluation_record[self.mode][self.dialogue_idx][self.turn_idx].groundtruth_dialogue_state = metadata.groundtruth_dialogue_state
        self.evaluation_record[self.mode][self.dialogue_idx][self.turn_idx].dialogue_state = dialogue_state
        self.evaluation_record[self.mode][self.dialogue_idx][self.turn_idx].system_response = system_response
        # Compute the dialogue state accuracy
        matched_turns, true_positive, false_positive, false_negative, joint_goal_accuracy_detail = self.compute_dst(dialogue_state)
        self.evaluation_record[self.mode][self.dialogue_idx][self.turn_idx].matched_turns += matched_turns
        self.evaluation_record[self.mode][self.dialogue_idx][self.turn_idx].true_positive += true_positive
        self.evaluation_record[self.mode][self.dialogue_idx][self.turn_idx].false_positive += false_positive
        self.evaluation_record[self.mode][self.dialogue_idx][self.turn_idx].false_negative += false_negative
        self.evaluation_record[self.mode][self.dialogue_idx][self.turn_idx].total_turns += 1
        self.evaluation_record[self.mode][self.dialogue_idx][self.turn_idx].joint_goal_accuracy = self.evaluation_record[self.mode][self.dialogue_idx][self.turn_idx].matched_turns / (self.evaluation_record[self.mode][self.dialogue_idx][self.turn_idx].total_turns + 1e-10)
        self.evaluation_record[self.mode][self.dialogue_idx][self.turn_idx].slot_recall = self.evaluation_record[self.mode][self.dialogue_idx][self.turn_idx].true_positive / (self.evaluation_record[self.mode][self.dialogue_idx][self.turn_idx].true_positive + self.evaluation_record[self.mode][self.dialogue_idx][self.turn_idx].false_negative + 1e-10)
        self.evaluation_record[self.mode][self.dialogue_idx][self.turn_idx].slot_precision = self.evaluation_record[self.mode][self.dialogue_idx][self.turn_idx].true_positive / (self.evaluation_record[self.mode][self.dialogue_idx][self.turn_idx].true_positive + self.evaluation_record[self.mode][self.dialogue_idx][self.turn_idx].false_positive + 1e-10)
        self.evaluation_record[self.mode][self.dialogue_idx][self.turn_idx].slot_f1 = 2 * self.evaluation_record[self.mode][self.dialogue_idx][self.turn_idx].slot_precision * self.evaluation_record[self.mode][self.dialogue_idx][self.turn_idx].slot_recall / (self.evaluation_record[self.mode][self.dialogue_idx][self.turn_idx].slot_precision + self.evaluation_record[self.mode][self.dialogue_idx][self.turn_idx].slot_recall + 1e-10)
        self.evaluation_record[self.mode][self.dialogue_idx][self.turn_idx].joint_goal_accuracy_detail = joint_goal_accuracy_detail
        # Compute the system response accuracy
        delexicalized_system_response = delexicalize(system_response, self.delexicalization_map, tool_usage)
        goal = self.goals[self.dialogue_idx]
        inform, inform_detail, success, success_detail = self.compute_success(delexicalized_system_response, tool_usage, goal)
        self.evaluation_record[self.mode][self.dialogue_idx][self.turn_idx].delixicalized_system_response = delexicalized_system_response
        self.evaluation_record[self.mode][self.dialogue_idx][self.turn_idx].inform = inform
        self.evaluation_record[self.mode][self.dialogue_idx][self.turn_idx].inform_detail = inform_detail
        self.evaluation_record[self.mode][self.dialogue_idx][self.turn_idx].success = success
        self.evaluation_record[self.mode][self.dialogue_idx][self.turn_idx].success_detail = success_detail
        return self.evaluation_record[self.mode][self.dialogue_idx][self.turn_idx]
        
    def init_evaluation_record(self):
        if self.mode not in self.evaluation_record:
            self.evaluation_record[self.mode] = dict()
        if self.dialogue_idx not in self.evaluation_record[self.mode]:
            self.evaluation_record[self.mode][self.dialogue_idx] = dict()
        self.evaluation_record[self.mode][self.dialogue_idx][self.turn_idx] = MultiWOZ24MetaData()

    # The following function is adapted from uiuc-conversational-ai-lab/multiwoz-helper:
    # https://github.com/uiuc-conversational-ai-lab/multiwoz-helper/blob/main/mwzeval/metrics.py#L265
    def compute_dst(self, dialogue_state: dict) -> Tuple[float, float, float, float]:
        dialogue_state = normalize_data(dialogue_state, type="state")
        true_positive, false_negative, false_positive = 0.0, 0.0, 0.0
        joint_goal_accuracy_detail = {"true_positive": dict(), "false_positive": dict(), "false_negative": list()}
        ground_truth_dialogue_state = self.groundtruth_dialogue_state
        for dialogue_state_slot, dialogue_state_value in dialogue_state.items():
            if dialogue_state_slot in ground_truth_dialogue_state:
                domain, slot = dialogue_state_slot.split("-")
                if domain in FUZZY_KEYS and slot in FUZZY_KEYS[domain]:
                    if (fuzz.partial_ratio(dialogue_state_value, ground_truth_dialogue_state[dialogue_state_slot]) >= self.fuzzy_ratio) or (fuzz.partial_ratio(ground_truth_dialogue_state[dialogue_state_slot], dialogue_state_value) >= self.fuzzy_ratio):
                        true_positive += 1
                        joint_goal_accuracy_detail["true_positive"][dialogue_state_slot] = dialogue_state_value
                    else:
                        false_positive += 1
                        joint_goal_accuracy_detail["false_positive"][dialogue_state_slot] = dialogue_state_value
                else:
                    if dialogue_state_value == ground_truth_dialogue_state[dialogue_state_slot]:
                        true_positive += 1
                        joint_goal_accuracy_detail["true_positive"][dialogue_state_slot] = dialogue_state_value
                    else:
                        false_positive += 1
                        joint_goal_accuracy_detail["false_positive"][dialogue_state_slot] = dialogue_state_value
            else:
                false_positive += 1
                joint_goal_accuracy_detail["false_positive"][dialogue_state_slot] = dialogue_state_value
        for ground_truth_slot in ground_truth_dialogue_state.keys():
            if ground_truth_slot not in dialogue_state:
                false_negative += 1
                joint_goal_accuracy_detail["false_negative"].append(ground_truth_slot)
        matched_turns = float((false_positive + false_negative) == 0)
        return matched_turns, true_positive, false_positive, false_negative, joint_goal_accuracy_detail
    
    # The following function is adapted from uiuc-conversational-ai-lab/multiwoz-helper:
    # https://github.com/uiuc-conversational-ai-lab/multiwoz-helper/blob/main/mwzeval/metrics.py#L160
    def compute_success(self, system_response: str, tool_usage: dict, goal: dict) -> Tuple[dict, dict]:
        system_response = normalize_data(system_response, type="response")
        requestable_slots_in_goal = {domain : set(goal[domain]["requestable"]) for domain in goal}
        offered_venues = {domain : list() for domain in goal}
        provided_requestable_slots = {domain : set() for domain in goal}
        booked_domain = self.booked_domains[self.dialogue_idx][self.turn_idx]
        # Find offered venues and provided requestable slots in system utterances
        match_detail = dict()
        for current_domain in goal:
            # In order to calculate the INFORM metric, we look at the NAME and TRAINID spans because these are the only
            # ones that identify a venue, search for the NAME or TRAINID in the current system response       
            match_detail[current_domain] = {"provided": list(), "requested": list()}
            if ("NAME" in system_response and current_domain in ["restaurant", "hotel", "attraction"]) or ("TRAINID" in system_response and current_domain == "train"):
                # The INFORM rate metric takes into account just the *last* mention about the particular venue into account
                for tool_call in tool_usage:
                    if current_domain in tool_call["tool_name"] and "query" in tool_call["tool_name"]:
                        tool_result = json.loads(tool_call["tool_result"])
                        match_detail[current_domain]["provided"].append(tool_call["tool_arguments"])
                        if "result" in tool_result and tool_result["result"]:
                            offered_venues[current_domain] = tool_result["result"]
                            if current_domain == "train":
                                offered_venues[current_domain] = [venue["trainID"] for venue in offered_venues[current_domain]]
                            else:
                                offered_venues[current_domain] = [venue["name"] for venue in offered_venues[current_domain]]
            # In order to calculate the SUCCESS metric, we look at the provided requestable slots in the system utterances
            for requestable_slot in requestable_slots_in_goal[current_domain]:
                if requestable_slot in system_response:
                    # We do not want to add the REFERENCE to the set of mentioned slots if it could not have been known. But the MultiWOZ
                    # dataset does not provide any information about booking availability. Thus we need to rely on the ground-truth anotations. 
                    # On the other hand, if the system provides a reference code in the turn where it is not supposed to, the requestable slot 
                    # is not added. So even if the evaluated system does not use the ground-truth booking information during evaluation and 
                    # it always suceeds if it has any database results, it does not affect these metrics. 
                    if requestable_slot == "REFERENCE":
                        if current_domain in booked_domain:
                            provided_requestable_slots[current_domain].add("REFERENCE")               
                    else: 
                        provided_requestable_slots[current_domain].add(requestable_slot)
        for domain in goal:
            # If the crowd worker was instructed to mention the name, the match is being done automatically
            if 'name' in goal[domain]["informable"]:
                offered_venues[domain] = "MATCHED"
            # 'taxi', 'police', 'hospital' are special domains - do not have any database and entity does not need to be provided
            if domain in ["taxi", "police", "hospital"]:
                offered_venues[domain] = "MATCHED"
            # if TRAINID was not requested and train was *not* found, we match the dialog goals, but if TRAINID was not requested
            # and train *was* found we want to keep it in order to check if it is a right train
            if domain == "train" and not offered_venues["train"] and "TRAINID" not in goal["train"]["requestable"]:
                offered_venues[domain] = "MATCHED"
        # Calculate the INFORM rate of this dialog, either +1 or 0
        match = dict()
        for domain in goal:
            match_detail[domain]["requested"].append(goal[domain]["informable"])
            match_domain = False
            if offered_venues[domain] == "MATCHED":
                match_domain = True
            elif domain in ["restaurant", "hotel", "attraction", "train"] and len(offered_venues[domain]) > 0:
                # Get venues from the database that match all the information provided by the user
                goal_venues = self._query_basic(domain, max_retrieval=50, max_return=50, **goal[domain]["informable"])
                goal_venues = json.loads(goal_venues)
                if "result" not in goal_venues or not goal_venues["result"]:
                    goal_venues = []
                else:
                    goal_venues = goal_venues["result"]
                if domain == "train":
                    goal_venues = [venue["trainID"] for venue in goal_venues]
                else:
                    goal_venues = [venue["name"] for venue in goal_venues]
                # Compare the venues that could be offered by the system and the venues that match the information
                # in dialg goals, these two sets do not have to match exactly and there are two ways to compare them:
                # Venues are matching if the goal venues are a super set of the possibly offered venues.
                offered_venues_set = set(offered_venues)
                goal_venues_set = set(goal_venues)
                if set(offered_venues_set).issubset(goal_venues_set):
                    match_domain = True
            match[domain] = float(match_domain)
        # The inform rate +1 if the goal venues are matched for all domains, otherwise 0 
        # match["total"] = float(sum(match.values()) == len(match.keys()))
        # Calculate the SUCCESS rate, either +1 or 0
        success = dict()
        success_detail = dict()
        for domain in goal:
            # If values in sentences are super set of requestables
            provided_and_wanted_slots = provided_requestable_slots[domain] & requestable_slots_in_goal[domain]
            domain_success = len(provided_and_wanted_slots) == len(requestable_slots_in_goal[domain])
            success[domain] = float(domain_success)
            success_detail[domain] = {"requested": requestable_slots_in_goal[domain], "provided": provided_and_wanted_slots}
        # success["total"] = float(sum(success.values()) >= len(success.keys()))
        return match, match_detail, success, success_detail

    def _query_basic(self, domain: str, max_retrieval: int = 10, max_return: int = 3, **query) -> str:
        valid_items = []
        for database_item in self.database[domain]:
            valid = True
            for query_key, query_value in query.items():
                database_value = database_item.get(query_key, None)
                if database_value and query_value:
                    if (domain not in DOMAIN_FINITE_KEYS) or (query_key not in DOMAIN_FINITE_KEYS[domain]) or (query_value in DOMAIN_FINITE_KEYS[domain][query_key]):
                        if query_key in FUZZY_KEYS[domain] and (fuzz.partial_ratio(database_value, query_value) < self.fuzzy_ratio or fuzz.partial_ratio(query_value, database_value) < self.fuzzy_ratio):
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
                    return json.dumps({"result": valid_items[:max_return], "message": "Too many retrieved results! Please query more accurately!"})
                valid_items.append(database_item)
        return json.dumps({"result": valid_items[:max_return]})
    
    def _query_restaurant(
            self,
            area: Optional[str] = None,
            pricerange: Optional[str] = None,
            food: Optional[str] = None,
            name: Optional[str] = None,
        ) -> str:
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
        ) -> str:
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
        ) -> str:
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
        ) -> str:
        return self._query_basic(
            domain = "train",
            day = day,
            departure = departure,
            destination = destination,
            leaveAt = leaveAt,
            arriveBy = arriveBy,
            trainID = trainID,
        )
    
    def _book_basic(self, domain: str, **query) -> str:
        primary_key = PRIMARY_KEYS[domain]["primary_key"]
        other_keys = PRIMARY_KEYS[domain]["other_keys"]
        found = False
        if primary_key is None and (other_keys is None or set(other_keys).issubset(set(query.keys()))):
            found = True
        elif primary_key in query.keys() and (other_keys is None or set(other_keys).issubset(set(query.keys()))):
            for database_item in self.database[domain]:
                database_value = database_item.get(primary_key, None)
                query_value = query[primary_key]
                if (primary_key in FUZZY_KEYS[domain]) and (fuzz.partial_ratio(database_value, query_value) >= self.fuzzy_ratio) or (fuzz.partial_ratio(query_value, database_value) >= self.fuzzy_ratio):
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
        ) -> str:
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
        ) -> str:
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
        ) -> str:
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
        ) -> str:
        booking_result = self._book_basic(
            domain = "taxi",
            departure = departure,
            destination = destination,
            arriveBy = arriveBy,
            leaveAt = leaveAt,
        )
        booking_result = json.loads(booking_result)
        if booking_result["result"]:
            entity = random.choice(self.database["taxi"])
            booking_result["result"]["phone"] = entity["phone"]
            booking_result["result"]["type"] = entity["type"]
        booking_result = json.dumps(booking_result)
        return booking_result