import json
from collections import Counter

# Path to the JSON file
file_path = '/home/xy61/AgentChord/src/agentchord/environment/multiwoz_24_environment/MultiWOZ2.4/data/mwz24/data.json'

# Load the JSON data
with open(file_path, 'r') as file:
    data = json.load(file)

taxi_info_counter = dict()
taxi_info_counter2 = dict()

for dialog_idx, dialog in data.items():
    for domain, domain_content in dialog["goal"].items():
        if domain_content and domain == "taxi":
            for type, slots in domain_content.items():
                if type == "info":
                    if "-".join(sorted(slots.keys())) not in taxi_info_counter:
                        taxi_info_counter["-".join(sorted(slots.keys()))] = list()
                    taxi_info_counter["-".join(sorted(slots.keys()))].append(dialog_idx)

    for dialog_turn in dialog["log"]:
        if dialog_turn["metadata"]:
            for domain, domain_content in dialog_turn["metadata"].items():
                if domain_content and domain == "attraction":
                    for type, slots in domain_content.items():
                        if type == "semi" and "booked" in domain_content["book"] and domain_content["book"]["booked"]:
                            valid_keys = [key for key in slots.keys() if slots[key] not in ["", "not mentioned", "dont care", "none"]]
                            if "-".join(sorted(valid_keys)) not in taxi_info_counter2:
                                taxi_info_counter2["-".join(sorted(valid_keys))] = list()
                            taxi_info_counter2["-".join(sorted(valid_keys))].append(dialog_idx)

# longest_key = max(taxi_info_counter2.keys(), key=len)
# print("The longest key in taxi_info_counter2 is:", longest_key)

print(taxi_info_counter2.keys())