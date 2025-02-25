import random
import re
import string

# Path
ENV_PATH = "src/agentchord/environment/multiwoz_24_environment"

# Constants for the MultiWOZ 2.4 environment
CLEAN_DOMAINS = ["hotel", "restaurant", "attraction", "train", "taxi"]
FUZZY_KEYS = {  # Useful for querying and booking
    "hotel" : ["name"],
    "attraction" : ["name"],
    "restaurant" : ["name"],
    "train" : ["departure", "destination"],
    "taxi": [],
}
PRIMARY_KEYS = {  # Useful for booking
    "hotel" : {"primary_key": "name", "other_keys": ["people", "day", "stay"]},
    "attraction" : {},
    "restaurant" : {"primary_key": "name", "other_keys": ["people", "day", "time"]},
    "train" : {"primary_key": "trainID", "other_keys": ["people"]},  # Ideally, this should be the primary key. However, the trainID is not always present in the dialogue.
    "taxi" : {"primary_key": None, "other_keys": ["departure", "destination", "arriveBy", "leaveAt"]},
}
DOMAIN_FINITE_KEYS = {
    "restaurant": {
        "food": ["international", "indian", "mediterranean", "italian", "vietnamese", "lebanese", "african", "modern european", "french", "european", "portuguese", "japanese", "seafood", "chinese", "turkish", "gastropub", "british", "thai", "spanish", "korean", "north american", "mexican", "asian oriental"],
        "area": ["centre", "north", "south", "east", "west"],
        "pricerange": ["cheap", "moderate", "expensive"],
        "day": ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"],
    },
    "hotel": {
        "area": ["centre", "north", "south", "east", "west"],
        "internet": ["yes", "no"],
        "parking": ["yes", "no"],
        "pricerange": ["cheap", "moderate", "expensive"],
        "stars": ["0", "1", "2", "3", "4", "5"],
        "type": ["bed and breakfast", "guesthouse", "hotel"],
        "day": ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"],
    },
    "attraction": {
        "area": ["centre", "north", "south", "east", "west"],
        "type": ["museum", "swimmingpool", "architecture", "boat", "college", "nightclub", "entertainment", "cinema", "concerthall", "mutliple sports", "park", "theatre"],
    },
    "train": {
        "day": ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"],
    },
}
ARRIVAL_TIME_KEY = "arriveBy"
DEPARTURE_TIME_KEY = "leaveAt"

# Functions for the MultiWOZ 2.4 environment
def time_str_to_minutes(time_string: str) -> int:
    if not re.match(r"[0-9][0-9]:[0-9][0-9]", time_string):
        return 0
    return int(time_string.split(":")[0]) * 60 + int(time_string.split(":")[1])

def generate_reference_number(length=10):
    characters = string.ascii_uppercase + string.digits
    reference_number = ''.join(random.choice(characters) for _ in range(length))
    return reference_number