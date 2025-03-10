import json
import os
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
    reference_number = "".join(random.choice(characters) for _ in range(length))
    return reference_number

# The following function is adapted from uiuc-conversational-ai-lab/multiwoz-helper
# https://github.com/uiuc-conversational-ai-lab/multiwoz-helper/blob/main/utils/delexicalize.py#L19
def prepareSlotValuesIndependent(database: dict) -> dict:
    dic = []
    for domain in CLEAN_DOMAINS:
        try:
            db_json = database[domain]
            for ent in db_json:
                for key, val in ent.items():
                    if val == "?" or val == "free":
                        pass
                    elif key == "address":
                        dic.append((normalize(val), "[" + domain + "_" + "address" + "]"))
                        if "road" in val:
                            val = val.replace("road", "rd")
                            dic.append((normalize(val), "[" + domain + "_" + "address" + "]"))
                        elif "rd" in val:
                            val = val.replace("rd", "road")
                            dic.append((normalize(val), "[" + domain + "_" + "address" + "]"))
                        elif "st" in val:
                            val = val.replace("st", "street")
                            dic.append((normalize(val), "[" + domain + "_" + "address" + "]"))
                        elif "street" in val:
                            val = val.replace("street", "st")
                            dic.append((normalize(val), "[" + domain + "_" + "address" + "]"))
                    elif key == "name":
                        dic.append((normalize(val), "[" + domain + "_" + "name" + "]"))
                        if "b & b" in val:
                            val = val.replace("b & b", "bed and breakfast")
                            dic.append((normalize(val), "[" + domain + "_" + "name" + "]"))
                        elif "bed and breakfast" in val:
                            val = val.replace("bed and breakfast", "b & b")
                            dic.append((normalize(val), "[" + domain + "_" + "name" + "]"))
                        elif "hotel" in val and "gonville" not in val:
                            val = val.replace("hotel", "")
                            dic.append((normalize(val), "[" + domain + "_" + "name" + "]"))
                        elif "restaurant" in val:
                            val = val.replace("restaurant", "")
                            dic.append((normalize(val), "[" + domain + "_" + "name" + "]"))
                    elif key == "postcode":
                        dic.append((normalize(val), "[" + domain + "_" + "postcode" + "]"))
                    elif key == "phone":
                        dic.append((val, "[" + domain + "_" + "phone" + "]"))
                    elif key == "trainID":
                        dic.append((normalize(val), "[" + domain + "_" + "id" + "]"))
                    elif key == "department":
                        dic.append((normalize(val), "[" + domain + "_" + "department" + "]"))
                    elif key == "area":
                        dic.append((normalize(val), "[" + "value" + "_" + "area" + "]"))
                    elif key == "food":
                        dic.append((normalize(val), "[" + "value" + "_" + "food" + "]"))
                    elif key == "pricerange":
                        dic.append((normalize(val), "[" + "value" + "_" + "pricerange" + "]"))
                    elif key == "departure" or key == "destination":
                        dic.append((normalize(val), "[" + "value" + "_" + "place" + "]"))
                    else:
                        pass
        except:
            pass

    for key in ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]:
        dic.append((normalize(key), "[" + "value" + "_" + "day" + "]"))

    return dic

# The following function is adapted from uiuc-conversational-ai-lab/multiwoz-helper
# https://github.com/uiuc-conversational-ai-lab/multiwoz-helper/blob/main/convert_input.py#L9
def delexicalize(system_response: str, delexicalization_map: dict, tool_usages: list) -> str:
    # Normalize system response
    system_response = normalize(system_response)
    system_response = " " + system_response + " "
    # Delexicalize system response
    for key, val in delexicalization_map:
        system_response = system_response.replace(" " + key + " ", " " + val + " ")
    # Delexicalize reference numbers
    for tool_usage in tool_usages:
        tool_result = tool_usage["tool_result"]
        if isinstance(tool_result, str):
            tool_result = json.loads(tool_result)
        tool_name = tool_usage["tool_name"]
        if tool_result.get("result", None) and tool_result["result"].get("reference", None):
            domain = None
            for domain_candidate in CLEAN_DOMAINS:
                if domain_candidate.lower() in tool_name:
                    domain = domain_candidate
                    break
            assert domain is not None
            slot = "reference"
            slot_value = tool_result["result"]["reference"]
            val = "[" + domain + "_" + slot + "]"
            key = normalize(slot_value)
            system_response = system_response.replace(" " + key + " ", " " + val + " ")
            # try reference with hashtag
            key = normalize("#" + slot_value)
            system_response = system_response.replace(" " + key + " ", " " + val + " ")
            # try reference with ref#
            key = normalize("ref#" + slot_value)
            system_response = system_response.replace(" " + key + " ", " " + val + " ")
    # Delexicalize general numbers
    digitpat = re.compile(r"\d+")
    processed_turn_response = re.sub(digitpat, "[value_count]", processed_turn_response)
    system_response = system_response.strip()
    return system_response

# The following function is adapted from uiuc-conversational-ai-lab/multiwoz-helper
# https://github.com/uiuc-conversational-ai-lab/multiwoz-helper/blob/main/utils/nlp.py#L17
def insertSpace(token, text):
    sidx = 0
    while True:
        sidx = text.find(token, sidx)
        if sidx == -1:
            break
        if sidx + 1 < len(text) and re.match("[0-9]", text[sidx - 1]) and re.match("[0-9]", text[sidx + 1]):
            sidx += 1
            continue
        if text[sidx - 1] != " ":
            text = text[:sidx] + " " + text[sidx:]
            sidx += 1
        if sidx + len(token) < len(text) and text[sidx + len(token)] != " ":
            text = text[:sidx + 1] + " " + text[sidx + 1:]
        sidx += 1
    return text

# The following function is adapted from uiuc-conversational-ai-lab/multiwoz-helper
# https://github.com/uiuc-conversational-ai-lab/multiwoz-helper/blob/main/utils/nlp.py#L36
def normalize(text):
    # lower case every word
    text = text.lower()

    # replace white spaces in front and end
    text = re.sub(r"^\s*|\s*$", "", text)

    # hotel domain pfb30
    text = re.sub(r"b&b", "bed and breakfast", text)
    text = re.sub(r"b and b", "bed and breakfast", text)

    # normalize phone number
    ms = re.findall("\(?(\d{3})\)?[-.\s]?(\d{3})[-.\s]?(\d{4,5})", text)
    if ms:
        sidx = 0
        for m in ms:
            sidx = text.find(m[0], sidx)
            if text[sidx - 1] == "(":
                sidx -= 1
            eidx = text.find(m[-1], sidx) + len(m[-1])
            text = text.replace(text[sidx:eidx], "".join(m))

    # normalize postcode
    ms = re.findall("([a-z]{1}[\. ]?[a-z]{1}[\. ]?\d{1,2}[, ]+\d{1}[\. ]?[a-z]{1}[\. ]?[a-z]{1}|[a-z]{2}\d{2}[a-z]{2})",
                    text)
    if ms:
        sidx = 0
        for m in ms:
            sidx = text.find(m, sidx)
            eidx = sidx + len(m)
            text = text[:sidx] + re.sub("[,\. ]", "", m) + text[eidx:]

    # weird unicode bug
    text = re.sub(u"(\u2018|\u2019)", "'", text)

    # replace time and and price
    timepat = re.compile("\d{1,2}[:]\d{1,2}")
    pricepat = re.compile("\d{1,3}[.]\d{1,2}")
    text = re.sub(timepat, " [value_time] ", text)
    text = re.sub(pricepat, " [value_price] ", text)
    #text = re.sub(pricepat2, "[value_price]", text)

    # replace st.
    text = text.replace(";", ",")
    text = re.sub("$\/", "", text)
    text = text.replace("/", " and ")

    # replace other special characters
    text = text.replace("-", " ")
    text = re.sub("[\":\<>@\(\)]", "", text)

    # insert white space before and after tokens:
    for token in ["?", ".", ",", "!"]:
        text = insertSpace(token, text)

    # insert white space for "s
    text = insertSpace("\"s", text)

    # replace it"s, does"t, you"d ... etc
    text = re.sub("^\"", "", text)
    text = re.sub("\"$", "", text)
    text = re.sub("\"\s", " ", text)
    text = re.sub("\s\"", " ", text)
    with open(os.path.join(ENV_PATH, "mapping.pair")) as fin:
        replacements = []
        for line in fin.readlines():
            tok_from, tok_to = line.replace("\n", "").split("\t")
            replacements.append((" " + tok_from + " ", " " + tok_to + " "))
    for fromx, tox in replacements:
        text = " " + text + " "
        text = text.replace(fromx, tox)[1:-1]

    # remove multiple spaces
    text = re.sub(" +", " ", text)

    # concatenate numbers
    tmp = text
    tokens = text.split()
    i = 1
    while i < len(tokens):
        if re.match(u"^\d+$", tokens[i]) and re.match(u"\d+$", tokens[i - 1]):
            tokens[i - 1] += tokens[i]
            del tokens[i]
        else:
            i += 1
    text = " ".join(tokens)

    return text