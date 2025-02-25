QUERY_RESTAURANT_DESCRIPTION = {
    "type": "functions",
    "function": {
        "name": "query_restaurant",
        "description": "Query a restaurant.",
        "parameters": {
            "area": {
                "type": "string",
                "description": "The area in which the restaurant is located.",
                "enum": ["centre", "north", "south", "east", "west"]
            },
            "pricerange": {
                "type": "string",
                "description": "The price range of the restaurant.",
                "enum": ["cheap", "moderate", "expensive"]
            },
            "food": {
                "type": "string",
                "description": "The type of food served at the restaurant.",
                "enum": ["international", "indian", "mediterranean", "italian", "vietnamese", "lebanese", "african", "modern european", "french", "european", "portuguese", "japanese", "seafood", "chinese", "turkish", "gastropub", "british", "thai", "spanish", "korean", "north american", "mexican", "asian oriental"]
            },
            "name": {
                "type": "string",
                "description": "The name of the restaurant.",
            },
            "additionalProperties": False
        }
    }
}

QUERY_HOTEL_DESCRIPTION = {
    "type": "functions",
    "function": {
        "name": "query_hotel",
        "description": "Query a hotel.",
        "parameters": {
            "area": {
                "type": "string",
                "description": "The area in which the hotel is located.",
                "enum": ["centre", "north", "south", "east", "west"]
            },
            "pricerange": {
                "type": "string",
                "description": "The price range of the hotel.",
                "enum": ["cheap", "moderate", "expensive"]
            },
            "internet": {
                "type": "string",
                "description": "Whether the hotel has internet.",
                "enum": ["yes", "no"]
            },
            "parking": {
                "type": "string",
                "description": "Whether the hotel has parking.",
                "enum": ["yes", "no"]
            },
            "stars": {
                "type": "string",
                "description": "The number of stars of the hotel.",
                "enum": ["0", "1", "2", "3", "4", "5"]
            },
            "type": {
                "type": "string",
                "description": "The type of the hotel.",
                "enum": ["bed and breakfast", "guesthouse", "hotel"]
            },
            "name": {
                "type": "string",
                "description": "The name of the hotel."
            },
            "additionalProperties": False
        }
    }
}

QUERY_ATTRACTION_DESCRIPTION = {
    "type": "functions",
    "function": {
        "name": "query_attraction",
        "description": "Query an attraction.",
        "parameters": {
            "area": {
                "type": "string",
                "description": "The area in which the attraction is located.",
                "enum": ["centre", "north", "south", "east", "west"]
            },
            "type": {
                "type": "string",
                "description": "The type of the attraction.",
                "enum": ["museum", "swimmingpool", "architecture", "boat", "college", "nightclub", "entertainment", "cinema", "concerthall", "mutliple sports", "park", "theatre"]
            },
            "name": {
                "type": "string",
                "description": "The name of the attraction."
            },
            "additionalProperties": False
        }
    }
}

QUERY_TRAIN_DESCRIPTION = {
    "type": "functions",
    "function": {
        "name": "query_train",
        "description": "Query a train.",
        "parameters": {
            "day": {
                "type": "string",
                "description": "The day on which the train departs.",
                "enum": ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
            },
            "departure": {
                "type": "string",
                "description": "The departure location of the train."
            },
            "destination": {
                "type": "string",
                "description": "The destination location of the train."
            },
            "leaveAt": {
                "type": "string",
                "description": "The time at which the train leaves in the format HH:MM."
            },
            "arriveBy": {
                "type": "string",
                "description": "The time by which the train arrives in the format HH:MM."
            },
            "trainID": {
                "type": "string",
                "description": "The ID of the train."
            },
            "additionalProperties": False
        }
    }
}

BOOK_RESTAURANT_DESCRIPTION = {
    "type": "functions",
    "function": {
        "name": "book_restaurant",
        "description": "Book a restaurant.",
        "parameters": {
            "people": {
                "type": "string",
                "description": "The number of people in the booking."
            },
            "day": {
                "type": "string",
                "description": "The day of the booking.",
                "enum": ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
            },
            "time": {
                "type": "string",
                "description": "The time of the booking in the format HH:MM."
            },
            "name": {
                "type": "string",
                "description": "The name of the restaurant."
            },
            "additionalProperties": False,
            "required": ["people", "day", "time", "name"]
        }
    }
}

BOOK_HOTEL_DESCRIPTION = {
    "type": "functions",
    "function": {
        "name": "book_hotel",
        "description": "Book a hotel.",
        "parameters": {
            "people": {
                "type": "string",
                "description": "The number of people in the booking."
            },
            "day": {
                "type": "string",
                "description": "The day of the booking.",
                "enum": ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
            },
            "stay": {
                "type": "string",
                "description": "The number of days of the booking."
            },
            "name": {
                "type": "string",
                "description": "The name of the hotel."
            },
            "additionalProperties": False,
            "required": ["people", "day", "stay", "name"]
        }
    }
}

BOOK_TAXI_DESCRIPTION = {
    "type": "functions",
    "function": {
        "name": "book_taxi",
        "description": "Book a taxi.",
        "parameters": {
            "departure": {
                "type": "string",
                "description": "The departure location of the taxi."
            },
            "destination": {
                "type": "string",
                "description": "The destination location of the taxi."
            },
            "arriveBy": {
                "type": "string",
                "description": "The time by which the taxi should arrive in the format HH:MM."
            },
            "leaveAt": {
                "type": "string",
                "description": "The time at which the taxi should leave in the format HH:MM."
            },
            "additionalProperties": False,
            "required": ["departure", "destination", "arriveBy", "leaveAt"]
        }
    }
}

BOOK_TRAIN_DESCRIPTION = {
    "type": "functions",
    "function": {
        "name": "book_train",
        "description": "Book a train.",
        "parameters": {
            "people": {
                "type": "string",
                "description": "The number of people in the booking."
            },
            "trainID": {
                "type": "string",
                "description": "The ID of the train."
            },
            "additionalProperties": False,
            "required": ["people", "trainID"]
        }
    }
}