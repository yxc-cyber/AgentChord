import json
import os
import random
import re
import string

# Path
ENV_PATH = os.path.dirname(os.path.abspath(__file__))  # src/agentchord/environment/taubench_environment
DATA_PATH = os.path.join(ENV_PATH, "tau-bench")
REPO_URL = "https://github.com/sierra-research/tau-bench"

# Action names used in the TauBench environment
# RESPOND_ACTION_NAME is the tool name used when the agent sends a message to the user.
# The tool argument key for the message content is "output".
RESPOND_ACTION_NAME = "respond"

# TERMINATE_ACTION_NAME is the tool name used to signal task completion without
# modifying the database (e.g., a graceful exit action). It is skipped when
# replaying ground-truth actions onto a fresh environment for comparison.
TERMINATE_ACTION_NAME = "finish_task"

# Tool description for the respond action
RESPOND_TOOL_DESCRIPTION = {
    "type": "function",
    "function": {
        "name": RESPOND_ACTION_NAME,
        "description": "Send a response message to the user.",
        "parameters": {
            "type": "object",
            "properties": {
                "output": {
                    "type": "string",
                    "description": "The message to send to the user.",
                }
            },
            "required": ["output"],
        },
    },
}

# Tool description for the finish_task action
FINISH_TASK_TOOL_DESCRIPTION = {
    "type": "function",
    "function": {
        "name": TERMINATE_ACTION_NAME,
        "description": (
            "Signal that the task is complete. Call this when the user's request has been "
            "fully handled and the conversation should end."
        ),
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
}
