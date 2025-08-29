import json

from .base_environment import BaseEnvironment

TERMINATE_DESCRIPTION = {
    "type": "function",
    "function": {
        "name": "terminate",
        "description": "Terminate the current agent turn and pass information to the next agent.",
        "parameters": {
            "type": "object",
            "properties": {
                "output": {
                    "type": "string",
                    "description": "The output of the current agent turn. Make it as concise as possible."
                },
                "note": {
                    "type": "string",
                    "description": "A note left for the next agent, additional information is necessart to the next agent. Usually should be empty. Make it as concise as possible."
                },
            },
            "additionalProperties": False,
            "required": ["output", "note"]
        }
    }
}

class InnerEnvironment(BaseEnvironment):
    TERMINATE = "terminate"

    def __init__(self, initial_metadata = None):
        super().__init__(initial_metadata)
        self.register_tool(self.TERMINATE, TERMINATE_DESCRIPTION, self._terminate)

    def _terminate(self, output: str, note: str) -> str:
        return json.dumps({"output": str(output), "note": str(note)})