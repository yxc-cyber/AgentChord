from .base_environment import BaseEnvironment

TERMINATE_DESCRIPTION = {
    "type": "functions",
    "function": {
        "name": "terminate",
        "description": "Terminate the current agent turn and pass information to the next agent.",
        "parameters": {
            "output": {
                "type": "string",
                "description": "The output of the current agent turn."
            },
            "note": {
                "type": "string",
                "description": "A note left for the next agent."
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

    def _terminate(self, output: str, note: str) -> dict:
        return {"output": output, "note": note}