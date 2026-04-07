REPO_URL = "https://github.com/sierra-research/tau2-bench.git"

ACT_DESCRIPTION = {
    "type": "function",
    "function": {
        "name": "act",
        "description": "Send an agent action/message to tau2 gym and advance one environment step.",
        "parameters": {
            "type": "object",
            "properties": {
                "content": {
                    "type": "string",
                    "description": "The action text/message to pass to AgentGymEnv.step(...).",
                }
            },
            "required": ["content"],
        },
    },
}
