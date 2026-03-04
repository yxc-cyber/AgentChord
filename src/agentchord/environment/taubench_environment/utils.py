import os

# Path
ENV_PATH = os.path.dirname(os.path.abspath(__file__))  # src/agentchord/environment/taubench_environment
DATA_PATH = os.path.join(ENV_PATH, "tau-bench")
REPO_URL = "https://github.com/sierra-research/tau-bench"

# Prompts
USER_PROMPT_TEMPLATE = """
You are a user simulator for a task-oriented dialogue system. You will interact with an agent that tries to complete a task based on your instructions. Your goal is to provide responses that help the agent understand and complete the task successfully.
Here is your profile:
{profile}
""".strip()

USER_INIT_MESSAGE = """
(Empty. I am waiting for your message.)
"""