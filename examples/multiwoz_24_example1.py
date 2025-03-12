from agentchord import (
    BaseAgent,
    BaseAgentSystem,
    BaseEnvironment,
    BaseMetaData,
    ModelConfig,
    Multiwoz24Environment,
)


class Multiwoz24System(BaseAgentSystem):
    # Todo
    pass

for dialogue_case in Multiwoz24Environment().iterate_test_cases():
    for turn_case in dialogue_case.iterate_turns():
        # Todo
        pass