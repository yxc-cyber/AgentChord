from agentchord import (
    BaseAgent,
    BaseAgentSystem,
    BaseEnvironment,
    BaseMetaData,
    ModelConfig,
)


class CountDownSystem(BaseAgentSystem):
    def __init__(self, system_name, environment):
        super().__init__(system_name, environment)
        prompt = "You are a helpful agent that recieves a number and returns the value of that number minus one. If that value is zero, return !!! instead."
        model_config = ModelConfig(client_model="openai/gpt-4o-mini")
        self.add_subsystem(BaseAgent("countdown_agent", environment, prompt, model_config))
        self.add_on_completion_action("countdown_agent", "check_stopping_criteria", self._check_stopping_criteria)

    def _check_stopping_criteria(self, metadata: BaseMetaData) -> BaseMetaData:
        if "!!!" in metadata.output:
            self.subsystem_sequence.set_done()
            return metadata
        else:
            return BaseMetaData(input=metadata.output)
    
    def on_finalization(self, meta_data: BaseMetaData) -> str:
        return meta_data.output
    

inital_setup = BaseMetaData(input="Five")
countdown_environment = BaseEnvironment(initial_metadata=inital_setup)
countdown_system = CountDownSystem("countdown_system", countdown_environment)

print("System pipeline:")
print(countdown_system.get_pipeline_description())

print("Run system:")
final_result = countdown_system.run(debug=False, log_name="simple_example2.log", loop=True)
print(final_result)