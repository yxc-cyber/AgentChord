import json
import os
import pickle
import dotenv

from agentchord import (
    BaseAgent,
    BaseAgentSystem,
    ModelConfig,
    TaubenchEnvironment,
    TaubenchMetaData,
)

# evaluation configuration
CLIENT_MODEL = "openai/Qwen3-32B-FP8"
USER_CLIENT_MODEL = "openai/gpt-4o-mini"
LOG_NAME = "taubench_evaluation_example.log"
TARGET_DIR = "examples/taubench_examples"

# Configuration for the models
config = ModelConfig(
    client_model=CLIENT_MODEL,
    base_url=dotenv.get_key(dotenv.find_dotenv(), "Proxy_API_BASE"),
    api_key=dotenv.get_key(dotenv.find_dotenv(), "Proxy_API_KEY"),
    temperature=0.0,
    enable_thinking=False,
)
user_config = ModelConfig(
    client_model=USER_CLIENT_MODEL,
    api_key=dotenv.get_key(dotenv.find_dotenv(), "OPENAI_API_KEY"),
    temperature=0.0,
)

prompt = """
As an airline agent, you can help users book, modify, or cancel flight reservations.

- Before taking any actions that update the booking database (booking, modifying flights, editing baggage, upgrading cabin class, or updating passenger information), you must list the action details and obtain explicit user confirmation (yes) to proceed.

- You should not provide any information, knowledge, or procedures not provided by the user or available tools, or give subjective recommendations or comments.

- You should only make one tool call at a time, and if you make a tool call, you should not respond to the user simultaneously. If you respond to the user, you should not make a tool call at the same time.

- You should deny user requests that are against this policy.

- You should transfer the user to a human agent if and only if the request cannot be handled within the scope of your actions.
""".strip()

class TaubenchAirlineSystem(BaseAgentSystem):
    def __init__(self, system_name: str, environment: TaubenchEnvironment, maximum_loops: int = 10, log_name: str = ""):
        super().__init__(
            system_name=system_name,
            environment=environment,
            maximum_loops=maximum_loops,
            log_name=log_name
        )
        ResponderAgent = BaseAgent(
            system_name="responder_agent",
            environment=environment,
            prompt=prompt,
            tools=["book_reservation", "cancel_reservation", "get_reservation_details", "get_user_details", "list_all_airports", "search_direct_flight", "search_onestop_flight", "send_certificate", "update_reservation_baggages", "update_reservation_flights", "update_reservation_passengers"],
            model_config=config,
            maximum_loops=maximum_loops,
            log_name=log_name
        )
        self.add_subsystem(ResponderAgent)
        self.add_on_completion_action("responder_agent", "respond", self._respond)

    def on_initialization(self) -> TaubenchMetaData:
        metadata = self.environment.get_initial_metadata()
        dialogue_history_list = list()
        system_responses_num = len(metadata.responses)
        assert system_responses_num == len(metadata.user_responses) - 1, "The number of system responses should be equal to the number of user responses minus one."
        for idx in range(system_responses_num):
            user_response = metadata.user_responses[idx]
            system_response = metadata.responses[idx]
            dialogue_history_list.append(f"User Response:\n{user_response}")
            dialogue_history_list.append(f"System Response:\n{system_response}")
        dialogue_history_list.append(f"User Response:\n{metadata.user_responses[-1]}")
        dialogue_history = "\n".join(dialogue_history_list)
        metadata.input = f"Dialogue History:\n{dialogue_history}"
        return metadata
    
    def _respond(self, metadata: TaubenchMetaData) -> TaubenchMetaData:
        """
        Finalize the metadata after the responder agent has generated the response.
        """
        metadata.responses.append(metadata.output)
        metadata.user_responses.append(self.environment.user_response(metadata.output))

        print(f"User Response: {metadata.user_responses[-1]}")
        print(f"System Response: {metadata.output}")

        # Initialize all the agents for the next turn
        self.subsystems["responder_agent"].messages_initialization()

        dialogue_history_list = list()
        system_responses_num = len(metadata.responses)
        assert system_responses_num == len(metadata.user_responses) - 1, "The number of system responses should be equal to the number of user responses minus one."
        for idx in range(system_responses_num):
            user_response = metadata.user_responses[idx]
            system_response = metadata.responses[idx]
            dialogue_history_list.append(f"User Response:\n{user_response}")
            dialogue_history_list.append(f"System Response:\n{system_response}")
        dialogue_history_list.append(f"User Response:\n{metadata.user_responses[-1]}")
        dialogue_history = "\n".join(dialogue_history_list)
        metadata.input = f"Dialogue History:\n{dialogue_history}"

        return metadata


def convert_sets(obj):
    if isinstance(obj, set):
        return list(obj)
    raise TypeError(f"Type {type(obj)} not serializable")


taubench_airline_system = TaubenchAirlineSystem("taubench_airline_system", TaubenchEnvironment(domain="airline"), log_name=LOG_NAME)
print(taubench_airline_system.get_pipeline_description())


total_record = dict()
for dialogue_idx, dialogue_case in enumerate(TaubenchEnvironment.iterate_test_cases(domain="airline", task_split="test", random_seed=42, user_model_config=user_config, user_log_name=LOG_NAME)):
    print(f"Dialogue: {dialogue_idx} {dialogue_case.task_idx}")
    taubench_airline_system.set_environment(environment=dialogue_case)
    result = taubench_airline_system.run(loop=True)
    evaluation_result = dialogue_case.evaluate(result)
    print(f"Dialogue State: {evaluation_result.instruction}")
    system_responses_num = len(evaluation_result.responses)
    for idx in range(system_responses_num):
        user_response = evaluation_result.user_responses[idx]
        system_response = evaluation_result.responses[idx]
        print(f"User Response: {user_response}")
        print(f"System Response: {system_response}")
    print(f"Reward: {evaluation_result.reward}")
    print(f"Reward Details: {evaluation_result.reward_details}")
    total_record[f"dialogue_{dialogue_case.task_idx}"] = {
        "system_responses": evaluation_result.responses,
        "user_responses": evaluation_result.user_responses,
        "instruction": evaluation_result.instruction,
        "reward": evaluation_result.reward,
        "reward_details": evaluation_result.reward_details,
    }
    if dialogue_idx == 99:
        break


evaluation_record = TaubenchEnvironment.evaluation_record
evaluation_result = TaubenchEnvironment.evaluate_test_cases(domain="airline", task_split="test")
evaluation_result.to_json(file_name=f"experiments/taubench_experiments/[Qwen3-32B]_[baseline]_[Act]/airline/evaluation_result.json")
with open(f"experiments/taubench_experiments/[Qwen3-32B]_[baseline]_[Act]/airline/total_record.json", "w", encoding="utf-8") as f:
    json.dump(total_record, f, ensure_ascii=False, indent=2, default=convert_sets)
with open(f"experiments/taubench_experiments/[Qwen3-32B]_[baseline]_[Act]/airline/total_record.pkl", "wb") as f:
    pickle.dump(evaluation_record, f)