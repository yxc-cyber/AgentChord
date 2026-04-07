import json
import os
from importlib import import_module
from pathlib import Path
from typing import Iterator, List, Optional, Self, Union

from ...metadata import TaubenchMetaData
from ...model import ModelConfig
from ..base_environment import BaseEnvironment
from .utils import (
    ACT_DESCRIPTION,
    REPO_URL
)


class Tau2benchEnvironment(BaseEnvironment):
    """
    AgentChord wrapper for tau2bench environments.

    This wrapper prefers using the benchmark's gym-style environment interface
    and falls back to generic adapters for slightly different upstream APIs.
    """

    evaluation_record: dict = {}
    tasks: dict = {}
    domains_root = Path(__file__).parent / "domains"

    @classmethod
    def pre_initialize(cls):
        if not cls.pre_initialized:
            try:
                import tau2  # noqa: F401
            except ImportError:
                print("tau2_bench is not installed. Installing tau2_bench...")
                os.system(
                    "uv pip install --no-deps "
                    f"git+{REPO_URL}"
                )
                print(
                    "tau2_bench installation completed. "
                    "Please restart the environment to use tau2_bench."
                )
                raise Exception(
                    "tau2_bench installation completed. Please restart the environment."
                )

            gym_agent_module = import_module("tau2.gym.gym_agent")
            cls._agent_gym_env_cls = getattr(gym_agent_module, "AgentGymEnv", None)
            if cls._agent_gym_env_cls is None:
                raise ImportError(
                    "Failed to import AgentGymEnv from tau2.gym.gym_agent. "
                )

            cls.tasks = cls._discover_tasks_by_domain_split()

            cls.pre_initialized = True

    @classmethod
    def iterate_test_cases(
        cls,
        domain: str,
        task_split: str = "test",
        random_seed: Optional[int] = None,
        task_indices: Optional[List[Union[str, int]]] = None,
        user_model_config: Optional[ModelConfig] = None,
        user_log_name: str = "",
    ) -> Iterator[Self]:
        cls.pre_initialize()

        if task_indices is None:
            task_indices = cls._discover_task_ids(domain=domain, task_split=task_split)
        else:
            task_indices = [str(task_id) for task_id in task_indices]

        if random_seed is not None:
            import random as _random

            rng = _random.Random(random_seed)
            task_indices = list(task_indices)
            rng.shuffle(task_indices)

        for task_id in task_indices:
            yield cls(
                domain=domain,
                task_split=task_split,
                task_id=task_id,
                user_model_config=user_model_config,
                user_log_name=user_log_name,
            )

    @classmethod
    def evaluate_test_cases(
        cls,
        domain: str,
        task_split: str = "test",
        task_indices: Optional[Union[List[Union[str, int]], str, int]] = None,
    ) -> TaubenchMetaData:
        cls.pre_initialize()

        split_records = cls.evaluation_record.get(domain, {}).get(task_split, {})
        filtered = {}
        allowed_task_ids = None
        if task_indices is not None:
            if isinstance(task_indices, list):
                allowed_task_ids = {str(task_id) for task_id in task_indices}
            else:
                allowed_task_ids = {str(task_indices)}
        for idx, record in split_records.items():
            if allowed_task_ids is None or str(idx) in allowed_task_ids:
                filtered[idx] = record

        total_reward = sum(m.reward for m in filtered.values())
        total_tasks = len(filtered)
        mean_reward = total_reward / max(total_tasks, 1)

        return TaubenchMetaData(
            mean_reward=mean_reward,
            total_tasks=total_tasks,
            reward=mean_reward,
        )

    def __init__(
        self,
        domain: str = "retail",
        task_split: str = "test",
        task_id: Union[str, int] = "0",
        user_model_config: Optional[ModelConfig] = None,
        user_log_name: str = "",
    ):
        from ...agent_system import Input
        from ...gbc_object import GBC

        super().__init__()
        self.__class__.pre_initialize()

        self.domain = domain
        self.task_split = task_split
        self.task_id = str(task_id)
        self.user_model_config = user_model_config
        self.user_log_name = user_log_name
        self.last_info = {}
        self.last_reward = 0.0
        self.last_terminated = False
        self.last_truncated = False

        self.env = self._build_env(
            domain=domain,
            task_split=task_split,
            task_id=self.task_id,
            user_model_config=user_model_config,
        )
        initial_obs, initial_info = self._safe_reset()
        self.last_observation = initial_obs
        self.last_info = initial_info

        # Gym already simulates the user; the agent only needs one action entrypoint.
        self.register_tool("act", ACT_DESCRIPTION, self._act)

        instruction = self._extract_instruction(initial_info)
        self.set_initial_metadata(
            TaubenchMetaData(
                input=GBC(str(initial_obs), subject=Input()),
                instruction=instruction,
                wiki=str(initial_info.get("wiki", "")),
                rules=str(initial_info.get("rules", "")),
                user_responses=[str(initial_obs)],
            )
        )

    @classmethod
    def _discover_task_ids(cls, domain: str, task_split: str) -> List[str]:
        task_buckets = cls.tasks if cls.tasks else cls._discover_tasks_by_domain_split()
        split_tasks = task_buckets.get(domain, {}).get(task_split)
        if not isinstance(split_tasks, list) or not split_tasks:
            available_domains = list(task_buckets.keys())
            available_splits = list(task_buckets.get(domain, {}).keys())
            raise ValueError(
                f"No split tasks found for domain='{domain}', task_split='{task_split}'. "
                f"Available domains: {available_domains}. Available splits for domain: {available_splits}."
            )
        return [str(task_id) for task_id in split_tasks]

    @classmethod
    def _discover_tasks_by_domain_split(cls) -> dict:
        if cls.tasks:
            return cls.tasks

        task_buckets: dict = {}
        if cls.domains_root.exists():
            for split_file in cls.domains_root.glob("*/split_tasks.json"):
                domain = split_file.parent.name
                with open(split_file, "r", encoding="utf-8") as f:
                    split_map = json.load(f)
                task_buckets[domain] = {
                    split_name: [str(task_id) for task_id in task_ids]
                    for split_name, task_ids in split_map.items()
                    if isinstance(task_ids, list)
                }

        cls.tasks = task_buckets
        return cls.tasks

    @classmethod
    def _build_env(
        cls,
        domain: str,
        task_split: str,
        task_id: str,
        user_model_config: Optional[ModelConfig] = None,
    ):
        kwargs = {
            "domain": domain,
            "task_id": str(task_id),
            "solo_mode": False,
            "all_messages_as_observation": True,
        }
        if user_model_config is not None and user_model_config.client_model:
            kwargs["user_llm"] = user_model_config.client_model

        return cls._agent_gym_env_cls(**kwargs)

    def _safe_reset(self):
        reset_output = self.env.reset(seed=0)

        if isinstance(reset_output, tuple):
            if len(reset_output) == 2:
                return reset_output[0], reset_output[1] or {}
            if len(reset_output) > 2:
                return reset_output[0], {}
        return reset_output, {}

    def _extract_instruction(self, info: dict) -> str:
        if isinstance(info, dict):
            for key in ("instruction", "task_instruction", "goal"):
                if key in info and info[key]:
                    return str(info[key])

        task_attr = getattr(self.env, "task", None)
        if task_attr is not None and hasattr(task_attr, "instruction"):
            return str(task_attr.instruction)

        return ""

    def _step_with_action(self, action_text: str) -> str:
        observation, reward, terminated, truncated, info = self.env.step(action_text)
        self.last_observation = observation
        self.last_reward = float(reward)
        self.last_terminated = bool(terminated)
        self.last_truncated = bool(truncated)
        self.last_info = info if isinstance(info, dict) else {}
        if self.last_terminated or self.last_truncated:
            self.set_done()
        return str(observation)

    def _act(self, content: str) -> str:
        observation = self._step_with_action(str(content))
        return json.dumps({"output": observation})

    def init_evaluation_record(self):
        if self.domain not in self.__class__.evaluation_record:
            self.__class__.evaluation_record[self.domain] = {}
        if self.task_split not in self.__class__.evaluation_record[self.domain]:
            self.__class__.evaluation_record[self.domain][self.task_split] = {}
        self.__class__.evaluation_record[self.domain][self.task_split][self.task_id] = TaubenchMetaData()

    def evaluate(self, metadata: TaubenchMetaData) -> TaubenchMetaData:
        self.init_evaluation_record()

        reward = metadata.reward
        if reward == 0.0:
            reward = self.last_reward

        if reward == 0.0 and isinstance(self.last_info, dict):
            for key in ("reward", "final_reward", "score"):
                if key in self.last_info:
                    reward = float(self.last_info[key])
                    break
        if reward == 0.0 and isinstance(self.last_info, dict):
            reward_info = self.last_info.get("reward_info")
            if isinstance(reward_info, dict):
                for key in ("reward", "score"):
                    if key in reward_info:
                        reward = float(reward_info[key])
                        break

        metadata.reward = float(reward)
        metadata.reward_details = dict(self.last_info) if isinstance(self.last_info, dict) else {}
        metadata.instruction = metadata.instruction or self._extract_instruction(self.last_info)

        self.__class__.evaluation_record[self.domain][self.task_split][self.task_id] = metadata
        return metadata

    def user_response(self, message: Optional[str] = None) -> str:
        # Gym already contains the user simulator. Sending an agent message is a gym step.
        if message is None:
            return str(self.last_observation)
        return self._step_with_action(str(message))
