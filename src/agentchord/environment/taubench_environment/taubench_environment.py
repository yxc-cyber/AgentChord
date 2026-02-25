import os
from typing import Iterator, Literal, Optional, Self

from ...metadata import TaubenchMetaData
from ..base_environment import BaseEnvironment
from .utils import (
    FINISH_TASK_TOOL_DESCRIPTION,
    REPO_URL,
    RESPOND_ACTION_NAME,
    RESPOND_TOOL_DESCRIPTION,
    TERMINATE_ACTION_NAME,
)


class TaubenchEnvironment(BaseEnvironment):
    """
    An AgentChord environment that wraps the TauBench benchmark.

    Each instance corresponds to one TauBench task (identified by ``domain``,
    ``task_split``, and ``task_idx``).  The environment exposes all domain tools
    (airline or retail) plus two framework-level tools:

    * ``respond``       – agent sends a message to the user
    * ``finish_task``   – agent signals task completion

    The ``pre_initialize`` class-method loads every task split for both domains
    so that ``iterate_test_cases`` / ``evaluate_test_cases`` can be called
    without any further setup.
    """

    # Class-level caches populated by pre_initialize()
    evaluation_record: dict = {}   # {domain: {task_split: {task_idx: TaubenchMetaData}}}
    tasks: dict = {}               # {domain: {task_split: List[Task]}}
    wiki: dict = {}                # {domain: str}
    rules: dict = {}               # {domain: List[str]}

    @classmethod
    def pre_initialize(cls):
        """
        Pre-initializes the TaubenchEnvironment.

        * Installs ``tau_bench`` if it is not yet available.
        * Loads all tasks, wiki, and rules for the airline and retail domains.
        """
        if not cls.pre_initialized:
            # ------------------------------------------------------------------ #
            # Ensure tau_bench is installed
            # ------------------------------------------------------------------ #
            try:
                import tau_bench  # noqa: F401
            except ImportError:
                print("tau_bench is not installed. Installing tau_bench...")
                os.system(
                    "uv pip install --no-deps "
                    f"git+{REPO_URL}"
                )
                print(
                    "tau_bench installation completed. "
                    "Please restart the environment to use tau_bench."
                )
                raise Exception(
                    "tau_bench installation completed. Please restart the environment."
                )

            # ------------------------------------------------------------------ #
            # Load tasks
            # ------------------------------------------------------------------ #
            from tau_bench.envs.airline.tasks_test import TASKS as AIRLINE_TASKS_TEST
            from tau_bench.envs.retail.tasks_dev import TASKS_DEV as RETAIL_TASKS_DEV
            from tau_bench.envs.retail.tasks_test import TASKS_TEST as RETAIL_TASKS_TEST
            from tau_bench.envs.retail.tasks_train import (
                TASKS_TRAIN as RETAIL_TASKS_TRAIN,
            )

            cls.tasks = {
                "airline": {
                    "test": AIRLINE_TASKS_TEST,
                },
                "retail": {
                    "test": RETAIL_TASKS_TEST,
                    "train": RETAIL_TASKS_TRAIN,
                    "dev": RETAIL_TASKS_DEV,
                },
            }

            # ------------------------------------------------------------------ #
            # Load wiki (policy document) and rules
            # ------------------------------------------------------------------ #
            from tau_bench.envs.airline.rules import RULES as AIRLINE_RULES
            from tau_bench.envs.airline.wiki import WIKI as AIRLINE_WIKI
            from tau_bench.envs.retail.rules import RULES as RETAIL_RULES
            from tau_bench.envs.retail.wiki import WIKI as RETAIL_WIKI

            cls.wiki = {"airline": AIRLINE_WIKI, "retail": RETAIL_WIKI}
            cls.rules = {"airline": AIRLINE_RULES, "retail": RETAIL_RULES}

            cls.pre_initialized = True

    # ---------------------------------------------------------------------- #
    # Iteration helpers
    # ---------------------------------------------------------------------- #

    @classmethod
    def iterate_test_cases(
        cls,
        domain: Literal["airline", "retail"],
        task_split: str = "test",
        random_seed: Optional[int] = None,
        max_tasks: Optional[int] = None,
    ) -> Iterator[Self]:
        """
        Yield one ``TaubenchEnvironment`` instance per task in the requested split.

        Parameters
        ----------
        domain:
            ``"airline"`` or ``"retail"``
        task_split:
            ``"test"``, ``"train"``, or ``"dev"`` (airline only supports ``"test"``)
        random_seed:
            When provided, tasks are yielded in a reproducible shuffled order.
        max_tasks:
            If set, at most this many tasks are yielded.
        """
        cls.pre_initialize()
        task_list = cls.tasks[domain][task_split]
        indices = list(range(len(task_list)))
        if random_seed is not None:
            import random as _random
            rng = _random.Random(random_seed)
            rng.shuffle(indices)
        if max_tasks is not None:
            indices = indices[:max_tasks]
        for task_idx in indices:
            yield cls(domain=domain, task_split=task_split, task_idx=task_idx)

    @classmethod
    def evaluate_test_cases(
        cls,
        domain: Literal["airline", "retail"],
        task_split: str = "test",
    ) -> TaubenchMetaData:
        """
        Aggregate per-task rewards that have been stored by ``evaluate()``.

        Returns a ``TaubenchMetaData`` with ``mean_reward`` and ``total_tasks``
        populated.
        """
        cls.pre_initialize()
        split_records = cls.evaluation_record.get(domain, {}).get(task_split, {})
        total_reward = sum(m.reward for m in split_records.values())
        total_tasks = len(split_records)
        mean_reward = total_reward / max(total_tasks, 1)
        return TaubenchMetaData(
            domain=domain,
            task_split=task_split,
            mean_reward=mean_reward,
            total_tasks=total_tasks,
            reward=mean_reward,
        )

    # ---------------------------------------------------------------------- #
    # Instance construction
    # ---------------------------------------------------------------------- #

    def __init__(
        self,
        domain: Literal["airline", "retail"] = "retail",
        task_split: str = "test",
        task_idx: int = 0,
    ):
        """
        Parameters
        ----------
        domain:
            ``"airline"`` or ``"retail"``
        task_split:
            ``"test"``, ``"train"``, or ``"dev"``
        task_idx:
            Zero-based index of the task within the split.
        """
        super().__init__()

        self.domain = domain
        self.task_split = task_split
        self.task_idx = task_idx

        # ------------------------------------------------------------------ #
        # Load a fresh copy of the database for this task instance
        # ------------------------------------------------------------------ #
        if domain == "airline":
            from tau_bench.envs.airline.data import load_data as _load
        else:
            from tau_bench.envs.retail.data import load_data as _load

        self.data = _load()

        # ------------------------------------------------------------------ #
        # Register domain tools
        # ------------------------------------------------------------------ #
        if domain == "airline":
            from tau_bench.envs.airline.tools import ALL_TOOLS as _ALL_TOOLS
        else:
            from tau_bench.envs.retail.tools import ALL_TOOLS as _ALL_TOOLS

        for tool_idx, tool in enumerate(_ALL_TOOLS):
            tool_description = tool.get_info()
            function_name = tool_description["function"]["name"]
            # Skip the "think" scratch-pad tool – it does not modify state and
            # is not needed in the AgentChord evaluation loop.
            if function_name == "think":
                continue
            self.register_tool(
                function_name,
                tool_description,
                lambda tool_idx=tool_idx, _tools=_ALL_TOOLS, **kwargs: _tools[
                    tool_idx
                ].invoke(self.data, **kwargs),
            )

        # ------------------------------------------------------------------ #
        # Register framework-level tools (respond + finish_task)
        # ------------------------------------------------------------------ #
        self.register_tool(
            RESPOND_ACTION_NAME,
            RESPOND_TOOL_DESCRIPTION,
            lambda output="": output,  # Simply echoes back; user sim is external
        )
        self.register_tool(
            TERMINATE_ACTION_NAME,
            FINISH_TASK_TOOL_DESCRIPTION,
            lambda: (self.set_done() or "Task marked as complete."),
        )

        # ------------------------------------------------------------------ #
        # Build the initial metadata that the agent system will receive
        # ------------------------------------------------------------------ #
        task = self.__class__.tasks[domain][task_split][task_idx]

        # Compose the note field from the domain wiki and any explicit rules
        wiki_content = self.__class__.wiki.get(domain, "")
        rules_content = self.__class__.rules.get(domain, [])
        note_parts = []
        if wiki_content:
            note_parts.append(wiki_content)
        if rules_content:
            note_parts.append("\n".join(rules_content))
        note = "\n\n".join(note_parts)

        self.set_initial_metadata(
            TaubenchMetaData(
                input=task.instruction,
                note=note,
                instruction=task.instruction,
                task_idx=task_idx,
                domain=domain,
                task_split=task_split,
            )
        )

    # ---------------------------------------------------------------------- #
    # State accessors
    # ---------------------------------------------------------------------- #

    def get_database_state(self) -> dict:
        """Return the current (possibly modified) database state."""
        return self.data

    # ---------------------------------------------------------------------- #
    # Evaluation
    # ---------------------------------------------------------------------- #

    def init_evaluation_record(self):
        """Ensure a slot exists in the class-level evaluation record."""
        if self.domain not in self.__class__.evaluation_record:
            self.__class__.evaluation_record[self.domain] = {}
        if self.task_split not in self.__class__.evaluation_record[self.domain]:
            self.__class__.evaluation_record[self.domain][self.task_split] = {}

    def evaluate(self, metadata: TaubenchMetaData) -> TaubenchMetaData:
        """
        Compute the binary reward for a completed task and store the result.

        Parameters
        ----------
        metadata : TaubenchMetaData
            The metadata object filled in by the agent (``tool`` list populated
            with the agent's tool calls throughout the task).

        Returns
        -------
        TaubenchMetaData
            The same ``metadata`` object with ``reward`` set.
        """
        self.init_evaluation_record()

        task = self.__class__.tasks[self.domain][self.task_split][self.task_idx]
        ground_truth_actions = [
            {"tool_name": a.name, "tool_arguments": a.kwargs} for a in task.actions
        ]
        ground_truth_outputs = task.outputs

        reward = self._compute_reward(
            database_state=self.get_database_state(),
            actions=metadata.tool,
            domain=self.domain,
            ground_truth_actions=ground_truth_actions,
            ground_truth_outputs=ground_truth_outputs,
        )

        metadata.reward = reward
        metadata.task_idx = self.task_idx
        metadata.domain = self.domain
        metadata.task_split = self.task_split
        metadata.instruction = task.instruction

        self.__class__.evaluation_record[self.domain][self.task_split][
            self.task_idx
        ] = metadata
        return metadata

    # ---------------------------------------------------------------------- #
    # Core reward logic (mirrors tau_bench_env.py's classmethod evaluate)
    # ---------------------------------------------------------------------- #

    @staticmethod
    def _compute_reward(
        database_state: dict,
        actions: list,
        domain: str,
        ground_truth_state: Optional[dict] = None,
        ground_truth_actions: Optional[list] = None,
        ground_truth_outputs: Optional[list] = None,
    ) -> float:
        """
        Compute the binary task reward.

        The reward is 1.0 only when *both* conditions are satisfied:

        1. The final database state matches the ground-truth state (obtained
           either directly or by replaying the ground-truth actions on a fresh
           environment).
        2. Every required output string is found in at least one of the agent's
           ``respond`` tool calls (if ``ground_truth_outputs`` is provided).

        Parameters
        ----------
        database_state:
            The database state produced by the agent.
        actions:
            List of tool-call dicts (``tool_name``, ``tool_arguments``,
            ``tool_result``) recorded during the agent's run.
        domain:
            ``"airline"`` or ``"retail"``.
        ground_truth_state:
            Pre-computed ground-truth database state.  When *None*,
            ``ground_truth_actions`` must be provided instead.
        ground_truth_actions:
            Sequence of ground-truth tool-call dicts used to derive the
            ground-truth state when ``ground_truth_state`` is *None*.
        ground_truth_outputs:
            Required output substrings that must appear in the agent's
            ``respond`` calls.
        """
        from tau_bench.envs.base import consistent_hash, to_hashable
        from tau_bench.types import RewardActionInfo

        if ground_truth_state is None and ground_truth_actions is None:
            raise ValueError(
                "Either ground_truth_state or ground_truth_actions must be provided."
            )

        # ------------------------------------------------------------------ #
        # Derive ground-truth database state by replaying actions if needed
        # ------------------------------------------------------------------ #
        if ground_truth_state is None:
            gt_env = TaubenchEnvironment(domain=domain)
            for action in ground_truth_actions:
                if action["tool_name"] not in (
                    RESPOND_ACTION_NAME,
                    TERMINATE_ACTION_NAME,
                ):
                    gt_env.apply_tool(
                        tool_name=action["tool_name"],
                        tool_arguments=action["tool_arguments"],
                    )
            ground_truth_state = gt_env.get_database_state()

        # ------------------------------------------------------------------ #
        # Check database state equality
        # ------------------------------------------------------------------ #
        data_hash = consistent_hash(to_hashable(database_state))
        ground_truth_hash = consistent_hash(to_hashable(ground_truth_state))
        info = RewardActionInfo(
            r_actions=data_hash == ground_truth_hash,
            gt_data_hash=ground_truth_hash,
        )
        reward = 1.0 if info.r_actions else 0.0

        # ------------------------------------------------------------------ #
        # Check required output substrings
        # ------------------------------------------------------------------ #
        if ground_truth_outputs and len(ground_truth_outputs) > 0:
            for required_output in ground_truth_outputs:
                found = any(
                    action["tool_name"] == RESPOND_ACTION_NAME
                    and required_output.lower()
                    in action["tool_arguments"]
                    .get("output", "")
                    .lower()
                    .replace(",", "")
                    for action in actions
                )
                if not found:
                    reward = 0.0
                    break

        return reward