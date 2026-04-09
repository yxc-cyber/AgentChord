import itertools
from collections import Counter
from typing import Any, Dict, List, Tuple

from tau_bench.envs.retail.tasks_dev import TASKS_DEV as RETAIL_TASKS_TEST

# ====================================
# 1. Tool grouping for tau-bench retail
# ====================================

def assign_agents_tau_retail(tools):
    tool_to_agent = {}

    for tool in tools:
        if tool in {
            "find_user_id_by_email",
            "find_user_id_by_name_zip",
        }:
            agent = "user_resolution"

        elif tool in {
            "get_user_details",
            "get_order_details",
            "get_product_details",
            "list_all_product_types",
        }:
            agent = "retrieval"

        elif tool in {
            "modify_pending_order_address",
            "modify_pending_order_items",
            "modify_pending_order_payment",
            "cancel_pending_order",
        }:
            agent = "order_modification"

        elif tool in {
            "return_delivered_order_items",
            "exchange_delivered_order_items",
        }:
            agent = "post_delivery"

        elif tool in {
            "modify_user_address",
        }:
            agent = "user_profile"

        else:
            raise ValueError(f"Unmapped tool: {tool}")

        tool_to_agent[tool] = agent

    return tool_to_agent


# ====================================
# 2. Robust trajectory extraction
# ====================================

def _get_attr_or_key(obj: Any, name: str, default=None):
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _extract_tool_name_from_step(step: Any) -> str:
    """
    Tries multiple likely formats for a tool-call step.
    """
    # dict-like cases
    if isinstance(step, dict):
        for key in [
            "tool",
            "tool_name",
            "name",
            "function",
            "action",
        ]:
            if key in step and isinstance(step[key], str):
                return step[key]

        # OpenAI-style function/tool call nesting
        if "function" in step and isinstance(step["function"], dict):
            fn = step["function"].get("name")
            if isinstance(fn, str):
                return fn

        if "tool_call" in step and isinstance(step["tool_call"], dict):
            fn = step["tool_call"].get("name")
            if isinstance(fn, str):
                return fn

    # object-like cases
    for attr in ["tool", "tool_name", "name", "function", "action"]:
        value = getattr(step, attr, None)
        if isinstance(value, str):
            return value

    function_obj = getattr(step, "function", None)
    if function_obj is not None:
        fn_name = getattr(function_obj, "name", None)
        if isinstance(fn_name, str):
            return fn_name

    tool_call_obj = getattr(step, "tool_call", None)
    if tool_call_obj is not None:
        fn_name = getattr(tool_call_obj, "name", None)
        if isinstance(fn_name, str):
            return fn_name

    return None


def _extract_args_from_step(step: Any) -> Dict[str, Any]:
    """
    Tries to pull arguments from a tool-call step.
    """
    if isinstance(step, dict):
        for key in ["args", "arguments", "kwargs", "parameters"]:
            value = step.get(key)
            if isinstance(value, dict):
                return value

        if "function" in step and isinstance(step["function"], dict):
            for key in ["args", "arguments", "parameters"]:
                value = step["function"].get(key)
                if isinstance(value, dict):
                    return value

        if "tool_call" in step and isinstance(step["tool_call"], dict):
            for key in ["args", "arguments", "parameters"]:
                value = step["tool_call"].get(key)
                if isinstance(value, dict):
                    return value

    for attr in ["args", "arguments", "kwargs", "parameters"]:
        value = getattr(step, attr, None)
        if isinstance(value, dict):
            return value

    function_obj = getattr(step, "function", None)
    if function_obj is not None:
        for attr in ["args", "arguments", "parameters"]:
            value = getattr(function_obj, attr, None)
            if isinstance(value, dict):
                return value

    tool_call_obj = getattr(step, "tool_call", None)
    if tool_call_obj is not None:
        for attr in ["args", "arguments", "parameters"]:
            value = getattr(tool_call_obj, attr, None)
            if isinstance(value, dict):
                return value

    return {}


TOOLS_TO_IGNORE = {"calculate", "transfer_to_human_agents"}

def extract_tool_sequence_from_task(task):
    return [
        {
            "tool": action.name,
            "args": action.kwargs or {},
        }
        for action in task.actions
        if action.name not in TOOLS_TO_IGNORE
    ]


def collapse_consecutive_duplicates(seq: List[str]) -> List[str]:
    if not seq:
        return seq
    out = [seq[0]]
    for x in seq[1:]:
        if x != out[-1]:
            out.append(x)
    return out


def load_retail_tasks_with_trajectories() -> List[Dict[str, Any]]:
    """
    Converts RETAIL_TASKS_TEST into a normalized list of examples.
    """
    data = []
    for i, task in enumerate(RETAIL_TASKS_TEST):
        task_id = _get_attr_or_key(task, "task_id", None)
        if task_id is None:
            task_id = _get_attr_or_key(task, "id", f"task_{i}")

        trajectory = extract_tool_sequence_from_task(task)
        if not trajectory:
            continue

        data.append({
            "task_id": task_id,
            "trajectory": trajectory,
        })
    return data


# ====================================
# 3. Statistics
# ====================================

def preprocess_trajectories(
    data: List[Dict[str, Any]],
    collapse_duplicates: bool = True
) -> List[Dict[str, Any]]:
    processed = []
    for ex in data:
        seq = [step["tool"] for step in ex["trajectory"]]
        if collapse_duplicates:
            seq = collapse_consecutive_duplicates(seq)
        processed.append({
            "task_id": ex["task_id"],
            "trajectory": ex["trajectory"],
            "tool_seq": seq
        })
    return processed


def build_direct_transition_counts(tool_seqs: List[List[str]]) -> Counter:
    counts = Counter()
    for seq in tool_seqs:
        for a, b in zip(seq, seq[1:]):
            counts[(a, b)] += 1
    return counts


def build_precedence_counts(tool_seqs: List[List[str]]) -> Counter:
    counts = Counter()
    for seq in tool_seqs:
        for i in range(len(seq)):
            for j in range(i + 1, len(seq)):
                counts[(seq[i], seq[j])] += 1
    return counts


def build_tool_frequency(tool_seqs: List[List[str]]) -> Counter:
    freq = Counter()
    for seq in tool_seqs:
        freq.update(seq)
    return freq


def infer_data_dependencies(raw_data: List[Dict[str, Any]]) -> Counter:
    dep_counts = Counter()

    for ex in raw_data:
        traj = ex["trajectory"]
        arg_keys_seen_per_step = []

        for step in traj:
            args = step.get("args", {}) or {}
            arg_keys_seen_per_step.append(set(args.keys()))

        for i in range(len(traj)):
            tool_i = traj[i]["tool"]
            keys_i = arg_keys_seen_per_step[i]

            for j in range(i + 1, len(traj)):
                tool_j = traj[j]["tool"]
                keys_j = arg_keys_seen_per_step[j]

                if keys_i & keys_j:
                    dep_counts[(tool_i, tool_j)] += 1

    return dep_counts


def add_manual_dependencies(dep_counts: Counter) -> Counter:
    constraints = [
        ("find_user_id_by_email", "get_user_details"),
        ("find_user_id_by_email", "get_order_details"),
        ("find_user_id_by_name_zip", "get_user_details"),
        ("find_user_id_by_name_zip", "get_order_details"),

        ("get_order_details", "modify_pending_order_address"),
        ("get_order_details", "modify_pending_order_items"),
        ("get_order_details", "modify_pending_order_payment"),
        ("get_order_details", "cancel_pending_order"),

        ("get_order_details", "return_delivered_order_items"),
        ("get_order_details", "exchange_delivered_order_items"),
    ]

    for a, b in constraints:
        dep_counts[(a, b)] += 100

    return dep_counts


# ====================================
# 4. Agent order scoring
# ====================================

def project_tools_to_agents(seq: List[str], tool_to_agent: Dict[str, str]) -> List[str]:
    return [tool_to_agent[t] for t in seq]


def order_index_map(order: Tuple[str, ...]) -> Dict[str, int]:
    return {agent: i for i, agent in enumerate(order)}


def score_agent_order(
    processed_data: List[Dict[str, Any]],
    tool_to_agent: Dict[str, str],
    agent_order: Tuple[str, ...],
    dependency_counts: Counter = None,
    dependency_threshold: int = 1
) -> Dict[str, Any]:
    idx = order_index_map(agent_order)

    total_traj = len(processed_data)
    no_backward_count = 0
    total_backward = 0
    total_handoffs = 0

    for ex in processed_data:
        tool_seq = ex["tool_seq"]
        agent_seq = project_tools_to_agents(tool_seq, tool_to_agent)

        backward_here = 0
        handoffs_here = 0

        for a1, a2 in zip(agent_seq, agent_seq[1:]):
            if a1 != a2:
                handoffs_here += 1
            if idx[a2] < idx[a1]:
                backward_here += 1

        total_backward += backward_here
        total_handoffs += handoffs_here
        if backward_here == 0:
            no_backward_count += 1

    coverage = no_backward_count / total_traj if total_traj else 0.0
    avg_backward = total_backward / total_traj if total_traj else 0.0
    avg_handoffs = total_handoffs / total_traj if total_traj else 0.0

    dependency_violations = 0
    if dependency_counts is not None:
        for (tool_a, tool_b), c in dependency_counts.items():
            if c < dependency_threshold:
                continue
            agent_a = tool_to_agent.get(tool_a, "other")
            agent_b = tool_to_agent.get(tool_b, "other")
            if idx[agent_b] < idx[agent_a]:
                dependency_violations += c

    return {
        "agent_order": agent_order,
        "coverage_no_backward": coverage,
        "avg_backward_transitions": avg_backward,
        "avg_handoffs": avg_handoffs,
        "dependency_violations": dependency_violations
    }


def composite_score(
    result: Dict[str, Any],
    alpha: float = 5.0,
    beta: float = 2.0,
    gamma: float = 1.0,
    delta: float = 0.5
) -> float:
    return (
        alpha * result["coverage_no_backward"]
        - beta * result["avg_backward_transitions"]
        - gamma * result["avg_handoffs"]
        - delta * result["dependency_violations"]
    )


def search_best_agent_order(
    processed_data: List[Dict[str, Any]],
    tool_to_agent: Dict[str, str],
    dependency_counts: Counter = None,
    dependency_threshold: int = 1
) -> List[Dict[str, Any]]:
    agents = sorted(set(tool_to_agent.values()))
    results = []

    for order in itertools.permutations(agents):
        r = score_agent_order(
            processed_data=processed_data,
            tool_to_agent=tool_to_agent,
            agent_order=order,
            dependency_counts=dependency_counts,
            dependency_threshold=dependency_threshold
        )
        r["composite_score"] = composite_score(r)
        results.append(r)

    results.sort(key=lambda x: x["composite_score"], reverse=True)
    return results


# ====================================
# 5. Reporting
# ====================================

def print_top_transitions(counter: Counter, top_k: int = 20, title: str = ""):
    if title:
        print(f"\n=== {title} ===")
    for (a, b), c in counter.most_common(top_k):
        print(f"{a:35s} -> {b:35s} : {c}")


def print_agent_assignment(tool_to_agent: Dict[str, str]):
    print("\n=== Tool -> Agent Assignment ===")
    for tool in sorted(tool_to_agent):
        print(f"{tool:35s} -> {tool_to_agent[tool]}")


def print_top_orders(results: List[Dict[str, Any]], top_k: int = 10):
    print("\n=== Top Agent Orders ===")
    for r in results[:top_k]:
        print(
            f"order={r['agent_order']} | "
            f"score={r['composite_score']:.3f} | "
            f"coverage={r['coverage_no_backward']:.3f} | "
            f"avg_backward={r['avg_backward_transitions']:.3f} | "
            f"avg_handoffs={r['avg_handoffs']:.3f} | "
            f"dep_viol={r['dependency_violations']}"
        )


# ====================================
# 6. Main
# ====================================

def main():
    raw_data = load_retail_tasks_with_trajectories()

    if not raw_data:
        print("No trajectories were extracted from RETAIL_TASKS_TEST.")
        print("You likely need to adjust 'extract_tool_sequence_from_task' to match your tau-bench version.")
        print("\nTry printing one task object, e.g.:")
        print("print(RETAIL_TASKS_TEST[0])")
        return

    processed = preprocess_trajectories(raw_data, collapse_duplicates=True)
    tool_seqs = [ex["tool_seq"] for ex in processed]

    direct_counts = build_direct_transition_counts(tool_seqs)
    precedence_counts = build_precedence_counts(tool_seqs)
    dep_counts = infer_data_dependencies(raw_data)
    dep_counts = add_manual_dependencies(dep_counts)

    all_tools = sorted({tool for seq in tool_seqs for tool in seq})
    tool_to_agent = assign_agents_tau_retail(all_tools)

    print(f"Loaded {len(raw_data)} retail tasks with trajectories.")
    print(f"Unique tools found: {len(all_tools)}")

    print_top_transitions(direct_counts, top_k=20, title="Top Direct Tool Transitions")
    print_top_transitions(precedence_counts, top_k=20, title="Top Tool Precedence Pairs")
    print_top_transitions(dep_counts, top_k=20, title="Top Inferred / Manual Dependencies")
    print_agent_assignment(tool_to_agent)

    results = search_best_agent_order(
        processed_data=processed,
        tool_to_agent=tool_to_agent,
        dependency_counts=dep_counts,
        dependency_threshold=2
    )

    print_top_orders(results, top_k=10)

    if results:
        print("\n=== Best Order ===")
        print(results[0])


if __name__ == "__main__":
    main()