import math
from datetime import datetime

"""Task-level evaluator for API-Bank tool-use experiments.

Call-level errors are multi-label. Primary errors follow ERROR_PRECEDENCE.
Failed attempts at required goals are not automatically unnecessary calls.
Final-answer semantics and tool-sequence validity are reviewed separately.
"""

ERROR_PRECEDENCE = [
    "invalid_tool_selection",
    "invalid_model_arguments",
    "adapter_or_executor_exception",
    "tool_execution_error",
    "missing_required_tool_call",
    "unnecessary_tool_call",
    "task_completion_failure",
]

TOOL_ARGUMENTS = {
    "Calculator": {"formula"},
    "QueryScene": {"name"},
    "SearchEngine": {"keyword"},
    "Wiki": {"keyword"},
    "QueryHistoryToday": {"date"},
    "GetUserToken": {"username", "password"},
    "QueryAlarm": {"token", "time"},
    "AddAgenda": {"token", "content", "time", "location"},
    "AddAlarm": {"token", "time"},
}

KNOWN_TOOLS = set(TOOL_ARGUMENTS)


def _normalise(value):
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, dict):
        return {k: _normalise(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_normalise(v) for v in value]
    return value


def _valid_arguments(tool_name, arguments):
    if not (
        tool_name in KNOWN_TOOLS
        and isinstance(arguments, dict)
        and set(arguments) == TOOL_ARGUMENTS[tool_name]
        and all(isinstance(value, str) for value in arguments.values())
    ):
        return False

    # Match the formats accepted by API-Bank QueryHistoryToday.
    # Its implementation accepts both %m-%d and %Y-%m-%d.
    if tool_name == "QueryHistoryToday":
        date_value = arguments["date"].strip()
        for date_format in ("%m-%d", "%Y-%m-%d"):
            try:
                datetime.strptime(date_value, date_format)
                return True
            except ValueError:
                continue
        return False

    return True


def _matches(call, goal):
    return (
        call.get("tool_name") == goal.get("tool_name")
        and _normalise(call.get("arguments", {}))
        == _normalise(goal.get("arguments", {}))
    )


def _is_argument_parse_error(call):
    error = call.get("dispatch_exception")
    return isinstance(error, str) and error.startswith("ArgumentParseError:")


def _call_succeeded(call):
    name = call.get("tool_name")
    arguments = call.get("arguments")

    if not _valid_arguments(name, arguments):
        return False
    if call.get("dispatch_exception"):
        return False

    response = call.get("api_response")
    if not isinstance(response, dict):
        return False
    if "exception" not in response or response.get("exception"):
        return False

    return True


def _call_error_labels(call):
    labels = set()
    name = call.get("tool_name")
    arguments = call.get("arguments")
    valid_args = _valid_arguments(name, arguments)
    dispatch_error = call.get("dispatch_exception")
    response = call.get("api_response")

    if name not in KNOWN_TOOLS:
        labels.add("invalid_tool_selection")
    elif not valid_args:
        labels.add("invalid_model_arguments")

    if dispatch_error and valid_args:
        labels.add("adapter_or_executor_exception")

    # Do not attribute an API parsing failure to tool execution when
    # the model supplied an invalid argument in the first place.
    if isinstance(response, dict) and response.get("exception") and valid_args:
        labels.add("tool_execution_error")

    return labels


def evaluate_task(required_goals, calls, final_answer=None, expected_result=None):
    """Evaluate goal completion and call errors.

    Goal matching is independent of order. The runner separately evaluates
    observed tool-name order against the task's approved sequence manifest.
    A failed call using a required tool is treated as an attempted goal, not
    automatically as an unnecessary call.
    """
    labels = set()
    matched_call_indices = set()
    attempted_call_indices = set()
    completed_goal_ids = []
    unmatched_goals = list(range(len(required_goals)))
    successful_goal_call_indices = {}

    # First match successful calls with exact required tool and arguments.
    for goal_index in list(unmatched_goals):
        goal = required_goals[goal_index]
        match_index = next(
            (
                i for i, call in enumerate(calls)
                if i not in matched_call_indices
                and _call_succeeded(call)
                and _matches(call, goal)
            ),
            None,
        )
        if match_index is not None:
            matched_call_indices.add(match_index)
            attempted_call_indices.add(match_index)
            completed_goal_ids.append(goal.get("goal_id"))
            successful_goal_call_indices[goal_index] = match_index
            unmatched_goals.remove(goal_index)

    # Associate failed calls with still-unfulfilled goals using the tool name.
    # This prevents malformed arguments, execution faults, or missing responses
    # from being mislabeled as unnecessary when they attempted a required tool.
    for call_index, call in enumerate(calls):
        if call_index in matched_call_indices:
            continue
        if _call_succeeded(call):
            continue
        goal_index = next(
            (
                i for i, goal in enumerate(required_goals)
                if goal.get("tool_name") == call.get("tool_name")
                and (
                    i in unmatched_goals
                    or (
                        i in successful_goal_call_indices
                        and call_index < successful_goal_call_indices[i]
                    )
                )
            ),
            None,
        )
        if goal_index is not None:
            attempted_call_indices.add(call_index)

    for call in calls:
        labels.update(_call_error_labels(call))

    if unmatched_goals:
        labels.add("missing_required_tool_call")

    # Only calls that cannot be associated with any required goal attempt
    # are considered unnecessary.
    for i, call in enumerate(calls):
        if (
            i not in attempted_call_indices
            and call.get("tool_name") in KNOWN_TOOLS
        ):
            labels.add("unnecessary_tool_call")

    if not required_goals or unmatched_goals:
        labels.add("task_completion_failure")

    ordered_labels = [
        label for label in ERROR_PRECEDENCE if label in labels
    ]
    primary = ordered_labels[0] if ordered_labels else "none"

    # Independently verify the native Calculator output when the manifest
    # supplies one unambiguous expected numeric result.
    output_correctness = "not_applicable"
    observed_output = None
    calculator_goal_indices = [
        i for i, goal in enumerate(required_goals)
        if goal.get("tool_name") == "Calculator"
    ]

    if expected_result is not None:
        output_correctness = "not_evaluated"
        if len(calculator_goal_indices) == 1:
            goal_index = calculator_goal_indices[0]
            call_index = successful_goal_call_indices.get(goal_index)
            if call_index is not None:
                response = calls[call_index].get("api_response")
                if isinstance(response, dict):
                    observed_output = response.get("output")
                    numeric_expected = (
                        isinstance(expected_result, (int, float))
                        and not isinstance(expected_result, bool)
                    )
                    numeric_observed = (
                        isinstance(observed_output, (int, float))
                        and not isinstance(observed_output, bool)
                    )
                    if numeric_expected and numeric_observed:
                        output_correctness = (
                            "correct"
                            if math.isclose(
                                float(observed_output),
                                float(expected_result),
                                rel_tol=1e-6,
                                abs_tol=1e-6,
                            )
                            else "incorrect"
                        )

    return {
        "required_goal_count": len(required_goals),
        "completed_goal_count": len(completed_goal_ids),
        "completed_goal_ids": completed_goal_ids,
        "tool_call_count": len(calls),
        "attempted_required_call_count": len(attempted_call_indices),
        "task_success": (
            len(completed_goal_ids) == len(required_goals)
            and len(calls) == len(required_goals)
            and not labels
        ),
        "primary_error": primary,
        "secondary_error_labels": [
            label for label in ordered_labels if label != primary
        ],
        "all_error_labels": ordered_labels,
        "expected_output": expected_result,
        "observed_output": observed_output,
        "output_correctness": output_correctness,
        "final_answer": final_answer,
        "final_answer_review_status": "requires_separate_semantic_review",
    }
