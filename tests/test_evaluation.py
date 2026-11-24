import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from evaluation import evaluate_task


def goal(tool="Calculator", arguments=None, goal_id="g1"):
    return {
        "goal_id": goal_id,
        "tool_name": tool,
        "arguments": arguments or {"formula": "2+2"},
    }


def successful_call(tool="Calculator", arguments=None, output=4):
    return {
        "tool_name": tool,
        "arguments": arguments or {"formula": "2+2"},
        "dispatch_exception": None,
        "api_response": {"exception": None, "output": output},
    }


class EvaluatorAttributionTests(unittest.TestCase):
    def test_invalid_arguments_attempt_is_not_unnecessary(self):
        result = evaluate_task(
            [goal()],
            [{
                "tool_name": "Calculator",
                "arguments": {"expression": "2+2"},
                "dispatch_exception": "ArgumentParseError: invalid",
                "api_response": None,
            }],
        )
        self.assertIn("invalid_model_arguments", result["all_error_labels"])
        self.assertIn("missing_required_tool_call", result["all_error_labels"])
        self.assertNotIn("unnecessary_tool_call", result["all_error_labels"])
        self.assertNotIn("adapter_or_executor_exception", result["all_error_labels"])

    def test_executor_exception_attempt_is_not_unnecessary(self):
        result = evaluate_task(
            [goal()],
            [{
                "tool_name": "Calculator",
                "arguments": {"formula": "2+2"},
                "dispatch_exception": "RuntimeError: simulated defect",
                "api_response": None,
            }],
        )
        self.assertIn("adapter_or_executor_exception", result["all_error_labels"])
        self.assertIn("missing_required_tool_call", result["all_error_labels"])
        self.assertNotIn("unnecessary_tool_call", result["all_error_labels"])

    def test_missing_response_cannot_complete_goal(self):
        result = evaluate_task(
            [goal()],
            [{
                "tool_name": "Calculator",
                "arguments": {"formula": "2+2"},
                "dispatch_exception": None,
                "api_response": None,
            }],
        )
        self.assertEqual(result["completed_goal_count"], 0)
        self.assertFalse(result["task_success"])
        self.assertIn("missing_required_tool_call", result["all_error_labels"])
        self.assertNotIn("unnecessary_tool_call", result["all_error_labels"])

    def test_response_without_exception_field_cannot_complete_goal(self):
        result = evaluate_task(
            [goal()],
            [{
                "tool_name": "Calculator",
                "arguments": {"formula": "2+2"},
                "dispatch_exception": None,
                "api_response": {"output": 4},
            }],
        )
        self.assertEqual(result["completed_goal_count"], 0)
        self.assertFalse(result["task_success"])

    def test_tool_error_attempt_is_not_unnecessary(self):
        result = evaluate_task(
            [goal("Wiki", {"keyword": "known topic"}, "wiki")],
            [{
                "tool_name": "Wiki",
                "arguments": {"keyword": "unknown topic"},
                "dispatch_exception": None,
                "api_response": {"exception": "lookup failed"},
            }],
        )
        self.assertIn("tool_execution_error", result["all_error_labels"])
        self.assertIn("missing_required_tool_call", result["all_error_labels"])
        self.assertNotIn("unnecessary_tool_call", result["all_error_labels"])

    def test_repeated_failed_attempts_are_not_unnecessary(self):
        failed_call = {
            "tool_name": "Calculator",
            "arguments": {"formula": "2+2"},
            "dispatch_exception": "RuntimeError: simulated defect",
            "api_response": None,
        }
        result = evaluate_task([goal()], [failed_call, failed_call.copy()])

        self.assertEqual(result["completed_goal_count"], 0)
        self.assertEqual(result["attempted_required_call_count"], 2)
        self.assertIn("missing_required_tool_call", result["all_error_labels"])
        self.assertNotIn("unnecessary_tool_call", result["all_error_labels"])

    def test_failed_attempt_before_success_is_a_retry_not_unnecessary(self):
        failed_call = {
            "tool_name": "Calculator",
            "arguments": {"formula": "2+2"},
            "dispatch_exception": None,
            "api_response": {"exception": "temporary failure"},
        }
        result = evaluate_task([goal()], [failed_call, successful_call()])

        self.assertEqual(result["completed_goal_count"], 1)
        self.assertEqual(result["attempted_required_call_count"], 2)
        self.assertIn("tool_execution_error", result["all_error_labels"])
        self.assertNotIn("missing_required_tool_call", result["all_error_labels"])
        self.assertNotIn("unnecessary_tool_call", result["all_error_labels"])
        self.assertFalse(result["task_success"])

    def test_genuinely_extra_call_is_unnecessary(self):
        result = evaluate_task(
            [goal()],
            [
                successful_call(),
                successful_call("Wiki", {"keyword": "unrequested topic"}, {}),
            ],
        )
        self.assertTrue(result["task_success"] is False)
        self.assertIn("unnecessary_tool_call", result["all_error_labels"])

    def test_unknown_tool_is_not_also_unnecessary(self):
        result = evaluate_task(
            [goal()],
            [{
                "tool_name": "NotARealTool",
                "arguments": {},
                "dispatch_exception": None,
                "api_response": None,
            }],
        )
        self.assertIn("invalid_tool_selection", result["all_error_labels"])
        self.assertNotIn("unnecessary_tool_call", result["all_error_labels"])

    def test_multiple_required_goals_can_match_without_order_assumption(self):
        goals = [
            goal("QueryHistoryToday", {"date": "04-21"}, "history"),
            goal("Calculator", {"formula": "(4*2)/(1+3)"}, "calc"),
        ]
        calls = [
            successful_call("Calculator", {"formula": "(4*2)/(1+3)"}, 2),
            successful_call("QueryHistoryToday", {"date": "04-21"}, ["fixture"]),
        ]
        result = evaluate_task(goals, calls)
        self.assertEqual(result["completed_goal_count"], 2)
        self.assertTrue(result["task_success"])
        self.assertEqual(result["all_error_labels"], [])

    def test_invalid_history_date_is_model_argument_error(self):
        result = evaluate_task(
            [goal("QueryHistoryToday", {"date": "04-21"}, "history")],
            [{
                "tool_name": "QueryHistoryToday",
                "arguments": {"date": "April 21st"},
                "dispatch_exception": None,
                "api_response": {
                    "exception": "time data does not match format '%m-%d'"
                },
            }],
        )
        self.assertIn("invalid_model_arguments", result["all_error_labels"])
        self.assertNotIn("tool_execution_error", result["all_error_labels"])
        self.assertNotIn("unnecessary_tool_call", result["all_error_labels"])

    def test_history_api_supported_iso_date_is_valid(self):
        result = evaluate_task(
            [goal("QueryHistoryToday", {"date": "2024-04-21"}, "history")],
            [successful_call(
                "QueryHistoryToday", {"date": "2024-04-21"}, ["fixture"]
            )],
        )
        self.assertTrue(result["task_success"])
        self.assertNotIn("invalid_model_arguments", result["all_error_labels"])

    def test_valid_history_date_api_failure_remains_tool_execution_error(self):
        result = evaluate_task(
            [goal("QueryHistoryToday", {"date": "04-21"}, "history")],
            [{
                "tool_name": "QueryHistoryToday",
                "arguments": {"date": "04-21"},
                "dispatch_exception": None,
                "api_response": {
                    "exception": "The date is not in the database."
                },
            }],
        )
        self.assertIn("tool_execution_error", result["all_error_labels"])
        self.assertNotIn("invalid_model_arguments", result["all_error_labels"])

    def test_missing_required_goal_is_reported(self):
        result = evaluate_task([goal()], [])
        self.assertIn("missing_required_tool_call", result["all_error_labels"])
        self.assertIn("task_completion_failure", result["all_error_labels"])


class OutputCorrectnessTests(unittest.TestCase):
    def test_exact_calculator_output_is_correct(self):
        result = evaluate_task(
            [goal()],
            [successful_call(output=4.0)],
            expected_result=4.0,
        )
        self.assertEqual(result["output_correctness"], "correct")
        self.assertEqual(result["observed_output"], 4.0)

    def test_float_output_uses_documented_tolerance(self):
        result = evaluate_task(
            [goal(arguments={"formula": "25*3+7/3"})],
            [successful_call(
                arguments={"formula": "25*3+7/3"},
                output=77.33333333333333,
            )],
            expected_result=77.333333,
        )
        self.assertEqual(result["output_correctness"], "correct")

    def test_wrong_calculator_output_is_incorrect(self):
        result = evaluate_task(
            [goal()],
            [successful_call(output=5.0)],
            expected_result=4.0,
        )
        self.assertEqual(result["output_correctness"], "incorrect")
        # Output correctness remains separate from call/execution success.
        self.assertTrue(result["task_success"])

    def test_expected_result_without_successful_call_is_not_evaluated(self):
        result = evaluate_task(
            [goal()],
            [],
            expected_result=4.0,
        )
        self.assertEqual(result["output_correctness"], "not_evaluated")

    def test_no_expected_result_is_not_applicable(self):
        result = evaluate_task([goal()], [successful_call()])
        self.assertEqual(result["output_correctness"], "not_applicable")


if __name__ == "__main__":
    unittest.main(verbosity=2)
