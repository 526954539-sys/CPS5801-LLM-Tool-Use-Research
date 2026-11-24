import json
import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))



from src import experiment_runner
from src.evaluation import evaluate_task


class FakeAdapter:
    def __init__(self):
        self.calls = []

    def dispatch(self, tool_name, arguments):
        self.calls.append((tool_name, arguments))
        return {
            "api_name": tool_name,
            "input": arguments,
            "output": f"realistic-result-for-{tool_name}",
            "exception": None,
        }


class ExperimentRunnerTests(unittest.TestCase):

    def test_few_shot_contains_three_adapter_backed_demonstrations(self):
        adapter = FakeAdapter()
        messages = experiment_runner.few_shot_messages(adapter)

        self.assertEqual(len(adapter.calls), 3)
        self.assertEqual(
            [name for name, _ in adapter.calls],
            ["Calculator", "QueryScene", "Wiki"],
        )
        self.assertEqual(len(messages), 12)

        tool_messages = [
            message for message in messages
            if message["role"] == "tool"
        ]
        self.assertEqual(len(tool_messages), 3)

        for message in tool_messages:
            content = json.loads(message["content"])
            self.assertIsNone(content["exception"])
            self.assertIn("realistic-result-for-", content["output"])

    def test_few_shot_tool_arguments_are_schema_shaped(self):
        adapter = FakeAdapter()
        experiment_runner.few_shot_messages(adapter)

        self.assertEqual(
            adapter.calls,
            [
                ("Calculator", {"formula": "2+2"}),
                ("QueryScene", {"name": "Morning Routine"}),
                ("Wiki", {"keyword": "artificial intelligence"}),
            ],
        )

    def test_prompting_packages_remain_distinct(self):
        self.assertIn("Do not invent tool results", experiment_runner.PACKAGES["P1_ZERO_SHOT_DIRECT"])
        self.assertIn("Briefly state the next action", experiment_runner.PACKAGES["P3_ZERO_SHOT_REACT"])
        self.assertNotIn("demonstration", experiment_runner.PACKAGES["P1_ZERO_SHOT_DIRECT"].lower())
        self.assertNotIn("demonstration", experiment_runner.PACKAGES["P3_ZERO_SHOT_REACT"].lower())

    def test_argument_parser_accepts_json_object_and_dict(self):
        expected = {"formula": "2+2"}
        self.assertEqual(experiment_runner.parse_arguments(expected), expected)
        self.assertEqual(
            experiment_runner.parse_arguments('{"formula":"2+2"}'),
            expected,
        )

    def test_argument_parser_rejects_non_object(self):
        with self.assertRaises((TypeError, json.JSONDecodeError)):
            experiment_runner.parse_arguments("not-json")
        with self.assertRaises(TypeError):
            experiment_runner.parse_arguments(["2+2"])

    def test_evaluator_separates_invalid_arguments_from_executor_defects(self):
        result = evaluate_task(
            [{"goal_id": "calc", "tool_name": "Calculator",
              "arguments": {"formula": "2+2"}}],
            [{"tool_name": "Calculator",
              "arguments": {"expression": "2+2"},
              "dispatch_exception": "ArgumentParseError: invalid",
              "api_response": None}],
        )
        self.assertIn("invalid_model_arguments", result["all_error_labels"])
        self.assertNotIn("adapter_or_executor_exception", result["all_error_labels"])

    def test_evaluator_identifies_valid_argument_executor_exception(self):
        result = evaluate_task(
            [{"goal_id": "calc", "tool_name": "Calculator",
              "arguments": {"formula": "2+2"}}],
            [{"tool_name": "Calculator",
              "arguments": {"formula": "2+2"},
              "dispatch_exception": "RuntimeError: simulated executor defect",
              "api_response": None}],
        )
        self.assertIn("adapter_or_executor_exception", result["all_error_labels"])
        self.assertNotIn("invalid_model_arguments", result["all_error_labels"])


    def test_run_one_records_complete_mock_tool_cycle(self):
        import tempfile
        from types import SimpleNamespace
        from unittest.mock import patch

        task = {
            "user_request": "Calculate (5+6)*3.",
            "user_turns": ["Calculate (5+6)*3."],
            "goals": [
                {
                    "goal_id": "calc",
                    "tool_name": "Calculator",
                    "arguments": {"formula": "(5+6)*3"},
                }
            ],
            "valid_tool_sequences": [["Calculator"]],
            "provenance": "original_api_bank_task",
            "status": "development_candidate",
        }
        config = {"approval_state": "DRAFT_NOT_FROZEN"}

        tool_message = {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "function": {
                        "name": "Calculator",
                        "arguments": {"formula": "(5+6)*3"},
                    }
                }
            ],
        }
        final_message = {
            "role": "assistant",
            "content": "The answer is 33.",
            "tool_calls": [],
        }

        def fake_response(message, prompt_tokens, response_tokens):
            return SimpleNamespace(
                message=message,
                prompt_eval_count=prompt_tokens,
                eval_count=response_tokens,
                total_duration=100,
                load_duration=10,
                prompt_eval_duration=40,
                eval_duration=50,
            )

        class FakeClient:
            def __init__(self):
                self.responses = [
                    fake_response(tool_message, 10, 3),
                    fake_response(final_message, 20, 5),
                ]
                self.request_count = 0

            def list(self):
                return SimpleNamespace(
                    models=[SimpleNamespace(
                        model=experiment_runner.MODEL,
                        digest=experiment_runner.MODEL_DIGEST,
                    )]
                )

            def chat(self, **kwargs):
                self.request_count += 1
                return self.responses.pop(0)

        fake_client = FakeClient()

        class FakeAdapter:
            def dispatch(self, tool_name, arguments):
                self.last_call = (tool_name, arguments)
                return {
                    "api_name": tool_name,
                    "input": arguments,
                    "output": 33.0,
                    "exception": None,
                }

        with tempfile.TemporaryDirectory() as tmp:
            output_path = Path(tmp) / "mock_runs.jsonl"

            with (
                patch.object(experiment_runner, "load_task", return_value=(task, config)),
                patch.object(experiment_runner.ollama, "Client", return_value=fake_client),
                patch.object(experiment_runner, "APIBankAdapter", FakeAdapter),
                patch.object(experiment_runner, "OUTPUT", output_path),
            ):
                experiment_runner.run_one("L1-016", "P1_ZERO_SHOT_DIRECT")

            lines = output_path.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(lines), 1)
            record = json.loads(lines[0])

        self.assertEqual(fake_client.request_count, 2)
        self.assertEqual(record["model_call_count"], 2)
        self.assertEqual(record["total_prompt_tokens"], 30)
        self.assertEqual(record["total_response_tokens"], 8)
        self.assertEqual(record["total_model_tokens"], 38)
        self.assertEqual(len(record["model_requests"]), 2)
        self.assertEqual(len(record["tool_calls"]), 1)
        self.assertEqual(record["observed_tool_sequence"], ["Calculator"])
        self.assertTrue(record["sequence_valid"])
        self.assertEqual(len(record["turn_summaries"]), 1)
        self.assertEqual(record["turn_summaries"][0]["status"], "completed")
        self.assertEqual(record["final_answer"], "The answer is 33.")
        self.assertEqual(
            record["semantic_answer_review"],
            "pending_human_review",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
