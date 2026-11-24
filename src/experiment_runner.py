"""Controlled Ollama runner for the API-Bank prompting-package pilot."""

import argparse
import json
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import ollama

from .api_bank_adapter import APIBankAdapter, TOOL_SCHEMAS
from .evaluation import evaluate_task


MODEL = "llama3.1:8b"
MODEL_DIGEST = "46e0c10c039e019119339687c3c1757cc81b9da49709a3b3924863ba87ca666e"
MODEL_DIGESTS = {
    "llama3.1:8b": MODEL_DIGEST,
    "qwen2.5:7b": "845dbda0ea48ed749caafd9e6037047aa19acfcfd82e704d7ca97d631a0b697e",
}
TEMPERATURE = 0
MAX_TOOL_ROUNDS_PER_TURN = 8
OUTPUT = Path("results/api_bank_pilot_runs.jsonl")

PACKAGES = {
    "P1_ZERO_SHOT_DIRECT": (
        "You are an assistant that completes tasks using the available tools. "
        "Use a tool whenever it is needed. Supply arguments matching the tool "
        "schema exactly. Do not invent tool results. After receiving tool "
        "results, answer the user concisely. Do not make unnecessary tool calls."
    ),
    "P2_FEW_SHOT_DIRECT": (
        "You are an assistant that completes tasks using the available tools. "
        "Use a tool whenever it is needed. Supply arguments matching the tool "
        "schema exactly. Do not invent tool results. After receiving tool "
        "results, answer the user concisely. Do not make unnecessary tool calls."
    ),
    "P3_ZERO_SHOT_REACT": (
        "You are an assistant that solves tasks through observable action steps. "
        "Briefly state the next action when useful, call the appropriate tool, "
        "inspect its result, and then answer. Do not reveal private chain-of-thought. "
        "Use exact schema-compliant arguments and avoid unnecessary calls."
    ),
}



def load_task(task_id, manifest_path=None):
    if manifest_path is not None:
        config_path = Path(manifest_path)
        config = json.loads(config_path.read_text(encoding="utf-8"))
        if config.get("approval_state") != "FROZEN":
            raise RuntimeError("Formal manifest is not FROZEN.")
        if config.get("manifest_role") != "test":
            raise RuntimeError("Formal runs require the frozen test manifest.")
        matches = [t for t in config["tasks"] if t.get("task_id") == task_id]
        if len(matches) != 1:
            raise ValueError(f"Expected exactly one task with ID {task_id}")
        task = matches[0]
        if task.get("status") != "frozen_test":
            raise ValueError(f"Task {task_id} is not frozen_test.")
        if task.get("provenance") not in {
            "original_api_bank_task",
            "original_api_bank_task_with_reconstructed_tool_arguments",
        }:
            raise ValueError(f"Task {task_id} has an unsupported test-task provenance.")
    else:
        config = json.loads(
            Path("configs/development_tasks_v1.json").read_text(encoding="utf-8")
        )
        if config.get("approval_state") != "DRAFT_NOT_FROZEN":
            raise RuntimeError("Unexpected development config approval state.")
        matches = [t for t in config["tasks"] if t.get("task_id") == task_id]
        if len(matches) != 1:
            raise ValueError(f"Expected exactly one task with ID {task_id}")
        task = matches[0]
        if task.get("provenance") != "original_api_bank_task":
            raise ValueError("Pilot runner only permits original API-Bank tasks.")
        if task.get("status") != "core_candidate":
            raise ValueError("Task is not a core candidate.")

    if not task.get("goals") or not task.get("valid_tool_sequences"):
        raise ValueError("Task is missing goals or valid tool sequences.")
    return task, config

def few_shot_messages(adapter):
    """Build three few-shot demonstrations using genuine local API-Bank outputs.

    These are illustrative demonstrations, not benchmark task records.
    Their messages are included in the model prompt and token budget.
    """
    examples = [
        {
            "user": "Calculate 2 + 2.",
            "tool": "Calculator",
            "arguments": {"formula": "2+2"},
            "final": "The result is 4.",
        },
        {
            "user": "Look up the Morning Routine smart-home scene.",
            "tool": "QueryScene",
            "arguments": {"name": "Morning Routine"},
            "final": (
                "The Morning Routine scene includes a smart light, "
                "a coffee maker, and blinds."
            ),
        },
        {
            "user": "Look up artificial intelligence in the local Wiki.",
            "tool": "Wiki",
            "arguments": {"keyword": "artificial intelligence"},
            "final": (
                "The Wiki entry describes artificial intelligence as "
                "intelligence demonstrated by machines and gives examples "
                "such as speech recognition, computer vision, and translation."
            ),
        },
    ]

    messages = []
    for example in examples:
        result = adapter.dispatch(
            example["tool"],
            example["arguments"],
        )
        messages.extend([
            {
                "role": "user",
                "content": example["user"],
            },
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{
                    "function": {
                        "name": example["tool"],
                        "arguments": example["arguments"],
                    }
                }],
            },
            {
                "role": "tool",
                "tool_name": example["tool"],
                "content": json.dumps(
                    result,
                    ensure_ascii=False,
                    default=str,
                ),
            },
            {
                "role": "assistant",
                "content": example["final"],
            },
        ])

    return messages


def to_plain(value):
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if isinstance(value, dict):
        return value
    return value


def parse_arguments(value):
    if isinstance(value, str):
        return json.loads(value)
    if isinstance(value, dict):
        return value
    raise TypeError("Tool arguments must be a JSON object.")


def verify_model_digest(client, model_name=MODEL):
    if model_name not in MODEL_DIGESTS:
        raise ValueError(f"Unsupported model: {model_name}")

    expected_digest = MODEL_DIGESTS[model_name]
    models = client.list().models
    matches = [model for model in models if model.model == model_name]

    if len(matches) != 1:
        raise RuntimeError(
            f"Expected exactly one installed model named {model_name}; "
            f"found {len(matches)}."
        )

    actual_digest = matches[0].digest
    if actual_digest != expected_digest:
        raise RuntimeError(
            f"Model digest mismatch for {model_name}: "
            f"expected {expected_digest}, got {actual_digest}"
        )

    return actual_digest


def run_one(
    task_id,
    package_name,
    hardware_label="unspecified; hardware label not supplied",
    model_name=MODEL,
    manifest_path=None,
    output_path=None,
    experiment_metadata=None,
    seed=42,
    repetition_id=None,
    run_phase="development_pilot",
    print_record=True,
):
    preparation_start = time.perf_counter()
    task, config = load_task(task_id, manifest_path=manifest_path)
    if package_name not in PACKAGES:
        raise ValueError(f"Unknown prompting package: {package_name}")

    client = ollama.Client(host="http://localhost:11434")
    actual_model_digest = verify_model_digest(client, model_name)
    adapter = APIBankAdapter()

    system_prompt = PACKAGES[package_name]
    messages = [{"role": "system", "content": system_prompt}]

    if package_name == "P2_FEW_SHOT_DIRECT":
        messages.extend(few_shot_messages(adapter))

    calls = []
    request_records = []
    model_prompt_tokens = 0
    model_response_tokens = 0
    model_call_count = 0
    task_start = time.perf_counter()
    preparation_elapsed = task_start - preparation_start
    final_answer = ""
    turn_summaries = []

    user_turns = task.get("user_turns") or [task["user_request"]]

    for turn_index, user_text in enumerate(user_turns, start=1):
        messages.append({"role": "user", "content": user_text})
        turn_call_names = []
        turn_start = len(calls)

        turn_status = "completed"
        for round_index in range(1, MAX_TOOL_ROUNDS_PER_TURN + 1):
            request_messages = json.loads(json.dumps(messages, ensure_ascii=False))
            request_start = time.perf_counter()

            response = client.chat(
                model=model_name,
                messages=messages,
                tools=TOOL_SCHEMAS,
                options={"temperature": TEMPERATURE, "seed": seed},
                stream=False,
            )

            request_elapsed = time.perf_counter() - request_start
            response_data = to_plain(response)
            message_data = to_plain(response.message)
            model_call_count += 1

            prompt_tokens = response.prompt_eval_count or 0
            response_tokens = response.eval_count or 0
            model_prompt_tokens += prompt_tokens
            model_response_tokens += response_tokens

            request_records.append({
                "turn_index": turn_index,
                "round_index": round_index,
                "request_messages": request_messages,
                "response_message": message_data,
                "prompt_tokens": prompt_tokens,
                "response_tokens": response_tokens,
                "request_latency_seconds": request_elapsed,
                "total_duration_ns": response.total_duration,
                "load_duration_ns": response.load_duration,
                "prompt_eval_duration_ns": response.prompt_eval_duration,
                "eval_duration_ns": response.eval_duration,
            })

            messages.append(message_data)
            tool_calls = message_data.get("tool_calls") or []

            if not tool_calls:
                final_answer = message_data.get("content") or ""
                break

            for tool_call in tool_calls:
                function = tool_call.get("function") or {}
                tool_name = function.get("name")
                raw_arguments = function.get("arguments", {})

                call_record = {
                    "turn_index": turn_index,
                    "round_index": round_index,
                    "tool_name": tool_name,
                    "raw_arguments": raw_arguments,
                    "arguments": None,
                    "dispatch_exception": None,
                    "api_response": None,
                }

                try:
                    arguments = parse_arguments(raw_arguments)
                    call_record["arguments"] = arguments
                except Exception as exc:
                    call_record["dispatch_exception"] = (
                        f"ArgumentParseError: {type(exc).__name__}: {exc}"
                    )
                    arguments = None

                if arguments is not None:
                    try:
                        result = adapter.dispatch(tool_name, arguments)
                        call_record["api_response"] = result
                    except Exception as exc:
                        call_record["dispatch_exception"] = (
                            f"{type(exc).__name__}: {exc}"
                        )

                calls.append(call_record)
                turn_call_names.append(tool_name)

                if call_record["dispatch_exception"]:
                    tool_content = json.dumps(
                        {"dispatch_exception": call_record["dispatch_exception"]},
                        ensure_ascii=False,
                    )
                else:
                    tool_content = json.dumps(
                        call_record["api_response"],
                        ensure_ascii=False,
                        default=str,
                    )

                messages.append({
                    "role": "tool",
                    "tool_name": tool_name or "unknown",
                    "content": tool_content,
                })

        else:
            final_answer = ""
            turn_status = "maximum_tool_rounds_reached"

        turn_summaries.append({
            "turn_index": turn_index,
            "status": turn_status,
            "user_message": user_text,
            "tool_names": turn_call_names,
            "tool_call_count": len(calls) - turn_start,
            "assistant_last_message": final_answer,
        })

    elapsed = time.perf_counter() - task_start
    end_to_end_elapsed = time.perf_counter() - preparation_start
    evaluation = evaluate_task(task["goals"], calls, final_answer, task.get("expected_result"))

    observed_sequence = [call.get("tool_name") for call in calls]
    valid_sequences = task["valid_tool_sequences"]
    sequence_valid = observed_sequence in valid_sequences

    record = {
        "run_id": str(uuid.uuid4()),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "task_id": task_id,
        "task_provenance": task.get("provenance"),
        "task_status_at_run": task.get("status"),
        "config_approval_state": config.get("approval_state"),
        "prompting_package": package_name,
        "system_prompt": system_prompt,
        "few_shot_demonstrations_included": package_name == "P2_FEW_SHOT_DIRECT",
        "model_tag": model_name,
        "model_digest": actual_model_digest,
        "temperature": TEMPERATURE,
        "seed": 42,
        "user_turns": user_turns,
        "turn_summaries": turn_summaries,
        "model_requests": request_records,
        "tool_calls": calls,
        "observed_tool_sequence": observed_sequence,
        "valid_tool_sequences": valid_sequences,
        "sequence_valid": sequence_valid,
        "final_answer": final_answer,
        "model_call_count": model_call_count,
        "total_prompt_tokens": model_prompt_tokens,
        "total_response_tokens": model_response_tokens,
        "total_model_tokens": model_prompt_tokens + model_response_tokens,
        "total_latency_seconds": round(elapsed, 6),
        "preparation_latency_seconds": round(preparation_elapsed, 6),
        "end_to_end_latency_seconds": round(end_to_end_elapsed, 6),
        "latency_measurement_scope": (
            "total_latency_seconds covers task execution; "
            "preparation_latency_seconds covers task loading, client/adapter setup "
            "and few-shot demonstration preparation; "
            "end_to_end_latency_seconds covers both."
        ),
        "evaluation": evaluation,
        "semantic_answer_review": "pending_human_review",
        "hardware": hardware_label,
        "notes": (
            "Development pilot only; not a frozen or confirmatory experiment. "
            "Few-shot demonstrations are illustrative prompt examples."
        ),
    }


    target_output = Path(output_path) if output_path else OUTPUT
    target_output.parent.mkdir(parents=True, exist_ok=True)

    record["run_phase"] = run_phase
    record["repetition_id"] = repetition_id
    record["experiment_id"] = (
        experiment_metadata.get("experiment_id")
        if experiment_metadata else None
    )
    if experiment_metadata:
        record["experiment_metadata"] = experiment_metadata

    if run_phase == "formal":
        record["notes"] = (
            "Frozen confirmatory test run. Final-answer semantic review "
            "remains separate from automatic tool-goal evaluation."
        )

    with target_output.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")

    if print_record:
        print(json.dumps(record, ensure_ascii=False, indent=2, default=str))

    return record


def run_formal_batch(config_path="configs/formal_experiment_v1.json", dry_run=False):
    config_path = Path(config_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))

    if config.get("approval_state") != "FROZEN":
        raise RuntimeError("Formal experiment config is not FROZEN.")

    test_path = Path(config["test_manifest"])
    test_manifest = json.loads(test_path.read_text(encoding="utf-8"))

    if test_manifest.get("approval_state") != "FROZEN":
        raise RuntimeError("Test manifest is not FROZEN.")
    if test_manifest.get("manifest_role") != "test":
        raise RuntimeError("Manifest role must be test.")
    if test_manifest.get("dataset_commit") != config.get("dataset_commit"):
        raise RuntimeError("Dataset commit differs between config and test manifest.")

    tasks = test_manifest.get("tasks", [])
    if len(tasks) != test_manifest.get("task_count"):
        raise RuntimeError("Test manifest task_count does not match tasks.")
    if not tasks:
        raise RuntimeError("Test manifest contains no tasks.")

    for task in tasks:
        if task.get("status") != "frozen_test":
            raise RuntimeError(f"Task {task.get('task_id')} is not frozen_test.")
        if task.get("provenance") not in {
            "original_api_bank_task",
            "original_api_bank_task_with_reconstructed_tool_arguments",
        }:
            raise RuntimeError(f"Task {task.get('task_id')} has an unsupported test-task provenance.")
        if not task.get("goals") or not task.get("valid_tool_sequences"):
            raise RuntimeError(f"Task {task.get('task_id')} is missing evaluation rules.")

    models = config.get("models", [])
    packages = config.get("prompting_packages", [])
    repetitions = config.get("repetition_ids", [])

    if len(models) != 2:
        raise RuntimeError("Expected exactly two frozen models.")
    for model in models:
        tag = model.get("tag")
        if tag not in MODEL_DIGESTS or model.get("digest") != MODEL_DIGESTS[tag]:
            raise RuntimeError(f"Frozen model digest mismatch or unsupported model: {tag}")

    if not packages or any(package not in PACKAGES for package in packages):
        raise RuntimeError("Formal config contains an unknown prompting package.")
    if len(repetitions) != config.get("repetitions"):
        raise RuntimeError("Repetition IDs do not match configured repetition count.")

    expected = len(tasks) * len(models) * len(packages) * len(repetitions)
    if expected != config.get("expected_run_count"):
        raise RuntimeError(
            f"Expected run count mismatch: calculated {expected}, "
            f"configured {config.get('expected_run_count')}."
        )

    output_path = Path(config["output_jsonl"])
    completed = set()

    if output_path.exists():
        with output_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise RuntimeError(
                        f"Malformed JSONL at line {line_number}; "
                        "inspect output before resuming."
                    ) from exc

                if record.get("run_phase") != "formal":
                    raise RuntimeError(
                        "Formal output file contains a non-formal record."
                    )
                if record.get("experiment_id") != config["experiment_id"]:
                    raise RuntimeError(
                        "Formal output contains a different experiment ID."
                    )

                key = (
                    record.get("model_tag"),
                    record.get("prompting_package"),
                    record.get("task_id"),
                    record.get("repetition_id"),
                )
                if key in completed:
                    raise RuntimeError(f"Duplicate completed run found: {key}")
                completed.add(key)

    valid_keys = {
        (model["tag"], package, task["task_id"], repetition)
        for model in models
        for package in packages
        for task in tasks
        for repetition in repetitions
    }
    unexpected = completed - valid_keys
    if unexpected:
        raise RuntimeError(f"Output contains unexpected run keys: {unexpected}")

    remaining = expected - len(completed)
    print("FORMAL EXPERIMENT:", config["experiment_id"])
    print("TEST TASKS:", len(tasks))
    print("MODELS:", len(models))
    print("PROMPT PACKAGES:", len(packages))
    print("REPETITIONS:", len(repetitions))
    print("EXPECTED RUNS:", expected)
    print("ALREADY COMPLETED:", len(completed))
    print("REMAINING:", remaining)
    print("OUTPUT:", output_path)
    print("EXECUTION:", config.get("execution_policy"))

    if dry_run:
        print("DRY RUN ONLY: no model calls were made.")
        return

    for model in models:
        model_tag = model["tag"]
        print(f"\n===== MODEL: {model_tag} =====")

        for package in packages:
            for task in tasks:
                for repetition in repetitions:
                    key = (model_tag, package, task["task_id"], repetition)
                    if key in completed:
                        continue

                    print(
                        f"Running {len(completed)+1}/{expected}: "
                        f"{model_tag} | {package} | {task['task_id']} | "
                        f"repeat {repetition}",
                        flush=True,
                    )

                    run_one(
                        task["task_id"],
                        package,
                        hardware_label="NVIDIA RTX A5000 24GB VRAM; Windows",
                        model_name=model_tag,
                        manifest_path=test_path,
                        output_path=output_path,
                        experiment_metadata={
                            "experiment_id": config["experiment_id"],
                            "dataset": config["dataset"],
                            "dataset_commit": config["dataset_commit"],
                            "temperature": config["temperature"],
                            "seed": config["seed"],
                        },
                        seed=config["seed"],
                        repetition_id=repetition,
                        run_phase="formal",
                        print_record=False,
                    )
                    completed.add(key)

    print(f"FORMAL BATCH FINISHED: {len(completed)}/{expected} records.")



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--formal-batch", action="store_true")
    parser.add_argument("--formal-config", default="configs/formal_experiment_v1.json")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--task-id")
    parser.add_argument("--package", choices=sorted(PACKAGES))
    parser.add_argument(
        "--model",
        choices=sorted(MODEL_DIGESTS),
        default=MODEL,
        help="Installed Ollama model; defaults to llama3.1:8b.",
    )
    parser.add_argument(
        "--hardware-label",
        default="unspecified; hardware label not supplied",
        help="Explicit hardware description recorded in the run output.",
    )
    args = parser.parse_args()

    if args.formal_batch:
        run_formal_batch(args.formal_config, dry_run=args.dry_run)
        return

    if args.dry_run:
        parser.error("--dry-run requires --formal-batch.")
    if not args.task_id or not args.package:
        parser.error("Single-run mode requires --task-id and --package.")

    run_one(
        args.task_id,
        args.package,
        args.hardware_label,
        model_name=args.model,
    )


if __name__ == "__main__":
    main()
