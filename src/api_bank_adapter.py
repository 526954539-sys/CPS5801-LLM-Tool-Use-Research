"""Safe adapter for the five read-only API-Bank tools used in the pilot."""
import copy
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
API_BANK_ROOT = PROJECT_ROOT / "data/raw/DAMO-ConvAI/api-bank"

if str(API_BANK_ROOT) not in sys.path:
    sys.path.insert(0, str(API_BANK_ROOT))

from apis.calculator import Calculator
from apis.query_scene import QueryScene
from apis.search_engine import SearchEngine
from apis.wiki import Wiki
from apis.query_history_today import QueryHistoryToday


def load_database(name):
    path = API_BANK_ROOT / "init_database" / f"{name}.json"
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "Calculator",
            "description": "Calculate an arithmetic formula using the API-Bank calculator.",
            "parameters": {
                "type": "object",
                "properties": {"formula": {"type": "string"}},
                "required": ["formula"],
                "additionalProperties": False
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "QueryScene",
            "description": "Look up devices in a named smart-home scene in the local API-Bank dataset.",
            "parameters": {
                "type": "object",
                "properties": {"name": {"type": "string"}},
                "required": ["name"],
                "additionalProperties": False
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "SearchEngine",
            "description": "Search the local API-Bank BM25 document collection. This is not live web search.",
            "parameters": {
                "type": "object",
                "properties": {"keyword": {"type": "string"}},
                "required": ["keyword"],
                "additionalProperties": False
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "Wiki",
            "description": "Look up an exact topic in the local API-Bank knowledge base. This is not live Wikipedia.",
            "parameters": {
                "type": "object",
                "properties": {"keyword": {"type": "string"}},
                "required": ["keyword"],
                "additionalProperties": False
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "QueryHistoryToday",
            "description": "Look up historical events by month and day in the local API-Bank dataset. It does not distinguish years.",
            "parameters": {
                "type": "object",
                "properties": {"date": {"type": "string"}},
                "required": ["date"],
                "additionalProperties": False
            }
        }
    }
]


class APIBankAdapter:
    def __init__(self):
        # Each adapter instance gets its own copies. These five tools are used read-only.
        self.databases = {
            "Scenes": copy.deepcopy(load_database("Scenes")),
            "SearchEngine": copy.deepcopy(load_database("SearchEngine")),
            "Wiki": copy.deepcopy(load_database("Wiki")),
            "History": copy.deepcopy(load_database("History")),
        }

        self.tools = {
            "Calculator": Calculator(),
            "QueryScene": QueryScene(copy.deepcopy(self.databases["Scenes"])),
            "SearchEngine": SearchEngine(copy.deepcopy(self.databases["SearchEngine"])),
            "Wiki": Wiki(copy.deepcopy(self.databases["Wiki"])),
            "QueryHistoryToday": QueryHistoryToday(copy.deepcopy(self.databases["History"])),
        }

        self.expected_args = {
            "Calculator": {"formula"},
            "QueryScene": {"name"},
            "SearchEngine": {"keyword"},
            "Wiki": {"keyword"},
            "QueryHistoryToday": {"date"},
        }

    def dispatch(self, tool_name, arguments):
        if tool_name not in self.tools:
            raise ValueError(f"Unknown tool: {tool_name}")

        if not isinstance(arguments, dict):
            raise TypeError("Tool arguments must be a JSON object")

        expected = self.expected_args[tool_name]
        if set(arguments) != expected:
            raise ValueError(
                f"Invalid arguments for {tool_name}: "
                f"expected keys {sorted(expected)}, received {sorted(arguments)}"
            )

        if not all(isinstance(value, str) for value in arguments.values()):
            raise TypeError("All arguments for these five tools must be strings")

        # Preserve the API-Bank native argument names and native response structure.
        return self.tools[tool_name].call(**arguments)


if __name__ == "__main__":
    adapter = APIBankAdapter()

    tests = [
        ("Calculator", {"formula": "(5+6)*3"}),
        ("QueryScene", {"name": "Morning Routine"}),
        ("SearchEngine", {"keyword": "the sustainable development"}),
        ("Wiki", {"keyword": "artificial intelligence"}),
        ("QueryHistoryToday", {"date": "04-21"}),
    ]

    print("API-BANK READ-ONLY ADAPTER TEST")
    passed = 0

    for name, arguments in tests:
        result = adapter.dispatch(name, arguments)
        ok = result.get("api_name") == name and result.get("exception") is None
        print(
            f"{'PASS' if ok else 'FAIL'}: {name} | "
            f"exception={result.get('exception')} | "
            f"output_type={type(result.get('output')).__name__}"
        )
        if ok:
            passed += 1

    print(f"RESULT: {passed}/{len(tests)} tools passed")

    try:
        adapter.dispatch("Calculator", {"expression": "2+2"})
    except (ValueError, TypeError):
        print("PASS: non-native argument name rejected")
    else:
        print("FAIL: non-native argument name was accepted")

    try:
        adapter.dispatch("UnknownTool", {})
    except ValueError:
        print("PASS: unknown tool rejected")
    else:
        print("FAIL: unknown tool was accepted")
