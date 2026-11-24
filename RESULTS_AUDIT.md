# Formal Results Audit
126 valid records; 0 malformed. 63 records/model; 42/package; 18/task; 42/repetition. 126 unique run IDs and design combinations; no missing/unexpected combinations. All formal and FROZEN.

Strict success 21/126 (16.7%). Llama P1/P2/P3: 3/21 each (14.3%). Qwen P1: 3/21 (14.3%); P2: 6/21 (28.6%); P3: 3/21 (14.3%).

Task successes: L1-017 3/18; L1-116 0/18; L2-058 0/18; L2-061 0/18; L2-006 0/18; L3-1 18/18; L3-2 0/18.

Overlapping failure-label occurrences: missing required call 83; invalid arguments 54; unnecessary call 48; tool execution error 21; invalid tool selection 0; adapter/executor exception 0. These are multi-label occurrences, not unique failed runs.

All 126 semantic reviews remain pending human review.
