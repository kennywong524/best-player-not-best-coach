# Task definitions (`tasks/*.yaml`)

Each file defines one benchmark task.

## Stable frontier model set (`20260609_stable_frontier_v2`)

All seven tasks share the same assistant, automation, and evaluator lists (see any `tasks/*.yaml`). Key IDs:

| Role | Model IDs |
|------|-----------|
| Worker | `gpt-3.5-turbo` |
| Assistants / automation | `gpt-4.1`, `gpt-5-mini-2025-08-07`, `deepseek-ai/DeepSeek-V3.1`, `o4-mini-2025-04-16`, `o3-mini-2025-01-31`, `openai/gpt-oss-120b`, `anthropic/claude-sonnet-4-6`, `anthropic/claude-opus-4-8`, `google/gemini-3.1-pro` |
| Evaluators | `gpt-4.1`, `anthropic/claude-opus-4-8`, `google/gemini-3.1-pro`, `deepseek-ai/DeepSeek-V3.1` |

`gpt-5-pro` was replaced with **`gpt-5-mini-2025-08-07`** (label `GPT-5-Mini`) after format-stub failures on long prompts.

Token limits:

| Role | `max_tokens` |
|------|----------------|
| Scaffold generation | `700` on all tasks (process-only; keep short) |
| Augmentation worker (`gpt-3.5-turbo`) | **omitted** — use model default |
| Automation | **per-model cap in `runner.py`** (see table below); YAML `automation_model_max_tokens` is only an upper bound |

Automation per-model output caps — **Expected Parrot verified** (`runner.py` applies `min(yaml_cap, model_cap)`). Matches v4 headroom; only caps models where EP rejects 128k.

| Model ID | Cap | Notes |
|----------|-----|-------|
| `gpt-3.5-turbo` | 4,096 | EP hard limit |
| `gpt-4.1` | 32,768 | EP hard limit (128k → HTTP 500) |
| `gpt-5-mini-2025-08-07` | task-specific | Omit where stable; use task-specific caps from `20260612_fresh_rep1` repairs: `12,000` counselling, `6,000` meal planning, `12,000` tax prep, `8,000` travel planning, `8,000` tutoring |
| `deepseek-ai/DeepSeek-V3.1` | 128,000 | EP verified |
| `o4-mini-2025-04-16` | 128,000 | EP verified |
| `o3-mini-2025-01-31` | 128,000 | EP verified (reasoning + output budget) |
| `openai/gpt-oss-120b` | 128,000 | EP verified |
| `anthropic/claude-sonnet-4-6` | 128,000 | EP verified |
| `anthropic/claude-opus-4-8` | 128,000 | EP verified |
| `google/gemini-3.1-pro` | 65,536 | EP rejects 128k |
| *(unknown model)* | 128,000 | Default |

Probe / verify: `python scripts/probe_automation_max_tokens.py` · `... --verify-config`

Optional overrides: `scaffold_model_max_tokens`, `worker_model_max_tokens`, `automation_model_max_tokens` in YAML only if you need an explicit cap.

Working automation token policy from `20260612_fresh_rep1`:

| Scope | Working `max_tokens` policy |
|-------|-----------------------------|
| `gpt-3.5-turbo` | 4,096 |
| `gpt-4.1` | 32,768 |
| `gpt-5-mini-2025-08-07`, `counselling` automation | 12,000 |
| `gpt-5-mini-2025-08-07`, `meal_plan` automation | 6,000 |
| `gpt-5-mini-2025-08-07`, `tax_prep` automation | 12,000 |
| `gpt-5-mini-2025-08-07`, `travel_planning` automation | 8,000 |
| `gpt-5-mini-2025-08-07`, `tutoring` automation | 8,000 |
| `gpt-5-mini-2025-08-07`, other automation tasks | omitted / model default |
| `google/gemini-3.1-pro` | 65,536 provider cap, further limited by task cap |
| `deepseek-ai/DeepSeek-V3.1`, `meal_plan` automation | 6,000 |
| DeepSeek on other tasks, o3-mini, o4-mini, GPT-OSS-120B, Claude Sonnet, Claude Opus | 128,000 provider cap, further limited by task cap |

Task-level request caps in `runner.py`: 16,384 for `counselling`, `market_trends`, and `tutoring`; 24,576 for `operations_research`; 32,768 for `travel_planning` and `meal_plan`; 12,000 for `tax_prep`.

## YAML fields

| Field | Meaning |
|-------|---------|
| `slug` | Directory name under `results/` / `artifacts/` |
| `task_prompt` | Worker-facing task text |
| `scaffold_prompt_template` | Assistant instruction (process-only scaffold) |
| `worker_instruction` | System instruction for the worker under augmentation |
| `automation_worker_instruction` | Shorter instruction when a model solves the task alone |
| `scaffold_model_max_tokens` | Optional cap for scaffold generation (default: omitted except YAML `700`) |
| `worker_model_max_tokens` | Optional cap for augmentation worker (default: omitted) |
| `automation_model_max_tokens` | Optional cap for automation runs (default: omitted) |
| `pairwise_task_context` | Text shown to the judge as “the request” |
| `pairwise_eval_prompt` | Judge agent instruction for JSON pairwise choice, rubric scores, averages, and rationale |
| `models.assistants` | Map of `model_id: display_label` for scaffold generators |
| `models.automation` | Map of models that run the task end-to-end (`null` to skip) |
| `models.evaluators` | Map of evaluator models used for multi-judge panel evaluation |
| `pairwise.n_evals_per_pair` | Repeated blind judgments per pair (default 1; use for judge noise; generation variability comes from `replicates`) |
| `pairwise.n_outer_runs` | Repeat full tournament N times (legacy judge stability; prefer `replicates` for output variability) |
| `replicates` | Independent generation trials per model (default: 5). Each replicate uses a cache-busted API call. |
| `rubric_prompt` | Optional free-text rubric block for `centaur-benchmark rubric` |

## Regenerate from legacy notebooks

```bash
python3 scripts/build_task_yamls.py
```

See also [Adding a new task](../docs/ADDING_A_TASK.md) and the [main pipeline README](../README.md#reproducing-the-stable-frontier-run-20260609_stable_frontier_v2).
