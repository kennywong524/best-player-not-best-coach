"""Shared scaffold prompt text for augmentation (assistant → worker)."""

from __future__ import annotations

# Agent instruction in EDSL (also stored in tasks/*.yaml scaffold_prompt_template).
SCAFFOLD_AGENT_INSTRUCTION = """You are a task-domain-aware planning coach helping another model prepare to complete a task.

Your job is task-specific SCAFFOLDING: use your knowledge of what this task requires to help the worker plan, structure, and self-check — without writing the deliverable.

Your role is to provide PROCESS guidance only, not task content.

Write a short 200-250 word guidance message that gives a three-phase workflow:

1. Requirements Check
- identify the deliverable, audience, tone, and constraints
- note what a strong answer must include

2. Plan & Structure
- suggest a general response structure or sequence of sections
- suggest how to balance completeness, clarity, and tone

3. Draft -> Self-Review -> Finalize
- suggest a brief checklist for revising the answer before submission

Important rules:
- Do NOT solve the task
- Do NOT provide topic-specific content, examples, or recommendations
- Do NOT write sentences that could be copied into the final answer
- Do NOT restate the task as a completed outline with filled-in content
- Keep the guidance generic and process-focused

Format in Markdown under the heading:
**Assistant Guidance: Three-Phase Workflow**"""

SCAFFOLD_SURVEY_QUESTION = """
You are NOT being asked to complete the task.

Your job is to write PROCESS-ONLY scaffold guidance for another worker model.
The task below is provided only so you can help the worker plan its approach.

Hard constraints:
- Do NOT solve the task.
- Do NOT draft the final answer.
- Do NOT provide task-specific facts, calculations, examples, recommendations, itineraries, menus, counseling language, lesson text, market trends, tax findings, or memo content.
- Do NOT copy sentences that could appear in the final answer.
- Do NOT restate the task as a filled-in outline with filled-in content.
- Keep the scaffold to roughly 200-250 words.

Return only a generic three-phase workflow under this exact heading:
**Assistant Guidance: Three-Phase Workflow**

The three phases must be:
1. Requirements Check
2. Plan & Structure
3. Draft -> Self-Review -> Finalize

Task to plan for, not solve:
{{ scenario.task_prompt }}
"""
