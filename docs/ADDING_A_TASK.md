# Adding a new task

1. **Copy a template**  
   Duplicate `tasks/meal_plan.yaml` (or the closest domain) to `tasks/<your_slug>.yaml`.

2. **Edit prompts**  
   - `task_prompt`: what the worker (or automation model) must produce.  
   - `scaffold_prompt_template`: assistant-only process guidance (no direct solutions).  
   - `pairwise_task_context` + `pairwise_eval_prompt`: what the judge sees; the judge must answer with **only** `option_1` or `option_2` (see existing tasks).

3. **Model lists**  
   - `models.assistants`: scaffold models to compare.  
   - `models.automation`: set to `null` if you only care about augmentation; otherwise list end-to-end solvers.

4. **Run**  
   ```bash
   centaur-benchmark run --task your_slug --mode all
   centaur-benchmark judge --task your_slug --run <RUN_ID> --subset both
   centaur-benchmark summarize --task your_slug --run <RUN_ID>
   ```

5. **Optional rubric**  
   Set `rubric_prompt` with explicit numeric scale instructions, then:
   ```bash
   centaur-benchmark rubric --task your_slug --run <RUN_ID>
   ```

6. **Regenerate from notebooks (optional)**  
   If you still prototype in Colab, append your notebook path to `scripts/build_task_yamls.py` and re-run the script, or maintain YAML by hand (recommended once stable).

## Conventions

- **Slug**: lowercase letters, numbers, underscores only.  
- **Git**: keep large CSVs under `results/` (ignored); commit only `artifacts/` exports you need for the paper.
