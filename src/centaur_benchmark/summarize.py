"""Copy summaries to artifacts/ and optional bar plots."""

from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd

from centaur_benchmark.config import default_tasks_dir, load_task
from centaur_benchmark.io import repo_root, results_base


def export_run_artifacts(
    task_slug: str,
    run_id: str,
    *,
    dest: Path | None = None,
    plot: bool = True,
) -> Path:
    """
    Copy leaderboard CSVs from results/<task>/<run>/ to artifacts/<task>/<run>/.
    If plot=True, write simple bar charts when leaderboard exists.
    """
    run_src = results_base() / task_slug / run_id
    if not run_src.is_dir():
        raise FileNotFoundError(f"No such run directory: {run_src}")

    dest = dest or (repo_root() / "artifacts" / task_slug / run_id)
    dest.mkdir(parents=True, exist_ok=True)

    for mode in ("augmentation", "automation"):
        lb = run_src / mode / "leaderboard.csv"
        if lb.exists():
            shutil.copy2(lb, dest / f"leaderboard_{mode}.csv")
            if plot:
                _plot_leaderboard(lb, dest / f"win_rate_{mode}.png", title=f"{task_slug} / {run_id} / {mode}")

    # Optional ranked outputs (smaller than raw outputs)
    for mode in ("augmentation", "automation"):
        rk = run_src / mode / "pairwise_ranked.csv"
        if rk.exists():
            shutil.copy2(rk, dest / f"pairwise_ranked_{mode}.csv")

    (dest / "README.txt").write_text(
        f"Exported from results/{task_slug}/{run_id}/\n"
        "Leaderboards and ranked CSVs are suitable to commit; raw outputs stay in results/.\n",
        encoding="utf-8",
    )
    print(f"Artifacts written to {dest}")
    return dest


def _plot_leaderboard(csv_path: Path, png_path: Path, title: str) -> None:
    import matplotlib.pyplot as plt
    import seaborn as sns

    df = pd.read_csv(csv_path)
    if df.empty:
        return
    col = "avg_win_rate" if "avg_win_rate" in df.columns else None
    if not col:
        return
    label_col = "model_label" if "model_label" in df.columns else "model_id"
    plot_df = df.sort_values(col, ascending=False).head(20)
    plt.figure(figsize=(10, max(3, 0.35 * len(plot_df))))
    sns.barplot(data=plot_df, x=col, y=label_col, color="#4C72B0")
    plt.title(title)
    plt.xlabel(col)
    plt.ylabel("")
    plt.tight_layout()
    png_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(png_path, dpi=150)
    plt.close()


def _write_rank_heatmap(rank_matrix: pd.DataFrame, png_path: Path, title: str) -> None:
    import matplotlib.pyplot as plt
    import numpy as np
    import seaborn as sns

    if rank_matrix.empty:
        return
    data = rank_matrix.astype(float)
    vmax = max(3.0, float(np.nanmax(data.values)) if data.size else 3.0)
    plt.figure(figsize=(max(8, 0.55 * data.shape[1]), max(4, 0.45 * data.shape[0])))
    ax = sns.heatmap(
        data,
        annot=True,
        fmt=".2f",
        cmap="RdYlGn_r",
        vmin=1,
        vmax=vmax,
        linewidths=0.5,
        cbar_kws={"label": "Rank (lower is better)"},
    )
    ax.set_title(title)
    ax.set_xlabel("Model")
    ax.set_ylabel("Task")
    plt.tight_layout()
    png_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(png_path, dpi=160)
    plt.close()


def _matrix_from_records(records: pd.DataFrame, mode: str, *, judge: str | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    sub = records[records["mode"] == mode].copy()
    if judge is not None:
        sub = sub[sub["judge_model"] == judge]
    if sub.empty:
        return pd.DataFrame(), pd.DataFrame()
    rank_matrix = sub.pivot_table(index="task_title", columns="model_label", values="rank_value", aggfunc="mean")
    score_matrix = sub.pivot_table(index="task_title", columns="model_label", values="score", aggfunc="mean")
    rank_matrix.loc["Average"] = rank_matrix.mean(axis=0)
    score_matrix.loc["Average"] = score_matrix.mean(axis=0)
    return rank_matrix, score_matrix


def _judge_display_label(judge_model: str) -> str:
    labels = {
        "gpt-4.1": "GPT-4.1",
        "anthropic/claude-opus-4-8": "Claude-Opus-4.8",
        "deepseek-ai/DeepSeek-V3.1": "DeepSeek-V3.1",
    }
    return labels.get(judge_model, judge_model)


def _write_judge_alignment_tables(judge_df: pd.DataFrame, dest: Path) -> None:
    """
    Export pairwise judge-alignment diagnostics for the static dashboard.

    Each scatter row is one model-task-mode entry, with one rank column per
    judge. Missing values are expected when leave-family-out masking excludes a
    judge from scoring a model family.
    """
    if judge_df.empty:
        return

    index_cols = ["task_slug", "task_title", "mode", "model_id", "model_label", "condition"]
    pivot = (
        judge_df.pivot_table(
            index=index_cols,
            columns="judge_model",
            values="rank_value",
            aggfunc="mean",
        )
        .reset_index()
        .rename(columns={"task_slug": "task"})
    )
    pivot.to_csv(dest / "judge_rank_scatter_points.csv", index=False)

    judges = [c for c in pivot.columns if c not in {"task", "task_title", "mode", "model_id", "model_label", "condition"}]
    rows: list[dict] = []
    scopes: list[tuple[str, pd.DataFrame]] = [("all", pivot)]
    scopes.extend((mode, pivot[pivot["mode"] == mode]) for mode in sorted(pivot["mode"].dropna().unique()))

    for scope, scoped in scopes:
        for i, judge_a in enumerate(judges):
            for judge_b in judges[i + 1 :]:
                pair = scoped[[judge_a, judge_b]].dropna()
                n_pairs = int(len(pair))
                if n_pairs >= 2:
                    spearman = float(pair[judge_a].corr(pair[judge_b], method="spearman"))
                    pearson = float(pair[judge_a].corr(pair[judge_b], method="pearson"))
                    mean_abs_gap = float((pair[judge_a] - pair[judge_b]).abs().mean())
                    within_one_share = float(((pair[judge_a] - pair[judge_b]).abs() <= 1).mean())
                else:
                    spearman = pearson = mean_abs_gap = within_one_share = float("nan")

                base = {
                    "scope": scope,
                    "judge_a": _judge_display_label(judge_a),
                    "judge_b": _judge_display_label(judge_b),
                    "n_pairs": n_pairs,
                    "mean_abs_gap": mean_abs_gap,
                    "within_one_share": within_one_share,
                }
                rows.append({**base, "method": "spearman", "correlation": spearman})
                rows.append({**base, "method": "pearson", "correlation": pearson})

    pd.DataFrame(rows).to_csv(dest / "judge_rank_correlation_summary.csv", index=False)


def export_cross_task_matrices(
    run_id: str,
    *,
    task_slugs: list[str] | None = None,
    dest: Path | None = None,
) -> Path:
    task_paths = sorted(default_tasks_dir().glob("*.yaml"))
    tasks = [load_task(p) for p in task_paths]
    if task_slugs:
        wanted = set(task_slugs)
        tasks = [t for t in tasks if t.slug in wanted]
    dest = dest or (repo_root() / "artifacts" / "cross_task" / run_id)
    dest.mkdir(parents=True, exist_ok=True)

    aggregate_records: list[pd.DataFrame] = []
    judge_records: list[pd.DataFrame] = []
    for task in tasks:
        for mode in ("augmentation", "automation"):
            mode_dir = results_base() / task.slug / run_id / mode
            by_judge = mode_dir / "leaderboard_by_judge.csv"
            agg = mode_dir / "leaderboard_aggregate.csv"
            single = mode_dir / "leaderboard.csv"

            if by_judge.exists():
                df = pd.read_csv(by_judge)
                if not df.empty:
                    d = df.copy()
                    d["task_slug"] = task.slug
                    d["task_title"] = task.title
                    d["mode"] = mode
                    d["score"] = d["avg_win_rate"]
                    d["rank_value"] = d["avg_rank"]
                    d["source"] = "panel_judge"
                    judge_records.append(d)

            if agg.exists():
                df = pd.read_csv(agg)
                score_col = "avg_win_rate_across_judges"
                rank_col = "aggregate_rank"
                source = "panel_aggregate"
            elif single.exists():
                df = pd.read_csv(single)
                score_col = "avg_win_rate"
                rank_col = "avg_rank"
                source = "single_judge"
            else:
                continue
            if df.empty:
                continue
            d = df.copy()
            d["task_slug"] = task.slug
            d["task_title"] = task.title
            d["mode"] = mode
            d["score"] = d[score_col]
            d["rank_value"] = d[rank_col] if rank_col in d.columns else d["score"].rank(ascending=False, method="average")
            d["source"] = source
            d["judge_model"] = "aggregate"
            aggregate_records.append(d)

    if not aggregate_records and not judge_records:
        raise FileNotFoundError(f"No leaderboards found for run id {run_id}")

    if aggregate_records:
        all_df = pd.concat(aggregate_records, ignore_index=True)
        all_df.to_csv(dest / "all_leaderboards_long.csv", index=False)
        for mode in ("augmentation", "automation"):
            rank_matrix, score_matrix = _matrix_from_records(all_df, mode)
            if rank_matrix.empty:
                continue
            rank_matrix.to_csv(dest / f"rank_matrix_{mode}_aggregate.csv")
            score_matrix.to_csv(dest / f"win_rate_matrix_{mode}_aggregate.csv")
            _write_rank_heatmap(
                rank_matrix,
                dest / f"rank_heatmap_{mode}_aggregate.png",
                f"Centaur benchmark rankings by task — {mode.replace('_', ' ').title()} (aggregate judges)",
            )

    if judge_records:
        judge_df = pd.concat(judge_records, ignore_index=True)
        judge_df.to_csv(dest / "all_leaderboards_by_judge_long.csv", index=False)
        _write_judge_alignment_tables(judge_df, dest)
        judges = sorted(judge_df["judge_model"].dropna().unique())
        for judge in judges:
            judge_slug = judge.replace("/", "_").replace(":", "_")
            for mode in ("augmentation", "automation"):
                rank_matrix, score_matrix = _matrix_from_records(judge_df, mode, judge=judge)
                if rank_matrix.empty:
                    continue
                rank_matrix.to_csv(dest / f"rank_matrix_{mode}_{judge_slug}.csv")
                score_matrix.to_csv(dest / f"win_rate_matrix_{mode}_{judge_slug}.csv")
                _write_rank_heatmap(
                    rank_matrix,
                    dest / f"rank_heatmap_{mode}_{judge_slug}.png",
                    f"Centaur benchmark — {mode.replace('_', ' ').title()} — judge {judge}",
                )

        # Side-by-side augment vs automate aggregate heatmaps per judge family label
        try:
            import matplotlib.pyplot as plt

            for judge in judges:
                judge_slug = judge.replace("/", "_").replace(":", "_")
                aug_rank, _ = _matrix_from_records(judge_df, "augmentation", judge=judge)
                auto_rank, _ = _matrix_from_records(judge_df, "automation", judge=judge)
                if aug_rank.empty and auto_rank.empty:
                    continue
                fig, axes = plt.subplots(1, 2, figsize=(16, max(4, 0.45 * max(len(aug_rank), len(auto_rank)))))
                for ax, mat, title in (
                    (axes[0], aug_rank, "Augment mode"),
                    (axes[1], auto_rank, "Automate mode"),
                ):
                    if mat.empty:
                        ax.set_visible(False)
                        continue
                    import seaborn as sns

                    sns.heatmap(
                        mat.astype(float),
                        annot=True,
                        fmt=".2f",
                        cmap="RdYlGn_r",
                        vmin=1,
                        vmax=max(3.0, float(mat.max().max())),
                        linewidths=0.5,
                        ax=ax,
                        cbar_kws={"label": "Rank"},
                    )
                    ax.set_title(title)
                fig.suptitle(f"Centaur benchmark rankings by task — judge {judge}")
                plt.tight_layout()
                plt.savefig(dest / f"rank_heatmap_side_by_side_{judge_slug}.png", dpi=160)
                plt.close()
        except Exception:
            pass

    (dest / "README.txt").write_text(
        f"Cross-task matrices for run id {run_id}.\n"
        "rank_matrix_*_aggregate.csv: panel-aggregated ranks across judges; lower is better.\n"
        "rank_matrix_*_<judge>.csv: per-judge ranks; lower is better.\n"
        "rank_heatmap_*.png: visual matrices matching paper-style heatmaps.\n"
        "all_leaderboards_long.csv / all_leaderboards_by_judge_long.csv: source tables.\n",
        encoding="utf-8",
    )
    print(f"Cross-task matrices written to {dest}")
    return dest
