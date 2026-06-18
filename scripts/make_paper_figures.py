"""Generate paper-ready figures from the static dashboard bundle."""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.font_manager import FontProperties
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from textwrap import fill


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "dashboard" / "dashboard-data.json"
OUT = ROOT / "artifacts" / "paper_figures"
OUT.mkdir(parents=True, exist_ok=True)

TASK_ORDER = [
    "counselling",
    "market_trends",
    "meal_plan",
    "operations_research",
    "tax_prep",
    "travel_planning",
    "tutoring",
]
TASK_LABELS = {
    "counselling": "Counseling",
    "market_trends": "Market Trends",
    "meal_plan": "Menu Planning",
    "operations_research": "Operations Research",
    "tax_prep": "Tax Prep",
    "travel_planning": "Travel Agent",
    "tutoring": "Tutoring",
}
EXCLUDE = {"plain", "GPT-3.5-Turbo"}
EXCLUDE_AUGMENTATION = {"GPT-3.5-Turbo"}  # keep plain worker baseline
EXCLUDE_AUTOMATION = {"plain"}  # keep GPT-3.5-Turbo direct-solver baseline
AUGMENTATION_WORKER_BASELINE = "plain"
AUTOMATION_BASELINE_MODEL = "GPT-3.5-Turbo"
MODEL_DISPLAY = {
    "plain": "GPT-3.5-Turbo (plain)",
    "GPT-3.5-Turbo": "GPT-3.5-Turbo (plain)",
}
MODE_LABELS = {"augmentation": "Augmentation", "automation": "Automation"}
SHORT_LABELS = {
    "GPT-5-Mini": "G5M",
    "GPT-4.1": "G4.1",
    "GPT-O4-Mini": "O4",
    "GPT-O3-Mini": "O3",
    "GPT-OSS-120B": "OSS",
    "DeepSeek-V3.1": "DS",
    "Claude-Opus-4.8": "Opus",
    "Claude-Sonnet-4.6": "Sonnet",
    "Gemini-3.1-Pro": "Gemini",
}
# Newest release first (left → right); plain / GPT-3.5-Turbo baseline is appended last.
MODEL_RELEASE_ORDER = [
    "Claude-Opus-4.8",      # May 28, 2026
    "Gemini-3.1-Pro",       # Feb 19, 2026
    "Claude-Sonnet-4.6",    # Feb 17, 2026
    "DeepSeek-V3.1",        # Aug 21, 2025
    "GPT-5-Mini",           # Aug 7, 2025
    "GPT-OSS-120B",         # Aug 5, 2025
    "GPT-O4-Mini",          # Apr 16, 2025
    "GPT-4.1",              # Apr 14, 2025
    "GPT-O3-Mini",          # Jan 31, 2025
]


def load_data() -> dict:
    return json.loads(DATA_PATH.read_text(encoding="utf-8"))


def display_model_label(model: str) -> str:
    return MODEL_DISPLAY.get(model, model)


def exclude_for_mode(mode: str) -> set[str]:
    return EXCLUDE_AUGMENTATION if mode == "augmentation" else EXCLUDE_AUTOMATION


def rank_rows(data: dict, run_id: str, mode: str, *, exclude: set[str] | None = None) -> pd.DataFrame:
    excluded = exclude_for_mode(mode) if exclude is None else exclude
    rows = [
        r
        for r in data["runs_by_id"][run_id]["aggregate"]
        if r["mode"] == mode and r["model_label"] not in excluded
    ]
    out = []
    for task in TASK_ORDER:
        sub = [r for r in rows if r["task_slug"] == task]
        sub.sort(key=lambda r: (float(r["rank_value"]), -float(r["score"]), r["model_label"]))
        for i, r in enumerate(sub, start=1):
            out.append(
                {
                    "run_id": run_id,
                    "task_slug": task,
                    "task": TASK_LABELS[task],
                    "mode": mode,
                    "model": r["model_label"],
                    "rank": i,
                    "win_rate": float(r["score"]),
                }
            )
    return pd.DataFrame(out)


def all_rank_rows(data: dict, mode: str) -> pd.DataFrame:
    frames = [rank_rows(data, r["id"], mode) for r in data["meta"]["replicate_runs"]]
    return pd.concat(frames, ignore_index=True)


def column_release_order(columns: pd.Index, *, mode: str) -> list[str]:
    ordered = [m for m in MODEL_RELEASE_ORDER if m in columns]
    if mode == "augmentation" and AUGMENTATION_WORKER_BASELINE in columns:
        ordered.append(AUGMENTATION_WORKER_BASELINE)
    if mode == "automation" and AUTOMATION_BASELINE_MODEL in columns:
        ordered.append(AUTOMATION_BASELINE_MODEL)
    ordered.extend(m for m in columns if m not in ordered)
    return ordered


def mean_rank_matrix(data: dict, mode: str) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    df = all_rank_rows(data, mode)
    n_reps = len(data["meta"]["replicate_runs"])
    mean = df.pivot_table(index="task", columns="model", values="rank", aggfunc="mean")
    sd = df.pivot_table(
        index="task",
        columns="model",
        values="rank",
        aggfunc=lambda values: float(np.std(values, ddof=0)),
    ).fillna(0)
    mean = mean.loc[[TASK_LABELS[t] for t in TASK_ORDER]]
    sd = sd.loc[[TASK_LABELS[t] for t in TASK_ORDER]]
    avg_row = pd.DataFrame([mean.mean(axis=0)], index=["Average"])
    sd_avg = pd.DataFrame([df.groupby("model")["rank"].agg(lambda values: float(np.std(values, ddof=0)))], index=["Average"]).fillna(0)
    mean = pd.concat([mean, avg_row])
    sd = pd.concat([sd, sd_avg])
    order = column_release_order(mean.columns, mode=mode)
    return mean[order], sd[order], n_reps


def rank_of_ranks_matrix(mean: pd.DataFrame) -> pd.DataFrame:
    ranked = pd.DataFrame(index=mean.index, columns=mean.columns, dtype=int)
    for idx in mean.index:
        ranked.loc[idx] = mean.loc[idx].rank(method="min", ascending=True).astype(int)
    return ranked


def centaur_cmap() -> LinearSegmentedColormap:
    stops = [
        (37 / 255, 127 / 255, 99 / 255),
        (210 / 255, 232 / 255, 207 / 255),
        (244 / 255, 221 / 255, 124 / 255),
        (222 / 255, 105 / 255, 72 / 255),
        (158 / 255, 35 / 255, 42 / 255),
    ]
    return LinearSegmentedColormap.from_list("centaur_rank", stops, N=256)


def _draw_heatmap_base(
    matrix: pd.DataFrame,
    filename: str,
    *,
    cbar_label: str,
    cell_formatter,
) -> None:
    n_rows, n_cols = matrix.shape
    fig_w = max(13.6, n_cols * 1.32 + 3.2)
    fig_h = max(6.4, n_rows * 0.78 + 2.1)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=220)
    values = matrix.to_numpy(dtype=float)
    im = ax.imshow(values, cmap=centaur_cmap(), vmin=1, vmax=n_cols, aspect="auto")

    ax.set_xticks(np.arange(n_cols))
    ax.set_xticklabels([fill(display_model_label(c), 14) for c in matrix.columns], rotation=0, ha="center", fontsize=10)
    ax.set_yticks(np.arange(n_rows))
    ax.set_yticklabels(matrix.index, fontsize=11)
    ax.tick_params(axis="both", length=0)

    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks(np.arange(-0.5, n_cols, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n_rows, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.1)
    ax.tick_params(which="minor", bottom=False, left=False)

    for i in range(n_rows):
        for j in range(n_cols):
            val = values[i, j]
            color = "white" if val > n_cols * 0.72 else "#1f2433"
            weight = "semibold" if matrix.index[i] == "Average" else "normal"
            for dy, text in cell_formatter(val, i, j):
                ax.text(j, i + dy, text, ha="center", va="center", color=color, fontsize=12.2 if dy == 0 else 9.3, fontweight=weight, alpha=0.86 if dy != 0 else 1.0)
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.015)
    cbar.set_label(cbar_label, fontsize=11)
    cbar.ax.tick_params(labelsize=10)
    fig.subplots_adjust(left=0.055, right=0.955, bottom=0.12, top=0.96)
    for ext in ["png", "pdf"]:
        fig.savefig(OUT / f"{filename}.{ext}", bbox_inches="tight")
    plt.close(fig)


def draw_heatmap_rank_of_ranks(data: dict, mode: str, filename: str) -> None:
    mean, _, _ = mean_rank_matrix(data, mode)
    ranks = rank_of_ranks_matrix(mean)
    n_rows, n_cols = ranks.shape
    fig_w = max(13.6, n_cols * 1.32 + 3.2)
    fig_h = max(6.8, n_rows * 0.88 + 2.1)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=220)
    values = ranks.to_numpy(dtype=float)
    im = ax.imshow(values, cmap=centaur_cmap(), vmin=1, vmax=n_cols, aspect="auto")

    ax.set_xticks(np.arange(n_cols))
    ax.set_xticklabels([fill(display_model_label(c), 14) for c in ranks.columns], rotation=0, ha="center", fontsize=10)
    ax.set_yticks(np.arange(n_rows))
    ax.set_yticklabels(ranks.index, fontsize=11)
    ax.tick_params(axis="both", length=0)

    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks(np.arange(-0.5, n_cols, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n_rows, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.1)
    ax.tick_params(which="minor", bottom=False, left=False)

    for i in range(n_rows):
        for j in range(n_cols):
            val = int(values[i, j])
            color = "white" if val > n_cols * 0.72 else "#1f2433"
            weight = "bold" if ranks.index[i] == "Average" else "semibold"
            ax.text(
                j,
                i,
                str(val),
                ha="center",
                va="center",
                color=color,
                fontsize=18.5,
                fontweight=weight,
            )
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.015, ticks=range(1, n_cols + 1))
    cbar.set_label("Rank", fontsize=11)
    cbar.ax.tick_params(labelsize=10)
    fig.subplots_adjust(left=0.055, right=0.955, bottom=0.12, top=0.96)
    for ext in ["png", "pdf"]:
        fig.savefig(OUT / f"{filename}.{ext}", bbox_inches="tight")
    plt.close(fig)


def draw_heatmap_mean_se(data: dict, mode: str, filename: str) -> None:
    mean, sd, n_reps = mean_rank_matrix(data, mode)
    se = sd / math.sqrt(n_reps)

    def cell_formatter(val: float, i: int, j: int) -> list[tuple[float, str]]:
        err = se.iloc[i, j]
        return [(-0.10, f"{val:.1f}"), (0.23, f"±{err:.1f}")]

    _draw_heatmap_base(mean, filename, cbar_label="Mean rank", cell_formatter=cell_formatter)


def draw_heatmap(data: dict, mode: str, filename: str, title: str) -> None:
    draw_heatmap_mean_se(data, mode, filename)


def average_model_ranks(data: dict) -> pd.DataFrame:
    rows = []
    for mode in ["augmentation", "automation"]:
        df = all_rank_rows(data, mode)
        for model, sub in df.groupby("model"):
            rows.append(
                {
                    "model": model,
                    "mode": mode,
                    "mean_rank": sub["rank"].mean(),
                    "sd_rank": sub["rank"].std(ddof=0),
                }
            )
    wide = pd.DataFrame(rows).pivot(index="model", columns="mode", values="mean_rank").reset_index()
    return wide.dropna(subset=["augmentation", "automation"])


def draw_role_scatter(data: dict) -> None:
    df = average_model_ranks(data)
    df = df.assign(gap=(df["automation"] - df["augmentation"]).abs()).sort_values(
        ["gap", "augmentation"], ascending=[False, True]
    )
    fig, ax = plt.subplots(figsize=(8.4, 5.8), dpi=220)
    y = np.arange(len(df))

    for i, (_, r) in enumerate(df.iterrows()):
        ax.plot(
            [r["automation"], r["augmentation"]],
            [i, i],
            color="#aeb6c2",
            linewidth=1.4,
            zorder=1,
        )
    ax.scatter(df["augmentation"], y, s=68, color="#2f6fcb", edgecolor="white", linewidth=1.0, label="Augmentation", zorder=3)
    ax.scatter(df["automation"], y, s=68, color="#d96f31", edgecolor="white", linewidth=1.0, label="Automation", zorder=3)

    for i, (_, r) in enumerate(df.iterrows()):
        ax.text(r["augmentation"], i - 0.22, f"{r['augmentation']:.1f}", color="#2f6fcb", fontsize=8.5, ha="center")
        ax.text(r["automation"], i + 0.30, f"{r['automation']:.1f}", color="#d96f31", fontsize=8.5, ha="center")

    ax.set_yticks(y)
    ax.set_yticklabels(df["model"], fontsize=10)
    ax.invert_yaxis()
    ax.set_xlim(9.35, 0.65)
    ax.set_xticks(range(1, 10))
    ax.grid(axis="x", color="#d9dee7", linewidth=0.8)
    ax.grid(axis="y", color="#edf1f6", linewidth=0.6)
    ax.set_xlabel("Average rank (lower is better)", fontsize=10)
    ax.legend(loc="upper right", fontsize=9, frameon=True)
    for spine in ax.spines.values():
        spine.set_color("#c9d0da")
    handles = [
        plt.Line2D([0], [0], marker="o", linestyle="", markersize=8, markerfacecolor="#2f6fcb", markeredgecolor="white", label="Relatively stronger augmenter"),
        plt.Line2D([0], [0], marker="o", linestyle="", markersize=8, markerfacecolor="#d96f31", markeredgecolor="white", label="Relatively stronger automator"),
        plt.Line2D([0], [0], marker="o", linestyle="", markersize=8, markerfacecolor="#657083", markeredgecolor="white", label="Similar profile"),
    ]
    fig.subplots_adjust(left=0.25, right=0.98, bottom=0.13, top=0.97)
    for ext in ["png", "pdf"]:
        fig.savefig(OUT / f"figure2b_role_swap_scatter.{ext}", bbox_inches="tight")
    plt.close(fig)


def validation_metrics(data: dict) -> pd.DataFrame:
    checks = [
        {
            "label": "GPT ladder\n(Claude judge)",
            "judge": "anthropic/claude-opus-4-8",
            "models": ["GPT-3.5-Turbo", "GPT-O3-Mini", "GPT-O4-Mini", "GPT-4.1", "GPT-5-Mini"],
        },
        {
            "label": "Claude pair\n(GPT-4.1 judge)",
            "judge": "gpt-4.1",
            "models": ["Claude-Sonnet-4.6", "Claude-Opus-4.8"],
        },
    ]
    out = []
    for check in checks:
        wins = ties = total = 0
        for run in data["meta"]["replicate_runs"]:
            arr = data["runs_by_id"][run["id"]]["by_judge"]
            for task in TASK_ORDER:
                sub = [
                    r
                    for r in arr
                    if r["mode"] == "automation"
                    and r["task_slug"] == task
                    and r["judge_model"] == check["judge"]
                    and r["model_label"] in check["models"]
                    and isinstance(r.get("rank_value"), (int, float))
                ]
                sub.sort(key=lambda r: (float(r["rank_value"]), -float(r["score"]), check["models"].index(r["model_label"])))
                ranks = {r["model_label"]: i for i, r in enumerate(sub, 1)}
                for i in range(len(check["models"])):
                    for j in range(i + 1, len(check["models"])):
                        older, newer = check["models"][i], check["models"][j]
                        if older not in ranks or newer not in ranks:
                            continue
                        total += 1
                        if ranks[newer] < ranks[older]:
                            wins += 1
                        elif ranks[newer] == ranks[older]:
                            ties += 1
        out.append({"check": check["label"], "agreement": (wins + 0.5 * ties) / total, "wins": wins, "total": total})
    return pd.DataFrame(out)


def draw_stability_validation(data: dict) -> None:
    stats = []
    for mode in ["augmentation", "automation"]:
        df = all_rank_rows(data, mode)
        per_run = df.groupby(["model", "run_id"])["rank"].mean().reset_index()
        per_model = per_run.groupby("model")["rank"].std(ddof=0).dropna()
        for model, sd in per_model.items():
            stats.append({"model": model, "mode": MODE_LABELS[mode], "sd": sd})
    st = pd.DataFrame(stats)
    val = validation_metrics(data)

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2), dpi=220, gridspec_kw={"width_ratios": [1.25, 0.85]})
    ax = axes[0]
    pivot = st.pivot(index="model", columns="mode", values="sd").loc[
        st.groupby("model")["sd"].mean().sort_values().index
    ]
    y = np.arange(len(pivot))
    ax.barh(y - 0.18, pivot["Augmentation"], height=0.34, color="#2f6fcb", label="Augmentation")
    ax.barh(y + 0.18, pivot["Automation"], height=0.34, color="#d96f31", label="Automation")
    ax.set_yticks(y)
    ax.set_yticklabels([SHORT_LABELS.get(m, m) for m in pivot.index], fontsize=8)
    ax.set_xlabel("SD of model average rank across runs", fontsize=9)
    ax.set_title("A. Run-to-run stability", loc="left", fontsize=12)
    ax.grid(axis="x", color="#d9dee7", linewidth=0.8)
    ax.legend(fontsize=8)

    ax = axes[1]
    bars = ax.bar(val["check"], val["agreement"] * 100, color=["#2f6fcb", "#257f63"], width=0.55)
    for b, (_, r) in zip(bars, val.iterrows()):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 2, f"{r['agreement']*100:.0f}%", ha="center", fontsize=10, fontweight="bold")
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() - 10, f"{int(r['wins'])}/{int(r['total'])}", ha="center", fontsize=8, color="white", fontweight="bold")
    ax.set_ylim(0, 105)
    ax.set_ylabel("Expected-order agreement (%)", fontsize=9)
    ax.set_title("B. Held-out judge validation", loc="left", fontsize=12)
    ax.grid(axis="y", color="#d9dee7", linewidth=0.8)
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.14, top=0.92, wspace=0.28)
    for ext in ["png", "pdf"]:
        fig.savefig(OUT / f"figure3_stability_validation.{ext}", bbox_inches="tight")
    plt.close(fig)


def collect_pairwise_validation_rows(data: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    option_rows = []
    choice_rows = []
    for run in data["meta"]["replicate_runs"]:
        run_id = run["id"]
        for task in TASK_ORDER:
            for mode in ["augmentation", "automation"]:
                folder = ROOT / "results" / task / run_id / mode
                pairwise_path = folder / "pairwise_judgments_by_judge.csv"
                outputs_path = folder / "outputs.csv"
                if not pairwise_path.exists() or not outputs_path.exists():
                    continue
                pairs = pd.read_csv(pairwise_path)
                outputs = pd.read_csv(outputs_path)
                if outputs.empty or pairs.empty:
                    continue
                outputs = outputs.reset_index().rename(columns={"index": "output_idx"})
                model_by_idx = outputs.set_index("output_idx")["model_label"].to_dict()
                for _, row in pairs.iterrows():
                    if not bool(row.get("parse_ok", False)):
                        continue
                    opt1 = float(row.get("option_1_average", np.nan))
                    opt2 = float(row.get("option_2_average", np.nan))
                    if not np.isfinite(opt1) or not np.isfinite(opt2):
                        continue
                    winner = str(row.get("winner", "")).strip().lower()
                    left_idx = int(row["left_idx"])
                    right_idx = int(row["right_idx"])
                    judge = row["judge_label"]
                    if opt1 != opt2 and winner in {"option_1", "option_2"}:
                        higher = "option_1" if opt1 > opt2 else "option_2"
                        choice_rows.append(
                            {
                                "run_id": run_id,
                                "task": task,
                                "mode": mode,
                                "judge": judge,
                                "score_margin": abs(opt1 - opt2),
                                "aligned": winner == higher,
                            }
                        )
                    for option, idx, score in [("option_1", left_idx, opt1), ("option_2", right_idx, opt2)]:
                        model = model_by_idx.get(idx)
                        if model is None:
                            continue
                        option_rows.append(
                            {
                                "run_id": run_id,
                                "task": task,
                                "mode": mode,
                                "judge": judge,
                                "model": model,
                                "avg_score": score,
                                "win": 1.0 if winner == option else 0.0,
                            }
                        )
    return pd.DataFrame(option_rows), pd.DataFrame(choice_rows)


def draw_pairwise_rubric_validation(data: dict) -> None:
    option_rows, choice_rows = collect_pairwise_validation_rows(data)
    grouped = (
        option_rows.groupby(["run_id", "task", "mode", "judge", "model"], as_index=False)
        .agg(mean_rubric_score=("avg_score", "mean"), pairwise_win_rate=("win", "mean"), n=("win", "size"))
    )
    x = grouped["mean_rubric_score"].to_numpy()
    y = grouped["pairwise_win_rate"].to_numpy()
    corr = float(np.corrcoef(x, y)[0, 1]) if len(grouped) > 1 else float("nan")
    slope, intercept = np.polyfit(x, y, 1)

    by_judge = (
        choice_rows.groupby("judge", as_index=False)
        .agg(alignment=("aligned", "mean"), n=("aligned", "size"))
        .sort_values("alignment", ascending=True)
    )
    overall = choice_rows["aligned"].mean()

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.2), dpi=220, gridspec_kw={"width_ratios": [1.25, 0.95]})
    ax = axes[0]
    colors = grouped["mode"].map({"augmentation": "#2f6fcb", "automation": "#d96f31"}).fillna("#657083")
    ax.scatter(x, y, s=24, c=colors, alpha=0.55, edgecolor="white", linewidth=0.3)
    xs = np.linspace(x.min(), x.max(), 100)
    ax.plot(xs, slope * xs + intercept, color="#1f2433", linewidth=1.4)
    ax.set_xlabel("Mean rubric score")
    ax.set_ylabel("Pairwise win rate")
    ax.set_ylim(-0.03, 1.03)
    ax.grid(True, color="#d9dee7", linewidth=0.8)
    ax.set_title("A. Rubric scores track pairwise win rates", loc="left", fontsize=11)
    ax.text(
        0.03,
        0.95,
        f"r = {corr:.2f}\nn = {len(grouped)} model-task-mode-judge cells",
        transform=ax.transAxes,
        va="top",
        fontsize=8.5,
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "edgecolor": "#d9dee7"},
    )
    handles = [
        plt.Line2D([0], [0], marker="o", linestyle="", markersize=7, markerfacecolor="#2f6fcb", markeredgecolor="white", label="Augmentation"),
        plt.Line2D([0], [0], marker="o", linestyle="", markersize=7, markerfacecolor="#d96f31", markeredgecolor="white", label="Automation"),
    ]
    ax.legend(handles=handles, loc="lower right", fontsize=8)

    ax = axes[1]
    y_pos = np.arange(len(by_judge))
    ax.barh(y_pos, by_judge["alignment"] * 100, color="#257f63", height=0.55)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(by_judge["judge"], fontsize=8.5)
    ax.set_xlim(0, 100)
    ax.set_xlabel("Choices agreeing with higher rubric average (%)")
    ax.set_title("B. Agreement by judge", loc="left", fontsize=11)
    ax.axvline(overall * 100, color="#1f2433", linestyle=(0, (4, 4)), linewidth=1.1)
    for i, r in by_judge.iterrows():
        ax.text(r["alignment"] * 100 + 1, y_pos[list(by_judge.index).index(i)], f"{r['alignment']*100:.1f}%", va="center", fontsize=8)
    ax.grid(axis="x", color="#d9dee7", linewidth=0.8)
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.16, top=0.9, wspace=0.34)
    for ext in ["png", "pdf"]:
        fig.savefig(OUT / f"figureA_pairwise_rubric_validation.{ext}", bbox_inches="tight")
    plt.close(fig)


def criterion_validity_details(data: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    checks = [
        {
            "family": "GPT ladder",
            "judge": "anthropic/claude-opus-4-8",
            "models": ["GPT-3.5-Turbo", "GPT-O3-Mini", "GPT-O4-Mini", "GPT-4.1", "GPT-5-Mini"],
        },
        {
            "family": "Claude pair",
            "judge": "gpt-4.1",
            "models": ["Claude-Sonnet-4.6", "Claude-Opus-4.8"],
        },
    ]
    rank_rows = []
    agreement_rows = []
    for check in checks:
        wins = ties = total = 0
        for run in data["meta"]["replicate_runs"]:
            arr = data["runs_by_id"][run["id"]]["by_judge"]
            for task in TASK_ORDER:
                sub = [
                    r
                    for r in arr
                    if r["mode"] == "automation"
                    and r["task_slug"] == task
                    and r["judge_model"] == check["judge"]
                    and r["model_label"] in check["models"]
                    and isinstance(r.get("rank_value"), (int, float))
                ]
                sub.sort(key=lambda r: (float(r["rank_value"]), -float(r["score"]), check["models"].index(r["model_label"])))
                ranks = {r["model_label"]: i for i, r in enumerate(sub, 1)}
                scores = {r["model_label"]: float(r["score"]) for r in sub}
                for model, rank in ranks.items():
                    rank_rows.append(
                        {
                            "family": check["family"],
                            "judge": check["judge"],
                            "run_id": run["id"],
                            "task": task,
                            "model": model,
                            "reference_order": check["models"].index(model) + 1,
                            "rank": rank,
                            "win_rate": scores.get(model, np.nan),
                        }
                    )
                for i in range(len(check["models"])):
                    for j in range(i + 1, len(check["models"])):
                        older, newer = check["models"][i], check["models"][j]
                        if older not in ranks or newer not in ranks:
                            continue
                        total += 1
                        if ranks[newer] < ranks[older]:
                            wins += 1
                        elif ranks[newer] == ranks[older]:
                            ties += 1
        agreement_rows.append(
            {
                "family": check["family"],
                "agreement": (wins + 0.5 * ties) / total,
                "wins": wins,
                "total": total,
            }
        )
    return pd.DataFrame(rank_rows), pd.DataFrame(agreement_rows)


def draw_criterion_validity(data: dict) -> None:
    ranks, agreements = criterion_validity_details(data)
    summary = (
        ranks.groupby(["family", "model", "reference_order"], as_index=False)
        .agg(mean_rank=("rank", "mean"), sd_rank=("rank", lambda values: float(np.std(values, ddof=0))), mean_win_rate=("win_rate", "mean"))
        .sort_values(["family", "reference_order"])
    )

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.2), dpi=220, gridspec_kw={"width_ratios": [1.45, 0.9]})
    colors = ["#d8e6d3", "#b7d8c3", "#8bc0a6", "#5c9c83", "#287f63"]

    ax = axes[0]
    gpt = summary[summary["family"] == "GPT ladder"].copy()
    x = np.arange(len(gpt))
    ax.bar(x, gpt["mean_rank"], yerr=gpt["sd_rank"], color=colors[: len(gpt)], edgecolor="white", capsize=3)
    ax.set_xticks(x)
    ax.set_xticklabels(gpt["model"], rotation=25, ha="right", fontsize=8.5)
    ax.set_ylabel("Mean within-family rank\n(lower is better)")
    ax.set_ylim(5.35, 0.65)
    ax.grid(axis="y", color="#d9dee7", linewidth=0.8)
    ax.set_title("A. GPT-family outputs judged by Claude-Opus-4.8", loc="left", fontsize=11)
    agreement = agreements.loc[agreements["family"] == "GPT ladder"].iloc[0]
    ax.text(
        0.03,
        0.08,
        f"Expected-order agreement: {agreement['agreement']*100:.0f}%\n({int(agreement['wins'])}/{int(agreement['total'])} ordered pairs)",
        transform=ax.transAxes,
        fontsize=8.5,
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "edgecolor": "#d9dee7"},
    )

    ax = axes[1]
    claude = summary[summary["family"] == "Claude pair"].copy()
    x = np.arange(len(claude))
    ax.bar(x, claude["mean_rank"], yerr=claude["sd_rank"], color=["#9ec5b0", "#287f63"], edgecolor="white", capsize=3)
    ax.set_xticks(x)
    ax.set_xticklabels(claude["model"], rotation=20, ha="right", fontsize=8.5)
    ax.set_ylim(2.25, 0.75)
    ax.grid(axis="y", color="#d9dee7", linewidth=0.8)
    ax.set_title("B. Claude-family outputs judged by GPT-4.1", loc="left", fontsize=11)
    agreement = agreements.loc[agreements["family"] == "Claude pair"].iloc[0]
    ax.text(
        0.05,
        0.08,
        f"Expected-order agreement: {agreement['agreement']*100:.0f}%\n({int(agreement['wins'])}/{int(agreement['total'])} task-run comparisons)",
        transform=ax.transAxes,
        fontsize=8.5,
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "edgecolor": "#d9dee7"},
    )
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.28, top=0.9, wspace=0.32)
    for ext in ["png", "pdf"]:
        fig.savefig(OUT / f"figureA_criterion_validity.{ext}", bbox_inches="tight")
    plt.close(fig)


def draw_general_rubric_profiles(data: dict) -> None:
    dim_labels = {
        "general_instruction_following": "Instruction\nfollowing",
        "general_accuracy_specificity": "Accuracy &\nspecificity",
        "general_practical_usefulness": "Practical\nusefulness",
        "general_organization_readability": "Organization &\nreadability",
        "general_tone_audience_fit": "Tone &\naudience fit",
    }
    rows = [
        r
        for r in data["rubric_scores"]
        if r["mode"] == "augmentation"
        and r["model_label"] not in EXCLUDE
        and r["dimension"] in dim_labels
        and isinstance(r.get("mean_score"), (int, float))
    ]
    df = pd.DataFrame(rows)
    df["weighted"] = df["mean_score"] * df["n_scores"]
    summary = (
        df.groupby(["model_label", "dimension"], as_index=False)
        .agg(weighted=("weighted", "sum"), n_scores=("n_scores", "sum"))
    )
    summary["score"] = summary["weighted"] / summary["n_scores"]
    mat = summary.pivot(index="model_label", columns="dimension", values="score")
    mat = mat[[d for d in dim_labels if d in mat.columns]]
    mat = mat.loc[mat.mean(axis=1).sort_values(ascending=False).index]

    fig, ax = plt.subplots(figsize=(7.8, 4.8), dpi=220)
    im = ax.imshow(mat.to_numpy(), cmap=centaur_cmap(), vmin=1, vmax=10, aspect="auto")
    ax.set_xticks(np.arange(mat.shape[1]))
    ax.set_xticklabels([dim_labels[d] for d in mat.columns], fontsize=8.8)
    ax.set_yticks(np.arange(mat.shape[0]))
    ax.set_yticklabels(mat.index, fontsize=8.8)
    ax.tick_params(axis="both", length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks(np.arange(-0.5, mat.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, mat.shape[0], 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.1)
    ax.tick_params(which="minor", bottom=False, left=False)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            val = mat.iloc[i, j]
            ax.text(j, i, f"{val:.1f}", ha="center", va="center", fontsize=9.2, color="#1f2433")
    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("Mean rubric score", fontsize=9)
    cbar.ax.tick_params(labelsize=8)
    fig.subplots_adjust(left=0.26, right=0.94, bottom=0.17, top=0.98)
    for ext in ["png", "pdf"]:
        fig.savefig(OUT / f"figure4_general_rubric_profiles.{ext}", bbox_inches="tight")
    plt.close(fig)


def draw_prompt_rubric_schematic() -> None:
    fig, ax = plt.subplots(figsize=(12.2, 7.2), dpi=220)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    colors = {
        "prompt": "#eef5ff",
        "rubric": "#eef8f2",
        "judge": "#fff6df",
        "accent_blue": "#2f6fcb",
        "accent_green": "#287f63",
        "accent_gold": "#d59f3a",
        "text": "#1f2433",
        "muted": "#657083",
        "border": "#c9d0da",
        "purple": "#7d67b1",
        "orange": "#d96f31",
        "teal": "#0f8c84",
    }

    def rounded_box(x, y, w, h, face, edge, lw=1.0, radius=0.015):
        patch = FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle=f"round,pad=0.014,rounding_size={radius}",
            linewidth=lw,
            edgecolor=edge,
            facecolor=face,
        )
        ax.add_patch(patch)
        return patch

    def icon_card(x, y, w, h, emoji, title, body, face, edge, title_color=None):
        rounded_box(x, y, w, h, face, edge, lw=1.05, radius=0.017)
        ax.text(x + 0.035, y + h - 0.045, emoji, fontsize=16, color=edge, fontweight="semibold", ha="center", va="center")
        ax.text(
            x + 0.068,
            y + h - 0.039,
            title,
            fontsize=12.2,
            color=title_color or edge,
            fontweight="semibold",
            va="top",
        )
        ax.text(x + 0.068, y + h - 0.092, body, fontsize=8.8, color=colors["text"], va="top", linespacing=1.24)

    def chip(x, y, w, h, label, face, edge, fontsize=7.9, color=None):
        rounded_box(x, y, w, h, face, edge, lw=0.75, radius=0.010)
        ax.text(
            x + w / 2,
            y + h / 2,
            label,
            ha="center",
            va="center",
            fontsize=fontsize,
            color=color or colors["text"],
            linespacing=1.06,
        )

    def icon_chip(x, y, w, h, emoji, label, face, edge):
        rounded_box(x, y, w, h, face, edge, lw=0.8, radius=0.014)
        ax.text(x + 0.032, y + h / 2, emoji, fontsize=13.0, color=edge, fontweight="semibold", ha="center", va="center")
        ax.text(x + 0.091, y + h / 2, label, fontsize=7.8, color=colors["text"], ha="center", va="center", linespacing=1.06)

    def arrow(x1, y1, x2, y2, color):
        arr = FancyArrowPatch(
            (x1, y1),
            (x2, y2),
            arrowstyle="-|>",
            mutation_scale=14,
            linewidth=1.3,
            color=color,
            alpha=0.9,
        )
        ax.add_patch(arr)

    ax.text(0.5, 0.975, "Prompt and Rubric Design", fontsize=19, color=colors["text"], fontweight="semibold", va="top", ha="center")
    ax.text(
        0.5,
        0.925,
        "Each task prompt defines observable deliverable requirements; the micro-rubric mirrors those requirements as scored dimensions.",
        fontsize=9.6,
        color=colors["muted"],
        va="top",
        ha="center",
    )

    icon_card(
        0.04,
        0.64,
        0.27,
        0.22,
        "✎",
        "Task Prompt",
        "Specifies the final deliverable\nand required components.",
        colors["prompt"],
        colors["accent_blue"],
    )
    rounded_box(0.065, 0.655, 0.22, 0.07, "#f8fbff", colors["accent_blue"], lw=0.8, radius=0.010)
    ax.text(0.088, 0.690, "✈", fontsize=14, color=colors["accent_blue"], fontweight="semibold", ha="center", va="center")
    ax.text(
        0.175,
        0.690,
        "Travel example:\nquestions, itinerary, costs, budget",
        fontsize=7.6,
        color=colors["text"],
        ha="center",
        va="center",
        linespacing=1.05,
    )
    icon_card(
        0.365,
        0.64,
        0.27,
        0.22,
        "☑",
        "Task-Specific Micro-Rubric",
        "Scores the same observable\ncomponents on a 1-10 scale.",
        colors["rubric"],
        colors["accent_green"],
    )
    chip(0.392, 0.705, 0.103, 0.043, "cost realism", "#f8fcfa", colors["accent_green"])
    chip(0.505, 0.705, 0.103, 0.043, "itinerary quality", "#f8fcfa", colors["accent_green"])
    chip(0.392, 0.653, 0.103, 0.043, "transport\npracticality", "#f8fcfa", colors["accent_green"])
    chip(0.505, 0.653, 0.103, 0.043, "uncertainty\nhandling", "#f8fcfa", colors["accent_green"])
    icon_card(
        0.69,
        0.64,
        0.27,
        0.22,
        "⚖",
        "Pairwise Judge",
        "Scores both outputs dimension by\ndimension, then selects the better\nresponse.",
        colors["judge"],
        colors["accent_gold"],
    )
    rounded_box(0.725, 0.655, 0.20, 0.055, "#fffaf0", colors["accent_gold"], lw=0.75, radius=0.010)
    ax.text(0.748, 0.683, "✓", fontsize=13, color=colors["accent_gold"], fontweight="semibold", ha="center", va="center")
    ax.text(0.835, 0.683, "Auditable via\nscores + rationales", fontsize=7.4, color=colors["text"], ha="center", va="center", linespacing=1.02)
    arrow(0.315, 0.75, 0.36, 0.75, colors["accent_blue"])
    arrow(0.64, 0.75, 0.685, 0.75, colors["accent_green"])

    ax.plot([0.04, 0.35], [0.575, 0.575], color="#9aa3b2", linewidth=0.9)
    ax.plot([0.65, 0.96], [0.575, 0.575], color="#9aa3b2", linewidth=0.9)
    ax.text(0.5, 0.575, "General rubric dimensions used across all tasks", fontsize=11.2, color=colors["text"], fontweight="semibold", ha="center", va="center")
    general = [
        ("?", "Instruction\nfollowing", "#d9e8fb", colors["accent_blue"]),
        ("◎", "Accuracy &\nspecificity", "#dcefe5", colors["accent_green"]),
        ("◆", "Practical\nusefulness", "#fff0c2", colors["accent_gold"]),
        ("☷", "Organization &\nreadability", "#eadff8", colors["purple"]),
        ("●", "Tone &\naudience fit", "#f9e0d4", colors["orange"]),
    ]
    for i, (emoji, label, face, edge) in enumerate(general):
        x = 0.045 + i * 0.187
        icon_chip(x, 0.485, 0.16, 0.062, emoji, label, face, edge)

    ax.plot([0.04, 0.33], [0.425, 0.425], color="#9aa3b2", linewidth=0.9)
    ax.plot([0.67, 0.96], [0.425, 0.425], color="#9aa3b2", linewidth=0.9)
    ax.text(0.5, 0.425, "Examples of task-specific rubric dimensions", fontsize=11.2, color=colors["text"], fontweight="semibold", ha="center", va="center")
    icon_card(
        0.12,
        0.285,
        0.35,
        0.10,
        "✈",
        "Travel planning",
        "cost realism • itinerary quality\ntransport practicality • uncertainty handling",
        "#f8fbff",
        colors["accent_blue"],
    )
    icon_card(
        0.53,
        0.285,
        0.35,
        0.10,
        "☷",
        "Tax preparation",
        "rule accuracy • discrepancy detection\ncalculation quality • form guidance • client clarity",
        "#fbf8ff",
        colors["purple"],
        title_color=colors["purple"],
    )

    rounded_box(0.04, 0.055, 0.92, 0.17, "#f3fbfa", colors["teal"], lw=1.0, radius=0.016)
    ax.text(0.068, 0.195, "AI", fontsize=12.5, color=colors["teal"], fontweight="semibold", ha="center", va="center")
    ax.text(0.105, 0.198, "Augmentation scaffold constraint", fontsize=11.5, color=colors["teal"], fontweight="semibold", va="center")
    ax.text(0.365, 0.198, "Assistant provides guidance only, not the final deliverable.", fontsize=8.7, color=colors["text"], va="center")
    steps = [("1", "Requirements check"), ("2", "Execution plan"), ("3", "Final checklist")]
    for i, (num, label) in enumerate(steps):
        x = 0.12 + i * 0.275
        rounded_box(x, 0.095, 0.195, 0.055, "white", colors["teal"], lw=0.85, radius=0.012)
        ax.text(x + 0.029, 0.122, num, fontsize=9.5, color="white", fontweight="semibold", ha="center", va="center",
                bbox=dict(boxstyle="circle,pad=0.28", facecolor=colors["teal"], edgecolor=colors["teal"], linewidth=0.0))
        ax.text(x + 0.112, 0.122, label, fontsize=8.3, color=colors["text"], ha="center", va="center")
        if i < 2:
            arrow(x + 0.205, 0.122, x + 0.245, 0.122, colors["teal"])

    for ext in ["png", "pdf"]:
        fig.savefig(OUT / f"figure_method_prompt_rubric_alignment.{ext}", bbox_inches="tight")
    plt.close(fig)


def draw_evaluation_schematic() -> None:
    fig, ax = plt.subplots(figsize=(11.0, 5.4), dpi=220)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    colors = {
        "blue": "#2f6fcb",
        "green": "#287f63",
        "gold": "#d59f3a",
        "purple": "#7d67b1",
        "orange": "#d96f31",
        "text": "#1f2433",
        "muted": "#657083",
        "border": "#c9d0da",
        "light_blue": "#e8f0fb",
        "light_green": "#e7f4ee",
        "light_gold": "#fff3d6",
        "light_purple": "#eee8f7",
    }

    def box(x, y, w, h, title, body, face, edge):
        patch = FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.016,rounding_size=0.018",
            linewidth=1.0,
            edgecolor=edge,
            facecolor=face,
        )
        ax.add_patch(patch)
        ax.text(x + 0.02, y + h - 0.045, title, fontsize=10.8, color=edge, fontweight="semibold", va="top")
        ax.text(x + 0.02, y + h - 0.088, body, fontsize=8.0, color=colors["text"], va="top", linespacing=1.22)

    def arrow(x1, y1, x2, y2, color="#7b8493"):
        ax.add_patch(
            FancyArrowPatch(
                (x1, y1),
                (x2, y2),
                arrowstyle="-|>",
                mutation_scale=13,
                linewidth=1.2,
                color=color,
                alpha=0.9,
            )
        )

    ax.text(0.04, 0.955, "Rubric-Guided Pairwise Evaluation", fontsize=16, color=colors["text"], fontweight="semibold", va="top")
    ax.text(
        0.04,
        0.91,
        "Each task/regime produces a blind tournament: outputs are compared pairwise, scored on micro-rubrics, and aggregated into rankings.",
        fontsize=9.5,
        color=colors["muted"],
        va="top",
    )

    y_top = 0.61
    box(
        0.04,
        y_top,
        0.18,
        0.21,
        "Candidate Outputs",
        "Outputs for one task\nand regime.\n\nModel identities hidden.",
        colors["light_blue"],
        colors["blue"],
    )
    box(
        0.29,
        y_top,
        0.18,
        0.21,
        "Blind Pairing",
        "Randomized Option A\nvs. Option B matchups.\n\nOrder randomized.",
        "#f6f8fb",
        "#7b8493",
    )
    box(
        0.54,
        y_top,
        0.18,
        0.21,
        "Judge Scoring",
        "Score both options on\ntask and general rubrics.\n\nReturn scores + rationale.",
        colors["light_green"],
        colors["green"],
    )
    box(
        0.78,
        y_top,
        0.18,
        0.21,
        "Pairwise Choice",
        "Select the stronger\nresponse overall.\n\nAuditable against scores.",
        colors["light_gold"],
        colors["gold"],
    )
    arrow(0.225, y_top + 0.105, 0.285, y_top + 0.105, colors["blue"])
    arrow(0.475, y_top + 0.105, 0.535, y_top + 0.105, colors["green"])
    arrow(0.725, y_top + 0.105, 0.775, y_top + 0.105, colors["gold"])

    y_bottom = 0.25
    box(
        0.16,
        y_bottom,
        0.21,
        0.20,
        "Per-Judge Tournament",
        "Convert choices into\npairwise win rates.\n\nRank models per task.",
        colors["light_purple"],
        colors["purple"],
    )
    box(
        0.435,
        y_bottom,
        0.21,
        0.20,
        "Bias Control",
        "Apply leave-family-out\nmasking.\n\nNo same-family judging.",
        "#f6f8fb",
        "#7b8493",
    )
    box(
        0.71,
        y_bottom,
        0.21,
        0.20,
        "Aggregated Outputs",
        "Average ranks across\neligible judges and runs.\n\nReport rankings + audit trail.",
        "#fdece3",
        colors["orange"],
    )
    arrow(0.87, y_top, 0.80, y_bottom + 0.205, colors["gold"])
    arrow(0.375, y_bottom + 0.10, 0.43, y_bottom + 0.10, colors["purple"])
    arrow(0.65, y_bottom + 0.10, 0.705, y_bottom + 0.10, colors["orange"])

    ax.text(
        0.04,
        0.105,
        "Primary outcome: pairwise win rate. Secondary audit trail: rubric dimension scores and judge rationales.",
        fontsize=9.2,
        color=colors["muted"],
    )

    for ext in ["png", "pdf"]:
        fig.savefig(OUT / f"figure_method_evaluation_procedure.{ext}", bbox_inches="tight")
    plt.close(fig)


def write_best_model_table(data: dict) -> Path:
    task_keys = [TASK_LABELS[t] for t in TASK_ORDER] + ["Average"]
    task_labels = {**TASK_LABELS, "Average": "Average (all tasks)"}
    best_by_mode: dict[str, dict[str, str]] = {}
    for mode in ["automation", "augmentation"]:
        mean, _, _ = mean_rank_matrix(data, mode)
        best_by_mode[mode] = {
            task: display_model_label(mean.loc[task].idxmin()) for task in mean.index
        }

    lines = [
        r"\begin{table}[t]",
        r"    \centering",
        r"    \caption{Best-performing model by task and regime, based on lowest mean rank across three independent runs.}",
        r"    \label{tab:best-model-by-task}",
        r"    \small",
        r"    \begin{tabular}{lcc}",
        r"        \toprule",
        r"        Task & Automation & Augmentation \\",
        r"        \midrule",
    ]
    for task in task_keys:
        label = task_labels.get(task, task)
        auto = best_by_mode["automation"][task]
        aug = best_by_mode["augmentation"][task]
        lines.append(f"        {label} & {auto} & {aug} \\\\")
    lines.extend(
        [
            r"        \bottomrule",
            r"    \end{tabular}",
            r"\end{table}",
        ]
    )
    path = OUT / "table_best_model_by_task.tex"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main() -> None:
    data = load_data()
    draw_prompt_rubric_schematic()
    draw_evaluation_schematic()
    draw_heatmap_rank_of_ranks(data, "augmentation", "figure1_augmentation_heatmap")
    draw_heatmap_mean_se(data, "augmentation", "figureA_augmentation_heatmap_mean_se")
    draw_heatmap_rank_of_ranks(data, "automation", "figure2a_automation_heatmap")
    draw_heatmap_mean_se(data, "automation", "figureA_automation_heatmap_mean_se")
    draw_role_scatter(data)
    draw_stability_validation(data)
    draw_pairwise_rubric_validation(data)
    draw_criterion_validity(data)
    draw_general_rubric_profiles(data)
    table_path = write_best_model_table(data)
    paper_dir = ROOT / "paper_figures"
    paper_dir.mkdir(parents=True, exist_ok=True)
    (paper_dir / "table_best_model_by_task.tex").write_text(table_path.read_text(encoding="utf-8"), encoding="utf-8")
    print(f"Wrote figures to {OUT}")
    print(f"Wrote {table_path}")
    for p in sorted(OUT.glob("figure*.*")):
        print(p)


if __name__ == "__main__":
    main()
