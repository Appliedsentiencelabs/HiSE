"""Generate publication-style plots from the component validation evidence JSON."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

REPO_ROOT = Path(__file__).resolve().parents[2]
REPORT_PATH = REPO_ROOT / "Dcoumentation" / "reports" / "evidence_report.json"
OUT_DIR = REPO_ROOT / "Dcoumentation" / "reports" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams["figure.dpi"] = 180
sns.set_theme(style="whitegrid", context="talk")


def load_report() -> dict:
    return json.loads(REPORT_PATH.read_text(encoding="utf-8"))


def plot_zero_weight_equivalence(data: dict) -> None:
    rows = [
        {
            "sentence": s["index"],
            "hidden_max_abs_diff_nonpad": s["hidden_max_abs_diff_nonpad"],
            "cls_cosine": s["cls_cosine"],
            "mean_pool_cosine": s["mean_pool_cosine"],
        }
        for s in data["equivalence"]["per_sentence"]
    ]
    frame = pd.DataFrame(rows)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.lineplot(data=frame, x="sentence", y="hidden_max_abs_diff_nonpad", marker="o", ax=axes[0], color="#4C72B0")
    axes[0].set_title("Zero-weight equivalence: max abs diff per sentence")
    axes[0].set_xlabel("Sentence index")
    axes[0].set_ylabel("Max abs diff (non-padding tokens)")
    axes[0].axhline(data["thresholds_fixed_before_run"]["zero_weight_max_abs_diff"], color="red", linestyle="--", linewidth=1.5, label="threshold 1e-5")
    axes[0].legend()

    sns.lineplot(data=frame, x="sentence", y="cls_cosine", marker="o", ax=axes[1], color="#55A868")
    axes[1].set_title("Zero-weight equivalence: CLS cosine per sentence")
    axes[1].set_xlabel("Sentence index")
    axes[1].set_ylabel("CLS cosine")
    axes[1].axhline(data["thresholds_fixed_before_run"]["zero_weight_cls_cosine"], color="red", linestyle="--", linewidth=1.5, label="threshold 0.99999")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(OUT_DIR / "zero_weight_equivalence.png", bbox_inches="tight")
    plt.close(fig)


def plot_attention_shift(data: dict) -> None:
    rows = []
    for item in data["attention_shift"]["grid"]:
        for win in item["windows"]:
            rows.append(
                {
                    "parameter": item["parameter"],
                    "weight": float(item["weight"]),
                    "window": win["window"],
                    "gain": win["gain"],
                    "cls_cosine": win["cls_cosine"],
                }
            )
    frame = pd.DataFrame(rows)

    summary = (
        frame.groupby(["parameter", "weight"], as_index=False)[["gain", "cls_cosine"]]
        .mean()
        .rename(columns={"gain": "mean_gain", "cls_cosine": "mean_cls_cosine"})
    )

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(
        data=summary,
        x="parameter",
        y="mean_gain",
        hue="weight",
        hue_order=[-0.5, -1.0, -2.0],
        palette="coolwarm",
        dodge=True,
        ax=ax,
    )
    ax.set_title("Attention-shift mean gain by parameter and weight")
    ax.set_xlabel("SGAM parameter")
    ax.set_ylabel("Mean within-section gain")
    ax.legend(title="Weight")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "attention_shift_gain.png", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.lineplot(data=summary, x="weight", y="mean_gain", hue="parameter", marker="o", ax=ax)
    ax.set_title("Mean gain vs SGAM weight magnitude")
    ax.set_xlabel("Weight")
    ax.set_ylabel("Mean gain")
    ax.axhline(data["thresholds_fixed_before_run"]["attention_shift_gain"], color="red", linestyle="--", linewidth=1.5, label="threshold 0.10")
    ax.legend(title="Parameter")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "attention_shift_weight_curve.png", bbox_inches="tight")
    plt.close(fig)


def plot_overhead(data: dict) -> None:
    rows = []
    for arm_name, entries in data["overhead"].items():
        for entry in entries:
            rows.append(
                {
                    "arm": arm_name,
                    "batch_size": entry["batch_size"],
                    "median_ms": entry["measure"]["median_ms"],
                    "p95_ms": entry["measure"]["p95_ms"],
                    "peak_allocated_mb": entry["measure"]["peak_allocated_mb"],
                }
            )
    frame = pd.DataFrame(rows)

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    sns.barplot(data=frame, x="batch_size", y="median_ms", hue="arm", palette="Set2", ax=axes[0])
    axes[0].set_title("Median latency by batch size")
    axes[0].set_xlabel("Batch size")
    axes[0].set_ylabel("Median latency (ms)")

    sns.barplot(data=frame, x="batch_size", y="peak_allocated_mb", hue="arm", palette="Set2", ax=axes[1])
    axes[1].set_title("Peak allocated memory by batch size")
    axes[1].set_xlabel("Batch size")
    axes[1].set_ylabel("Peak allocated memory (MB)")

    fig.tight_layout()
    fig.savefig(OUT_DIR / "overhead_latency_memory.png", bbox_inches="tight")
    plt.close(fig)


def write_summary(data: dict) -> None:
    summary_lines = [
        "# Component validation charts",
        "",
        "This report summarizes the real measured values captured in the component-validation evidence JSON.",
        "",
        "## Key numbers",
        "",
        f"- Zero-weight hidden diff max: {data['equivalence']['hidden_diff']['max']:.8e}",
        f"- Zero-weight CLS cosine min/mean/max: {data['equivalence']['cls_cosine']['min']:.8f} / {data['equivalence']['cls_cosine']['mean']:.8f} / {data['equivalence']['cls_cosine']['max']:.8f}",
        f"- Attention shift mean gain (parent_weight, -1.0): {sum(w['gain'] for w in next(item['windows'] for item in data['attention_shift']['grid'] if item['parameter'] == 'parent_weight' and item['weight'] == -1.0)) / 5:.6f}",
        "",
        "## Files",
        "",
        "- `figures/zero_weight_equivalence.png`",
        "- `figures/attention_shift_gain.png`",
        "- `figures/attention_shift_weight_curve.png`",
        "- `figures/overhead_latency_memory.png`",
    ]
    (REPO_ROOT / "Dcoumentation" / "reports" / "component_validation_charts.md").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")


def main() -> None:
    data = load_report()
    plot_zero_weight_equivalence(data)
    plot_attention_shift(data)
    plot_overhead(data)
    write_summary(data)
    print(f"Generated validation charts under: {OUT_DIR}")
    print(f"Summary report: {REPO_ROOT / 'Dcoumentation' / 'reports' / 'component_validation_charts.md'}")


if __name__ == "__main__":
    main()
