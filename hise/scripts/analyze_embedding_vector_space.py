"""Measure BGE embedding geometry on a small structure-sensitive sample set."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("USE_TF", "0")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn.functional as F
from huggingface_hub import snapshot_download
from transformers import AutoModel, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = REPO_ROOT / "Documentation" / "reports" / "vector_space"
MODEL_ID = "BAAI/bge-base-en-v1.5"
SEED = 20261021

CASES = [
    {
        "id": "risk-01",
        "query": "Which risks could delay the migration?",
        "positive_path": "Operations / Risks",
        "positive": "The migration may be delayed if the vendor misses the delivery date.",
        "negative_path": "Operations / Benefits",
        "negative": "The migration may be delayed if the vendor misses the delivery date.",
    },
    {
        "id": "decision-01",
        "query": "What did the board decide about the budget?",
        "positive_path": "Governance / Decisions",
        "positive": "The board approved the revised annual budget during its meeting.",
        "negative_path": "Governance / Discussion",
        "negative": "The board approved the revised annual budget during its meeting.",
    },
    {
        "id": "exception-01",
        "query": "Which exception was granted to the policy?",
        "positive_path": "Policy / Exceptions",
        "positive": "Regional managers may extend the review period by ten days.",
        "negative_path": "Policy / Standard procedure",
        "negative": "Regional managers may extend the review period by ten days.",
    },
    {
        "id": "result-01",
        "query": "What result did the clinical study report?",
        "positive_path": "Clinical study / Results",
        "positive": "Participants showed a modest improvement after twelve weeks.",
        "negative_path": "Clinical study / Limitations",
        "negative": "Participants showed a modest improvement after twelve weeks.",
    },
    {
        "id": "owner-01",
        "query": "Who is responsible for the security review?",
        "positive_path": "Project plan / Owners",
        "positive": "The platform engineering team will complete the review by June.",
        "negative_path": "Project plan / Milestones",
        "negative": "The platform engineering team will complete the review by June.",
    },
    {
        "id": "assumption-01",
        "query": "What assumption underlies the forecast?",
        "positive_path": "Forecast / Assumptions",
        "positive": "Demand will remain stable through the second half of the year.",
        "negative_path": "Forecast / Actuals",
        "negative": "Demand will remain stable through the second half of the year.",
    },
    {
        "id": "action-01",
        "query": "What action must the supplier take next?",
        "positive_path": "Supplier review / Required actions",
        "positive": "The supplier must submit a corrective action plan within five business days.",
        "negative_path": "Supplier review / Background",
        "negative": "The supplier must submit a corrective action plan within five business days.",
    },
    {
        "id": "definition-01",
        "query": "How does the contract define a confidential record?",
        "positive_path": "Contract / Definitions",
        "positive": "A confidential record is information marked restricted when it is disclosed.",
        "negative_path": "Contract / Termination",
        "negative": "A confidential record is information marked restricted when it is disclosed.",
    },
    {
        "id": "cause-01",
        "query": "What caused the reduction in operating costs?",
        "positive_path": "Quarterly report / Causes",
        "positive": "Operating costs declined after the company consolidated two facilities.",
        "negative_path": "Quarterly report / Outlook",
        "negative": "Operating costs declined after the company consolidated two facilities.",
    },
    {
        "id": "recommendation-01",
        "query": "What does the report recommend for the next phase?",
        "positive_path": "Assessment / Recommendations",
        "positive": "The team should run a limited pilot before expanding the program.",
        "negative_path": "Assessment / Current status",
        "negative": "The team should run a limited pilot before expanding the program.",
    },
]


def git_commit() -> str | None:
    result = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def embedding_text(case: dict[str, str], candidate: str, arm: str) -> str:
    if arm == "plain":
        return case[candidate]
    return f"Section path: {case[candidate + '_path']}\n\n{case[candidate]}"


def encode(
    texts: list[str], tokenizer: Any, model: Any, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    batch = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt",
    ).to(device)
    with torch.inference_mode():
        hidden = model(**batch).last_hidden_state
    cls = hidden[:, 0, :].float()
    normalized = F.normalize(cls, p=2, dim=1)
    return cls.cpu(), normalized.cpu()


def chart_results(
    case_rows: list[dict[str, Any]],
    texts: list[str],
    vectors: torch.Tensor,
    raw_vectors: torch.Tensor,
) -> list[str]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="notebook")
    written: list[str] = []

    margins = pd.DataFrame(case_rows)
    fig, ax = plt.subplots(figsize=(12, 5.5))
    plot_data = margins.melt(
        id_vars="case_id",
        value_vars=["plain_margin", "path_prefix_margin"],
        var_name="arm",
        value_name="positive_minus_negative_cosine",
    )
    plot_data["arm"] = plot_data["arm"].map(
        {"plain_margin": "Plain text", "path_prefix_margin": "Section-path prefix"}
    )
    sns.barplot(
        data=plot_data,
        x="case_id",
        y="positive_minus_negative_cosine",
        hue="arm",
        palette=["#4C72B0", "#DD8452"],
        ax=ax,
    )
    ax.axhline(0, color="#333333", linewidth=1)
    ax.set_title("Query-to-candidate cosine margin by sample")
    ax.set_xlabel("Handwritten sample")
    ax.set_ylabel("Positive cosine − structural-negative cosine")
    ax.tick_params(axis="x", rotation=35)
    fig.tight_layout()
    path = OUTPUT_DIR / "query_candidate_margins.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    written.append(path.name)

    similarity = vectors @ vectors.T
    labels = [f"{i + 1:02d} {text[:26]}" for i, text in enumerate(texts)]
    fig, ax = plt.subplots(figsize=(13, 11))
    sns.heatmap(
        similarity.numpy(),
        xticklabels=labels,
        yticklabels=labels,
        cmap="vlag",
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
        ax=ax,
        cbar_kws={"label": "Cosine similarity"},
    )
    ax.set_title("BGE cosine-similarity matrix: section-path-prefixed candidates")
    ax.tick_params(axis="x", labelrotation=90, labelsize=7)
    ax.tick_params(axis="y", labelsize=7)
    fig.tight_layout()
    path = OUTPUT_DIR / "prefixed_candidate_cosine_heatmap.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    written.append(path.name)

    centered = vectors.numpy() - vectors.numpy().mean(axis=0, keepdims=True)
    left_vectors, singular_values, _ = np.linalg.svd(centered, full_matrices=False)
    projections = left_vectors[:, :2] * singular_values[:2]
    points = pd.DataFrame(
        {
            "pc1": projections[:, 0],
            "pc2": projections[:, 1],
            "case_id": [row["case_id"] for row in case_rows for _ in range(2)],
            "candidate_role": ["positive", "structural negative"] * len(case_rows),
        }
    )
    fig, ax = plt.subplots(figsize=(10, 7))
    sns.scatterplot(
        data=points,
        x="pc1",
        y="pc2",
        hue="candidate_role",
        style="candidate_role",
        s=110,
        palette={"positive": "#1B9E77", "structural negative": "#D95F02"},
        ax=ax,
    )
    for index, row in points.iterrows():
        ax.annotate(row["case_id"], (row["pc1"], row["pc2"]), xytext=(4, 4), textcoords="offset points", fontsize=7)
    ax.set_title("2D PCA of section-path-prefixed candidate vectors")
    ax.set_xlabel("PC1 (visualization only)")
    ax.set_ylabel("PC2 (visualization only)")
    ax.legend(title="Candidate role")
    fig.tight_layout()
    path = OUTPUT_DIR / "prefixed_candidates_pca.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    written.append(path.name)

    norms = torch.linalg.vector_norm(raw_vectors, dim=1).numpy()
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.histplot(norms, bins=12, kde=True, color="#8172B2", ax=ax)
    ax.set_title("Raw CLS vector L2 norms before normalization")
    ax.set_xlabel("L2 norm")
    ax.set_ylabel("Candidate count")
    fig.tight_layout()
    path = OUTPUT_DIR / "raw_cls_norm_distribution.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    written.append(path.name)
    return written


def run() -> dict[str, Any]:
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    configured_model_path = os.environ.get("HISE_BGE_MODEL_PATH")
    model_path = Path(configured_model_path) if configured_model_path else Path(
        snapshot_download(MODEL_ID, local_files_only=True)
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModel.from_pretrained(model_path, local_files_only=True).to(device).eval()

    candidate_texts: list[str] = []
    candidates_by_arm: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    query_texts = [case["query"] for case in CASES]
    all_texts = query_texts.copy()
    for case in CASES:
        for candidate in ("positive", "negative"):
            all_texts.append(embedding_text(case, candidate, "plain"))
            candidate_texts.append(
                embedding_text(case, candidate, "path_prefix")
            )
    started = time.perf_counter()
    raw_all, normalized_all = encode(all_texts, tokenizer, model, device)
    query_vectors = normalized_all[: len(CASES)]
    plain_vectors = normalized_all[len(CASES) :]
    raw_candidate_vectors = raw_all[len(CASES) :]
    prefixed_raw_vectors, prefixed_vectors = encode(candidate_texts, tokenizer, model, device)
    embedding_seconds = time.perf_counter() - started

    rows: list[dict[str, Any]] = []
    for index, case in enumerate(CASES):
        plain_positive = F.cosine_similarity(
            query_vectors[index], plain_vectors[2 * index], dim=0
        ).item()
        plain_negative = F.cosine_similarity(
            query_vectors[index], plain_vectors[2 * index + 1], dim=0
        ).item()
        prefixed_positive = F.cosine_similarity(
            query_vectors[index], prefixed_vectors[2 * index], dim=0
        ).item()
        prefixed_negative = F.cosine_similarity(
            query_vectors[index], prefixed_vectors[2 * index + 1], dim=0
        ).item()
        rows.append(
            {
                "case_id": case["id"],
                "query": case["query"],
                "positive_path": case["positive_path"],
                "negative_path": case["negative_path"],
                "plain_positive_cosine": plain_positive,
                "plain_negative_cosine": plain_negative,
                "plain_margin": plain_positive - plain_negative,
                "path_prefix_positive_cosine": prefixed_positive,
                "path_prefix_negative_cosine": prefixed_negative,
                "path_prefix_margin": prefixed_positive - prefixed_negative,
                "plain_correct": plain_positive > plain_negative + 1e-8,
                "path_prefix_correct": prefixed_positive > prefixed_negative + 1e-8,
                "plain_tie": abs(plain_positive - plain_negative) <= 1e-8,
                "path_prefix_tie": abs(prefixed_positive - prefixed_negative) <= 1e-8,
            }
        )

    summary = {
        "title": "Exploratory embedding vector-space analysis",
        "created_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "model": MODEL_ID,
        "model_snapshot": model_path.name,
        "model_path": str(model_path),
        "pooling": "last_hidden_state[:, 0, :] (CLS)",
        "normalization": "L2 normalization after CLS pooling; cosine is dot product of normalized vectors",
        "embedding_dimension": int(raw_all.shape[1]),
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "seed": SEED,
        "git_commit": git_commit(),
        "sample_count": len(CASES),
        "candidate_pair_count": len(CASES),
        "embedding_seconds": embedding_seconds,
        "arms": ["plain candidate text", "section path prepended to candidate text"],
        "plain_pairwise_accuracy": sum(row["plain_correct"] for row in rows) / len(rows),
        "plain_ties": sum(row["plain_tie"] for row in rows),
        "path_prefix_pairwise_accuracy": sum(row["path_prefix_correct"] for row in rows) / len(rows),
        "path_prefix_ties": sum(row["path_prefix_tie"] for row in rows),
        "plain_mean_margin": float(np.mean([row["plain_margin"] for row in rows])),
        "path_prefix_mean_margin": float(np.mean([row["path_prefix_margin"] for row in rows])),
        "raw_cls_norm_min": float(torch.linalg.vector_norm(raw_candidate_vectors, dim=1).min()),
        "raw_cls_norm_mean": float(torch.linalg.vector_norm(raw_candidate_vectors, dim=1).mean()),
        "raw_cls_norm_max": float(torch.linalg.vector_norm(raw_candidate_vectors, dim=1).max()),
        "scope_note": (
            "Small hand-authored mechanism probe only. Positive and structural-negative "
            "candidate bodies are intentionally identical within each pair, so plain-text "
            "ties are an expected control. This is not an unbiased retrieval benchmark, "
            "does not test trained HiSE/SGAM, and has no statistical generalization claim."
        ),
        "results": rows,
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    archive_path = OUTPUT_DIR / "embedding_vectors.npz"
    np.savez_compressed(
        archive_path,
        query_vectors=query_vectors.numpy(),
        plain_candidate_vectors=plain_vectors.numpy(),
        prefixed_candidate_vectors=prefixed_vectors.numpy(),
        plain_raw_cls_vectors=raw_candidate_vectors.numpy(),
        prefixed_raw_cls_vectors=prefixed_raw_vectors.numpy(),
        candidate_ids=np.array(
            [case["id"] for case in CASES for _ in ("positive", "negative")]
        ),
        candidate_roles=np.array(["positive", "structural negative"] * len(CASES)),
    )
    dimension_std = prefixed_vectors.std(dim=0, unbiased=False)
    summary["embedding_archive"] = archive_path.name
    summary["normalized_dimension_std"] = {
        "min": float(dimension_std.min()),
        "mean": float(dimension_std.mean()),
        "max": float(dimension_std.max()),
    }
    (OUTPUT_DIR / "vector_space_results.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    figures = chart_results(
        rows,
        candidate_texts,
        prefixed_vectors,
        torch.cat(
            [
                raw_candidate_vectors[::2],
                raw_candidate_vectors[1::2],
            ],
            dim=0,
        ),
    )
    summary["figures"] = figures
    (OUTPUT_DIR / "vector_space_results.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    write_markdown(summary)
    print(
        json.dumps(
            {
                "report": str(OUTPUT_DIR / "vector_space_analysis.md"),
                "results": str(OUTPUT_DIR / "vector_space_results.json"),
                "figures": figures,
                "device": summary["device"],
                "embedding_dimension": summary["embedding_dimension"],
                "plain_pairwise_accuracy": summary["plain_pairwise_accuracy"],
                "path_prefix_pairwise_accuracy": summary["path_prefix_pairwise_accuracy"],
            },
            indent=2,
        )
    )
    return summary


def write_markdown(summary: dict[str, Any]) -> None:
    lines = [
        "# Exploratory embedding vector-space analysis",
        "",
        "## Setup",
        "",
        f"- Model: `{summary['model']}`",
        f"- Snapshot: `{summary['model_snapshot']}`",
        f"- Embedding size: {summary['embedding_dimension']} dimensions",
        f"- Pooling: {summary['pooling']}",
        f"- Normalization: {summary['normalization']}",
        f"- Device: `{summary['device']}`"
        + (f" (`{summary['gpu']}`)" if summary["gpu"] else ""),
        f"- Handwritten structural pairs: {summary['sample_count']}",
        f"- Embedding pass time: {summary['embedding_seconds']:.3f} seconds",
        f"- Across-candidate normalized coordinate standard deviation (min/mean/max): "
        f"{summary['normalized_dimension_std']['min']:.6f} / "
        f"{summary['normalized_dimension_std']['mean']:.6f} / "
        f"{summary['normalized_dimension_std']['max']:.6f}",
        "",
        "## Pairwise results",
        "",
        "| Arm | Pairwise accuracy | Ties | Mean positive-minus-negative cosine margin |",
        "|---|---:|---:|---:|",
        f"| Plain candidate text | {summary['plain_pairwise_accuracy']:.3f} | {summary['plain_ties']} | {summary['plain_mean_margin']:.6f} |",
        f"| Section-path prefix | {summary['path_prefix_pairwise_accuracy']:.3f} | {summary['path_prefix_ties']} | {summary['path_prefix_mean_margin']:.6f} |",
        "",
        "### Per-pair cosine values",
        "",
        "| Pair | Plain positive | Plain negative | Plain margin | Prefixed positive | Prefixed negative | Prefixed margin |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary["results"]:
        lines.append(
            f"| {row['case_id']} | {row['plain_positive_cosine']:.6f} | "
            f"{row['plain_negative_cosine']:.6f} | {row['plain_margin']:.6f} | "
            f"{row['path_prefix_positive_cosine']:.6f} | "
            f"{row['path_prefix_negative_cosine']:.6f} | "
            f"{row['path_prefix_margin']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Figures",
            "",
            "![Query-candidate margins](query_candidate_margins.png)",
            "",
            "![Prefixed candidate cosine matrix](prefixed_candidate_cosine_heatmap.png)",
            "",
            "![PCA visualization](prefixed_candidates_pca.png)",
            "",
            "![Raw CLS norms](raw_cls_norm_distribution.png)",
            "",
            "## Interpretation limits",
            "",
            summary["scope_note"],
            "",
            "The PCA chart is a projection for visualization, not a metric or proof of cluster separation. "
            "Cosine margins and pairwise accuracy describe only these ten constructed pairs. "
            "A genuine retrieval-quality claim requires a larger held-out query set and the same "
            "frozen chunks compared across plain-text, path-prefix, windowed-zero-SGAM, and trained HiSE arms.",
            "",
            "Raw data: `vector_space_results.json`.",
            "Per-vector arrays: `embedding_vectors.npz` (normalized query/candidate vectors and raw CLS vectors).",
        ]
    )
    (OUTPUT_DIR / "vector_space_analysis.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    run()
