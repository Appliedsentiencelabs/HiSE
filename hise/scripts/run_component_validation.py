"""Run repeatable component tests and write JSON/Markdown evidence."""

from __future__ import annotations

import gc
import json
import os
import random
import statistics
import subprocess
import sys
import tempfile
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

os.environ.setdefault("USE_TF", "0")

import nbformat
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import transformers
from nbclient import NotebookClient
from transformers import AutoModel, AutoTokenizer

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PACKAGE_ROOT.parent
sys.path.insert(0, str(PACKAGE_ROOT / "src"))

from hise.bert_sgam import attach_sgam_to_bert_attention  # noqa: E402

MODEL_ID = "BAAI/bge-base-en-v1.5"
SEED = 20261021
FIXED_SENTENCES = [
    "The company reported higher revenue during the first quarter.",
    "Operating margin improved as manufacturing costs declined.",
    "The board approved the annual budget at its meeting.",
    "Customer retention increased after the support program launched.",
    "The research team evaluated three retrieval architectures.",
    "A new distribution center opened near the regional market.",
    "The policy requires approval from the department manager.",
    "Energy consumption fell after the equipment upgrade.",
    "The product roadmap includes a redesigned mobile application.",
    "Quarterly cash flow exceeded the forecast by five percent.",
    "The clinical study enrolled participants across four hospitals.",
    "The security review identified two configuration issues.",
    "The supplier will deliver replacement components next month.",
    "The proposal compares implementation cost and expected benefit.",
    "The software release includes accessibility improvements.",
    "The committee postponed its decision until additional data arrives.",
    "The warehouse recorded fewer damaged packages this week.",
    "The contract renews automatically unless either party gives notice.",
    "The sales team expanded coverage in the northern region.",
    "The report summarizes risks, milestones, and open questions.",
]
WINDOW_SECTIONS = [
    (
        "Revenue grew steadily as new customers joined the service.",
        "Operating margin improved because production waste declined.",
        "The board approved a revised annual investment plan.",
    ),
    (
        "The support team resolved more customer requests this month.",
        "Product reliability increased after the component redesign.",
        "The regional office will hire additional service staff.",
    ),
    (
        "Researchers measured performance across several test groups.",
        "The new method reduced processing time for each document.",
        "The appendix lists limitations and follow-up experiments.",
    ),
    (
        "The supplier met the delivery target for critical parts.",
        "Inventory costs fell after the forecasting change.",
        "The logistics team reviewed the updated contingency plan.",
    ),
    (
        "The policy defines responsibilities for data access.",
        "Managers must document exceptions and approval dates.",
        "The audit committee will review compliance next quarter.",
    ),
]

THRESHOLDS = {
    "zero_weight_max_abs_diff": 1e-5,
    "zero_weight_cls_cosine": 0.99999,
    "zero_weight_mean_pool_cosine": 0.99999,
    "attention_shift_gain": 0.10,
    "attention_shift_windows": 4,
    "attention_shift_cls_cosine": 0.90,
    "shuffled_true_section_max_gain": 0.02,
    "shuffled_control_rule": "uniform random permutation of true labels, preserving each window's section sizes; diagonal excluded from scored mass",
    "attention_weight_grid": [-0.5, -1.0, -2.0],
    "attention_parameters": ["depth_weight", "parent_weight", "sibling_weight"],
    "gradient_init": 0.0,
    "sgam_learning_rate": 0.05,
    "timing_warmups": 5,
    "timing_repeats": 20,
    "timing_sequence_length": 512,
}


def seed_everything() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)


def git_value(*args: str) -> str | None:
    result = subprocess.run(
        ["git", "-C", str(REPO_ROOT), *args],
        text=True,
        capture_output=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def model_snapshot(model_id: str) -> dict[str, str]:
    from huggingface_hub import snapshot_download

    path = Path(snapshot_download(model_id, local_files_only=True))
    return {"commit_hash": path.name, "path": str(path)}


def load_model(
    model_id: str, device: torch.device, *, implementation: str | None = None
) -> tuple[Any, Any]:
    tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
    kwargs: dict[str, Any] = {"local_files_only": True}
    if implementation:
        kwargs["attn_implementation"] = implementation
    model = AutoModel.from_pretrained(model_id, **kwargs).to(device).eval()
    return tokenizer, model


def tokenize_windows(
    tokenizer: Any, device: torch.device
) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor, list[dict[str, Any]]]:
    texts: list[str] = []
    spans_by_window: list[list[tuple[int, int, int]]] = []
    for sections in WINDOW_SECTIONS:
        parts: list[str] = []
        spans: list[tuple[int, int, int]] = []
        cursor = 0
        for section_id, section in enumerate(sections, start=1):
            if parts:
                parts.append("\n\n")
                cursor += 2
            start = cursor
            parts.append(section)
            cursor += len(section)
            spans.append((start, cursor, section_id))
        texts.append("".join(parts))
        spans_by_window.append(spans)

    encoded = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=512,
        return_offsets_mapping=True,
        return_tensors="pt",
    )
    offsets = encoded.pop("offset_mapping")
    batch, length = encoded["input_ids"].shape
    labels = torch.full((batch, length), -1, dtype=torch.long)
    addresses = torch.zeros((batch, length, 4), dtype=torch.float32)
    records: list[dict[str, Any]] = []

    for batch_index in range(batch):
        token_counts = [0, 0, 0, 0]
        for token_index, (start_tensor, end_tensor) in enumerate(offsets[batch_index]):
            start, end = int(start_tensor), int(end_tensor)
            section_id = -1
            if end > start:
                for span_start, span_end, candidate in spans_by_window[batch_index]:
                    if start < span_end and end > span_start:
                        section_id = candidate
                        break
            addresses[batch_index, token_index, 0] = token_index
            if section_id > 0:
                labels[batch_index, token_index] = section_id
                depth = 1 if section_id != 2 else 2
                addresses[batch_index, token_index, 1:] = torch.tensor(
                    [depth, section_id, token_counts[section_id]], dtype=torch.float32
                )
                token_counts[section_id] += 1
        records.append(
            {
                "window": batch_index + 1,
                "token_count_including_specials": int(encoded["attention_mask"][batch_index].sum()),
                "section_token_counts": token_counts[1:],
                "section_char_spans": [
                    {"start": start, "end": end, "section": section}
                    for start, end, section in spans_by_window[batch_index]
                ],
                "special_and_padding_tokens_excluded": int((labels[batch_index] < 0).sum()),
            }
        )
    return (
        {key: value.to(device) for key, value in encoded.items()},
        addresses.to(device),
        labels.to(device),
        records,
    )


def labels_to_addresses(
    base: torch.Tensor, labels: torch.Tensor, *, shuffle: bool
) -> torch.Tensor:
    result = base.clone()
    generator = torch.Generator().manual_seed(SEED)
    for batch_index in range(labels.shape[0]):
        indices = torch.nonzero(labels[batch_index] > 0, as_tuple=False).flatten()
        values = labels[batch_index, indices].cpu()
        if shuffle:
            values = values[torch.randperm(len(values), generator=generator)]
        counts = {1: 0, 2: 0, 3: 0}
        for token_index, section in zip(indices.tolist(), values.tolist()):
            result[batch_index, token_index, 1] = 1 if section != 2 else 2
            result[batch_index, token_index, 2] = section
            result[batch_index, token_index, 3] = counts[section]
            counts[section] += 1
    return result


def within_section_by_layer(
    attentions: tuple[torch.Tensor, ...], labels: torch.Tensor, *, exclude_diagonal: bool = False
) -> list[list[float]]:
    per_window = []
    for batch_index in range(labels.shape[0]):
        row = labels[batch_index]
        valid = row > 0
        same_section = (row[:, None] == row[None, :]) & valid[:, None] & valid[None, :]
        if exclude_diagonal:
            same_section.fill_diagonal_(False)
        layer_scores = []
        for layer_attention in attentions:
            probs = layer_attention[batch_index]
            same_mass = (probs * same_section.to(probs.dtype)[None, :, :]).sum(dim=-1)
            layer_scores.append(float(same_mass[:, valid].mean().item()))
        per_window.append(layer_scores)
    return per_window


def attention_diagnostics(
    attentions: tuple[torch.Tensor, ...],
    labels: torch.Tensor,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    *,
    sep_token_id: int,
) -> list[dict[str, float]]:
    rows = []
    for batch_index in range(labels.shape[0]):
        label_row = labels[batch_index]
        valid = label_row > 0
        token_indices = torch.nonzero(valid, as_tuple=False).flatten()
        same = (label_row[:, None] == label_row[None, :]) & valid[:, None] & valid[None, :]
        same.fill_diagonal_(False)
        nonself_content = valid[:, None] & valid[None, :]
        nonself_content.fill_diagonal_(False)
        section_sizes = torch.stack([(label_row == section).sum() for section in (1, 2, 3)])
        expected_by_query = torch.where(
            section_sizes[label_row[token_indices] - 1] > 1,
            (section_sizes[label_row[token_indices] - 1] - 1).float()
            / max(int(valid.sum().item()) - 1, 1),
            torch.zeros_like(section_sizes[label_row[token_indices]], dtype=torch.float32),
        )
        sep_positions = (input_ids[batch_index] == sep_token_id) & attention_mask[batch_index].bool()
        metric_values = []
        normalized_values = []
        uniform_values = []
        cls_values = []
        sep_values = []
        for layer_attention in attentions:
            probabilities = layer_attention[batch_index]
            content_pair_mask = nonself_content.to(probabilities.dtype)[None, :, :]
            total_content = (probabilities * content_pair_mask).sum(dim=-1)[:, token_indices]
            within_content = (
                probabilities * same.to(probabilities.dtype)[None, :, :]
            ).sum(dim=-1)[:, token_indices]
            metric_values.append(float(within_content.mean().item()))
            normalized_values.append(
                float((within_content / total_content.clamp_min(1e-12)).mean().item())
            )
            uniform_values.append(float(expected_by_query.mean().item()))
            cls_values.append(float(probabilities[:, token_indices, 0].mean().item()))
            if sep_positions.any():
                sep_values.append(
                    float(probabilities[:, token_indices][:, :, sep_positions].sum(dim=-1).mean().item())
                )
            else:
                sep_values.append(0.0)
        rows.append(
            {
                "off_diagonal_section_mass": statistics.mean(metric_values),
                "section_mass_renormalized_over_nonself_content": statistics.mean(normalized_values),
                "size_matched_uniform_expectation": statistics.mean(uniform_values),
                "cls_sink_mass": statistics.mean(cls_values),
                "sep_sink_mass": statistics.mean(sep_values),
            }
        )
    return rows


def count_attention_calls(model: Any) -> tuple[dict[str, int], list[tuple[Any, Any]]]:
    counter = {"calls": 0}
    originals = []
    for layer in model.encoder.layer:
        attention = layer.attention.self
        original = attention.forward
        originals.append((attention, original))

        def wrapped(self: Any, *args: Any, _forward: Any = original, **kwargs: Any) -> Any:
            counter["calls"] += 1
            return _forward(*args, **kwargs)

        attention.forward = wrapped.__get__(attention, type(attention))
    return counter, originals


def restore_attention_calls(originals: list[tuple[Any, Any]]) -> None:
    for attention, original in originals:
        attention.forward = original


def run_unit_tests() -> dict[str, Any]:
    package_python = PACKAGE_ROOT / ".venv" / "Scripts" / "python.exe"
    with tempfile.TemporaryDirectory(prefix="hise-component-tests-") as temp:
        junit = Path(temp) / "pytest.xml"
        result = subprocess.run(
            [
                str(package_python),
                "-m",
                "pytest",
                "-q",
                f"--junitxml={junit}",
                str(PACKAGE_ROOT / "tests"),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        root = ET.parse(junit).getroot() if junit.exists() else None
        per_file: dict[str, dict[str, int]] = {}
        for case in root.findall(".//testcase") if root is not None else []:
            file_path = case.attrib.get("file")
            if file_path:
                name = Path(file_path).name
            else:
                modules = case.attrib.get("classname", "").split(".")
                test_module = next((part for part in modules if part.startswith("test_")), "unknown")
                name = f"{test_module}.py" if test_module != "unknown" else "unknown"
            counts = per_file.setdefault(name, {"passed": 0, "failed": 0, "skipped": 0})
            if case.find("failure") is not None or case.find("error") is not None:
                counts["failed"] += 1
            elif case.find("skipped") is not None:
                counts["skipped"] += 1
            else:
                counts["passed"] += 1
        return {
            "exit_code": result.returncode,
            "passed": sum(row["passed"] for row in per_file.values()),
            "failed": sum(row["failed"] for row in per_file.values()),
            "skipped": sum(row["skipped"] for row in per_file.values()),
            "per_file": per_file,
            "output_tail": (result.stdout + "\n" + result.stderr).splitlines()[-12:],
        }


def run_notebooks() -> dict[str, Any]:
    results = []
    for path in sorted((REPO_ROOT / "notebooks").glob("*.ipynb")):
        notebook = nbformat.read(path, as_version=4)
        notebook.cells.insert(
            0,
            nbformat.v4.new_code_cell(
                "import sys, json; print('HISE_KERNEL=' + json.dumps({'executable': sys.executable}))"
            ),
        )
        try:
            executed = NotebookClient(notebook, timeout=600, kernel_name="python3").execute()
            executable = None
            for output in executed.cells[0].get("outputs", []):
                for line in output.get("text", "").splitlines():
                    if line.startswith("HISE_KERNEL="):
                        executable = json.loads(line.split("=", 1)[1])["executable"]
            results.append({"notebook": path.name, "passed": True, "kernel_executable": executable})
        except Exception as error:
            results.append(
                {"notebook": path.name, "passed": False, "kernel_executable": None, "error": str(error)}
            )
    return {
        "passed": sum(item["passed"] for item in results),
        "failed": sum(not item["passed"] for item in results),
        "notebooks": results,
    }


def measure(model: Any, inputs: dict[str, torch.Tensor]) -> dict[str, float | int]:
    with torch.inference_mode():
        for _ in range(5):
            model(**inputs)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    timings = []
    with torch.inference_mode():
        for _ in range(20):
            torch.cuda.synchronize()
            started = time.perf_counter()
            model(**inputs)
            torch.cuda.synchronize()
            timings.append((time.perf_counter() - started) * 1000)
    return {
        "warmups": 5,
        "repeats": 20,
        "median_ms": round(statistics.median(timings), 4),
        "p95_ms": round(float(np.percentile(timings, 95)), 4),
        "peak_allocated_mb": round(torch.cuda.max_memory_allocated() / (1024**2), 2),
        "peak_reserved_mb": round(torch.cuda.max_memory_reserved() / (1024**2), 2),
    }


def full_length_input(tokenizer: Any, batch: int, device: torch.device) -> dict[str, torch.Tensor]:
    base = tokenizer.encode(
        "The company reported results and reviewed its operating plan.", add_special_tokens=False
    )
    ids = [tokenizer.cls_token_id]
    while len(ids) < 511:
        ids.extend(base)
    ids = ids[:511] + [tokenizer.sep_token_id]
    input_ids = torch.tensor(ids, device=device).unsqueeze(0).expand(batch, -1).contiguous()
    return {"input_ids": input_ids, "attention_mask": torch.ones_like(input_ids)}


def gradient_step(shared: bool, model_id: str, device: torch.device) -> dict[str, Any]:
    tokenizer, model = load_model(model_id, device, implementation="eager")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    inputs, addresses, labels, _ = tokenize_windows(tokenizer, device)
    inputs = {key: value[:2] for key, value in inputs.items()}
    addresses = labels_to_addresses(addresses[:2], labels[:2], shuffle=False)
    attach_sgam_to_bert_attention(model, token_addresses=addresses, shared=shared)
    modules = [layer.attention.self.sgam for layer in model.encoder.layer]
    unique_modules = {id(module): module for module in modules}
    with torch.no_grad():
        for module in unique_modules.values():
            module.depth_weight.zero_()
            module.parent_weight.zero_()
            module.sibling_weight.zero_()
    model.train()
    sgam_parameters = [
        parameter for module in unique_modules.values() for parameter in module.parameters()
    ]
    optimizer = torch.optim.AdamW(
        [{"params": sgam_parameters, "lr": THRESHOLDS["sgam_learning_rate"], "weight_decay": 0.0}],
        betas=(0.9, 0.999),
    )
    optimizer.zero_grad(set_to_none=True)
    loss = model(**inputs).last_hidden_state[:, 1:, :].square().mean()
    loss.backward()
    rows = []
    seen = set()
    for layer_index, module in enumerate(modules):
        if id(module) in seen:
            continue
        seen.add(id(module))
        for name in ("depth_weight", "parent_weight", "sibling_weight"):
            parameter = getattr(module, name)
            grad = float(parameter.grad.detach().cpu())
            rows.append(
                {
                    "layer": "shared" if shared else layer_index,
                    "name": name,
                    "before": float(parameter.detach().cpu()),
                    "gradient": grad,
                    "nonzero": abs(grad) > 1e-12,
                }
            )
    optimizer.step()
    for row in rows:
        module = modules[0] if shared else modules[int(row["layer"])]
        row["after"] = float(getattr(module, row["name"]).detach().cpu())
    report = {
        "mode": "shared" if shared else "per_layer",
        "unique_parameters": sum(p.numel() for module in unique_modules.values() for p in module.parameters()),
        "expected_parameters": 3 if shared else 36,
        "nonzero_gradient_count": sum(row["nonzero"] for row in rows),
        "checked_gradient_count": len(rows),
        "encoder_frozen": all(
            not parameter.requires_grad
            for name, parameter in model.named_parameters()
            if ".sgam." not in name
        ),
        "encoder_trainable_parameter_count": sum(
            parameter.numel()
            for name, parameter in model.named_parameters()
            if ".sgam." not in name and parameter.requires_grad
        ),
        "sgam_trainable_parameter_count": sum(parameter.numel() for parameter in sgam_parameters),
        "optimizer": "AdamW(lr=0.05, betas=(0.9, 0.999), weight_decay=0)",
        "sgam_learning_rate": THRESHOLDS["sgam_learning_rate"],
        "initialization": THRESHOLDS["gradient_init"],
        "step_executed": True,
        "loss": float(loss.detach().cpu()),
        "parameters": rows,
    }
    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    return report


def write_report(report: dict[str, Any]) -> tuple[Path, Path]:
    output_dir = REPO_ROOT / "Dcoumentation" / "reports"
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "evidence_report.json"
    markdown_path = output_dir / "evidence_report.md"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    lines = [
        "# Component validation",
        "",
        f"- Model: `{report['model']['id']}` @ `{report['model']['revision']}`",
        f"- Device: {report['environment']['gpu']}; CUDA {report['environment']['cuda']}",
        f"- Python: `{report['environment']['python_executable']}` ({report['environment']['python_version']})",
        f"- Packages: torch {report['environment']['torch']}, Transformers {report['environment']['transformers']}, NumPy {report['environment']['numpy']}, pandas {report['environment']['pandas']}",
        f"- Git: `{report['environment']['git_commit']}`; seed: {report['environment']['seed']}",
        "",
        "## Tests and notebooks",
        "",
        f"- Pytest: {report['unit_tests']['passed']} passed, {report['unit_tests']['failed']} failed, {report['unit_tests']['skipped']} skipped",
        f"- Notebooks: {report['notebooks']['passed']} passed, {report['notebooks']['failed']} failed",
        "",
        "| Notebook | Result | Kernel executable |",
        "|---|---|---|",
    ]
    for item in report["notebooks"]["notebooks"]:
        lines.append(f"| {item['notebook']} | {'PASS' if item['passed'] else 'FAIL'} | `{item['kernel_executable']}` |")
    lines.extend(
        ["", "Per-file pytest results:", "", "| Test file | Passed | Failed | Skipped |", "|---|---:|---:|---:|"]
    )
    for name, counts in sorted(report["unit_tests"]["per_file"].items()):
        lines.append(
            f"| {name} | {counts['passed']} | {counts['failed']} | {counts['skipped']} |"
        )
    lines.extend(
        [
            "",
            "## Hook and equivalence",
            "",
            f"- Default: `{report['hook']['default_class']}` / {report['hook']['default_implementation']}; after attach: `{report['hook']['class_after_attach']}` / `{report['hook']['implementation_after_attach']}`",
            f"- Hook calls: {report['hook']['calls_first_forward']} first load; {report['hook']['calls_after_fresh_reload']} after reload (expected {report['hook']['expected_calls']} each)",
            f"- Train-mode attention dropout calls: {report['hook']['training_dropout']['attention_dropout_calls']} (expected {report['hook']['training_dropout']['expected_calls']}); same-seed stock-vs-hook max diff: {report['hook']['training_dropout']['stock_vs_zero_hook_same_seed_max_abs_diff']:.8g}; different-seed stock output diff: {report['hook']['training_dropout']['stock_different_seed_max_output_diff']:.8g}",
            f"- 20 padded sentences, shape `{report['equivalence']['batch_shape']}`",
            f"- Non-pad hidden-state max-abs-diff min/mean/max: {report['equivalence']['hidden_diff']['min']:.8g} / {report['equivalence']['hidden_diff']['mean']:.8g} / {report['equivalence']['hidden_diff']['max']:.8g}",
            f"- CLS cosine min/mean/max: {report['equivalence']['cls_cosine']['min']:.9f} / {report['equivalence']['cls_cosine']['mean']:.9f} / {report['equivalence']['cls_cosine']['max']:.9f}",
            f"- Mean-pool cosine min/mean/max: {report['equivalence']['mean_pool_cosine']['min']:.9f} / {report['equivalence']['mean_pool_cosine']['mean']:.9f} / {report['equivalence']['mean_pool_cosine']['max']:.9f}",
            "",
            "## Attention shift",
            "",
            "Tokenizer offsets map actual text spans to sections. The primary metric excludes special, padded, and self tokens; scores average over text queries, heads, and layers. The legacy metric including self is included per row for comparison.",
            "",
            "| Parameter | Weight | Window | Zero off-diagonal | Biased | Gain | Gain incl. self | CLS cosine |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for grid in report["attention_shift"]["grid"]:
        for row in grid["windows"]:
            lines.append(
                f"| {grid['parameter']} | {grid['weight']} | {row['window']} | {row['zero_mass']:.6f} | {row['biased_mass']:.6f} | "
                f"{row['gain']:.6f} | {row['gain_including_diagonal']:.6f} | {row['cls_cosine']:.6f} |"
            )
    lines.extend(
        [
            "",
            f"- Legacy shuffled-label gains including self: `{report['attention_shift']['shuffled_control']['legacy_including_diagonal_gains']}`.",
            f"- Corrected size-matched shuffled-label true-section gains excluding self: `{report['attention_shift']['shuffled_control']['diagonal_excluded_true_section_gains']}`; max {report['attention_shift']['shuffled_control']['max_gain']:.6f} vs unchanged threshold {report['attention_shift']['shuffled_control']['threshold']:.3f}; passed: {report['attention_shift']['shuffled_control']['passed']}.",
            f"- Zero-control maximum mass difference: {report['attention_shift']['zero_control_max_difference']:.9g}.",
            "",
            "Zero-weight attention diagnostics (section mass renormalizes over non-self text tokens; uniform expectation is size-matched; CLS/SEP sink mass is reported separately):",
            "",
            "| Window | Off-diagonal mass | Renormalized | Uniform expectation | CLS sink | SEP sink |",
            "|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for index, item in enumerate(report["attention_shift"]["zero_weight_diagnostics"], start=1):
        lines.append(
            f"| {index} | {item['off_diagonal_section_mass']:.6f} | {item['section_mass_renormalized_over_nonself_content']:.6f} | "
            f"{item['size_matched_uniform_expectation']:.6f} | {item['cls_sink_mass']:.6f} | {item['sep_sink_mass']:.6f} |"
        )
    lines.extend(["", "Per-layer same-section mass (12 values per window, zero vs biased):", ""])
    for grid in report["attention_shift"]["grid"]:
        lines.append(f"- {grid['parameter']} = {grid['weight']}:")
        for row in grid["windows"]:
            zero_layers = [round(value, 6) for value in row["zero_bias_per_layer"]]
            biased_layers = [round(value, 6) for value in row["per_layer"]]
            lines.append(
                f"  - Window {row['window']}: zero `{zero_layers}`; biased `{biased_layers}`"
            )
    lines.extend(
        [
            "",
            "## Gradients and optimizer step",
            "",
            "| Mode | Params | Nonzero grads | Encoder frozen | Step executed | Loss |",
            "|---|---:|---:|---|---|---:|",
        ]
    )
    for mode in report["gradients"]:
        lines.append(
            f"| {mode['mode']} | {mode['unique_parameters']} | {mode['nonzero_gradient_count']}/{mode['checked_gradient_count']} | "
            f"{mode['encoder_frozen']} | {mode['step_executed']} | {mode['loss']:.8g} |"
        )
        lines.append(
            f"  - Init `{mode['initialization']}`, encoder trainable params `{mode['encoder_trainable_parameter_count']}`, "
            f"SGAM LR `{mode['sgam_learning_rate']}`, optimizer `{mode['optimizer']}`."
        )
    lines.extend(["", "Per-parameter values and gradients:", ""])
    for mode in report["gradients"]:
        lines.extend(
            [
                f"### {mode['mode']}",
                "",
                "| Layer | Parameter | Before | Gradient | After step |",
                "|---:|---|---:|---:|---:|",
            ]
        )
        for item in mode["parameters"]:
            lines.append(
                f"| {item['layer']} | {item['name']} | {item['before']:.8g} | {item['gradient']:.8g} | {item['after']:.8g} |"
            )
    lines.extend(
        [
            "",
            "## GPU overhead: 512 tokens, 5 warmups and 20 timed forwards",
            "",
            "| Arm | Batch | Median ms | P95 ms | Peak allocated MB | Peak reserved MB |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for arm, rows in report["overhead"].items():
        for item in rows:
            measure_result = item["measure"]
            lines.append(
                f"| {arm} | {item['batch_size']} | {measure_result['median_ms']:.4f} | {measure_result['p95_ms']:.4f} | "
                f"{measure_result['peak_allocated_mb']:.2f} | {measure_result['peak_reserved_mb']:.2f} |"
            )
    lines.extend(["", "## Gates", ""])
    lines.extend(f"- `{name}`: **{'PASS' if passed else 'FAIL'}**" for name, passed in report["gates"].items())
    lines.extend(
        [
            "",
            "## Not tested",
            "",
            "- Retrieval quality (MRR/nDCG) on a held-out structure-aware benchmark.",
            "- Training convergence beyond the reported one-step dummy-loss gradient check.",
            "- Cross-domain transfer, production throughput, and energy consumption.",
            "- The five attention windows are manually authored synthetic examples, not retrieval evidence.",
            "",
            f"Raw results: `{json_path.name}`",
            "",
        ]
    )
    markdown_path.write_text("\n".join(lines), encoding="utf-8")
    return json_path, markdown_path


def main() -> None:
    seed_everything()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this evidence run.")
    device = torch.device("cuda")
    model_id = os.environ.get("HISE_MODEL_ID", MODEL_ID)
    from huggingface_hub import snapshot_download

    snapshot_path = Path(snapshot_download(model_id, local_files_only=True))
    report: dict[str, Any] = {
        "report_title": "Component validation",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "thresholds_fixed_before_run": THRESHOLDS,
        "environment": {
            "python_executable": sys.executable,
            "python_version": sys.version,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
            "seed": SEED,
            "git_commit": git_value("rev-parse", "HEAD"),
            "model_snapshot_path": str(snapshot_path),
        },
        "unit_tests": run_unit_tests(),
        "notebooks": run_notebooks(),
    }

    tokenizer, stock_sdpa = load_model(model_id, device)
    report["model"] = {
        "id": model_id,
        "revision": snapshot_path.name,
        "class": type(stock_sdpa).__name__,
        "layers": len(stock_sdpa.encoder.layer),
    }
    default_attention_class = type(stock_sdpa.encoder.layer[0].attention.self).__name__
    default_implementation = getattr(stock_sdpa.config, "_attn_implementation", None)
    padded = tokenizer(FIXED_SENTENCES, padding=True, truncation=True, max_length=128, return_tensors="pt")
    padded = {key: value.to(device) for key, value in padded.items()}
    with torch.inference_mode():
        stock_default_output = stock_sdpa(**padded).last_hidden_state

    eager_tokenizer, eager_model = load_model(model_id, device, implementation="eager")
    preattach_class = type(eager_model.encoder.layer[0].attention.self).__name__
    window_inputs, window_addresses, window_labels, offset_records = tokenize_windows(eager_tokenizer, device)
    dropout_inputs = {key: value[:2] for key, value in padded.items()}
    eager_model.train()
    torch.manual_seed(SEED)
    with torch.no_grad():
        stock_train_output = eager_model(**dropout_inputs).last_hidden_state
    torch.manual_seed(SEED + 1)
    with torch.no_grad():
        different_seed_train_output = eager_model(**dropout_inputs).last_hidden_state
    stock_dropout_seed_sensitivity = float(
        (stock_train_output - different_seed_train_output).abs().max().item()
    )
    eager_model.eval()
    zero_addresses = torch.zeros(
        (len(FIXED_SENTENCES), padded["input_ids"].shape[1], 4), dtype=torch.float32, device=device
    )
    positions = torch.arange(padded["input_ids"].shape[1], device=device)
    zero_addresses[:, :, 0] = positions
    zero_addresses[:, :, 2] = positions // 8
    attach_sgam_to_bert_attention(eager_model, token_addresses=zero_addresses)
    for layer in eager_model.encoder.layer:
        module = layer.attention.self.sgam
        with torch.no_grad():
            module.depth_weight.zero_()
            module.parent_weight.zero_()
            module.sibling_weight.zero_()
    dropout_counts = {"calls": 0}
    dropout_handles = []
    for layer in eager_model.encoder.layer:
        dropout = getattr(layer.attention.self, "dropout", None)
        if dropout is not None:
            dropout_handles.append(
                dropout.register_forward_hook(
                    lambda _module, _inputs, _output: dropout_counts.__setitem__(
                        "calls", dropout_counts["calls"] + 1
                    )
                )
            )
        layer.attention.self.sgam_addresses = zero_addresses[:2]
    eager_model.train()
    torch.manual_seed(SEED)
    with torch.no_grad():
        hooked_train_output = eager_model(**dropout_inputs).last_hidden_state
    for handle in dropout_handles:
        handle.remove()
    for layer in eager_model.encoder.layer:
        layer.attention.self.sgam_addresses = zero_addresses
    eager_model.eval()
    dropout_train_max_abs_diff = float((stock_train_output - hooked_train_output).abs().max().item())
    attached_class = type(eager_model.encoder.layer[0].attention.self).__name__
    call_counter, originals = count_attention_calls(eager_model)
    with torch.inference_mode():
        eager_model(**padded)
    restore_attention_calls(originals)

    call_counter_reload = {"calls": 0}
    reload_tokenizer, reloaded = load_model(model_id, device)
    reload_class = type(reloaded.encoder.layer[0].attention.self).__name__
    attach_sgam_to_bert_attention(reloaded, token_addresses=zero_addresses[:1])
    reload_counter, reload_originals = count_attention_calls(reloaded)
    with torch.inference_mode():
        reloaded(**{key: value[:1] for key, value in padded.items()})
    restore_attention_calls(reload_originals)
    call_counter_reload["calls"] = reload_counter["calls"]
    report["hook"] = {
        "default_class": default_attention_class,
        "default_implementation": default_implementation,
        "class_before_attach": preattach_class,
        "class_after_attach": attached_class,
        "implementation_after_attach": getattr(eager_model.config, "_attn_implementation", None),
        "calls_first_forward": call_counter["calls"],
        "fresh_reload_class_before_attach": reload_class,
        "calls_after_fresh_reload": call_counter_reload["calls"],
        "expected_calls": len(eager_model.encoder.layer),
        "training_dropout": {
            "attention_dropout_calls": dropout_counts["calls"],
            "expected_calls": len(eager_model.encoder.layer),
            "stock_different_seed_max_output_diff": stock_dropout_seed_sensitivity,
            "stock_vs_zero_hook_same_seed_max_abs_diff": dropout_train_max_abs_diff,
        },
    }
    del reloaded, reload_tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    attention_mask = padded["attention_mask"]
    with torch.inference_mode():
        eager_zero_output = eager_model(**padded).last_hidden_state
    zero_sentence_metrics = []
    for index, sentence in enumerate(FIXED_SENTENCES):
        valid = attention_mask[index].bool()
        stock_values = stock_default_output[index, valid]
        eager_values = eager_zero_output[index, valid]
        stock_mean = stock_values.mean(dim=0)
        eager_mean = eager_values.mean(dim=0)
        zero_sentence_metrics.append(
            {
                "index": index + 1,
                "sentence": sentence,
                "nonpad_tokens": int(valid.sum().item()),
                "hidden_max_abs_diff_nonpad": float((stock_values - eager_values).abs().max().item()),
                "cls_cosine": float(
                    F.cosine_similarity(
                        stock_default_output[index:index + 1, 0],
                        eager_zero_output[index:index + 1, 0],
                    ).item()
                ),
                "mean_pool_cosine": float(F.cosine_similarity(stock_mean, eager_mean, dim=0).item()),
            }
        )
    diff_values = [item["hidden_max_abs_diff_nonpad"] for item in zero_sentence_metrics]
    cls_values = [item["cls_cosine"] for item in zero_sentence_metrics]
    mean_values = [item["mean_pool_cosine"] for item in zero_sentence_metrics]
    report["equivalence"] = {
        "sentence_count": len(FIXED_SENTENCES),
        "batch_shape": list(padded["input_ids"].shape),
        "padding_used": bool((attention_mask == 0).any()),
        "per_sentence": zero_sentence_metrics,
        "hidden_diff": {
            "min": min(diff_values),
            "mean": statistics.mean(diff_values),
            "max": max(diff_values),
        },
        "cls_cosine": {"min": min(cls_values), "mean": statistics.mean(cls_values), "max": max(cls_values)},
        "mean_pool_cosine": {
            "min": min(mean_values),
            "mean": statistics.mean(mean_values),
            "max": max(mean_values),
        },
    }

    true_addresses = labels_to_addresses(window_addresses, window_labels, shuffle=False)
    shuffled_addresses = labels_to_addresses(window_addresses, window_labels, shuffle=True)
    for layer in eager_model.encoder.layer:
        layer.attention.self.sgam_addresses = true_addresses
        module = layer.attention.self.sgam
        with torch.no_grad():
            module.depth_weight.zero_()
            module.parent_weight.zero_()
            module.sibling_weight.zero_()
    with torch.inference_mode():
        zero_attention_output = eager_model(**window_inputs, output_attentions=True)
    zero_layer_mass_legacy = within_section_by_layer(zero_attention_output.attentions, window_labels)
    zero_layer_mass = within_section_by_layer(
        zero_attention_output.attentions, window_labels, exclude_diagonal=True
    )
    zero_diagnostics = attention_diagnostics(
        zero_attention_output.attentions,
        window_labels,
        window_inputs["input_ids"],
        window_inputs["attention_mask"],
        sep_token_id=eager_tokenizer.sep_token_id,
    )
    zero_cls = zero_attention_output.last_hidden_state[:, 0, :].detach()
    with torch.inference_mode():
        zero_repeat = eager_model(**window_inputs, output_attentions=True)
    zero_repeat_mass = within_section_by_layer(
        zero_repeat.attentions, window_labels, exclude_diagonal=True
    )
    zero_control_difference = max(
        abs(first - repeated)
        for row_a, row_b in zip(zero_layer_mass, zero_repeat_mass)
        for first, repeated in zip(row_a, row_b)
    )
    grid_rows = []
    for parameter_name in THRESHOLDS["attention_parameters"]:
        for weight in THRESHOLDS["attention_weight_grid"]:
            for layer in eager_model.encoder.layer:
                module = layer.attention.self.sgam
                layer.attention.self.sgam_addresses = true_addresses
                with torch.no_grad():
                    module.depth_weight.zero_()
                    module.parent_weight.zero_()
                    module.sibling_weight.zero_()
                    getattr(module, parameter_name).fill_(weight)
            with torch.inference_mode():
                output = eager_model(**window_inputs, output_attentions=True)
            per_layer = within_section_by_layer(
                output.attentions, window_labels, exclude_diagonal=True
            )
            per_layer_legacy = within_section_by_layer(output.attentions, window_labels)
            diagnostics = attention_diagnostics(
                output.attentions,
                window_labels,
                window_inputs["input_ids"],
                window_inputs["attention_mask"],
                sep_token_id=eager_tokenizer.sep_token_id,
            )
            cls_cosines = F.cosine_similarity(zero_cls, output.last_hidden_state[:, 0, :], dim=-1)
            windows = []
            for window_index in range(len(WINDOW_SECTIONS)):
                zero_mass = statistics.mean(zero_layer_mass[window_index])
                biased_mass = statistics.mean(per_layer[window_index])
                windows.append(
                    {
                        "window": window_index + 1,
                        "token_count": offset_records[window_index]["token_count_including_specials"],
                        "zero_mass": zero_mass,
                        "biased_mass": biased_mass,
                        "gain": biased_mass - zero_mass,
                        "zero_mass_including_diagonal": statistics.mean(
                            zero_layer_mass_legacy[window_index]
                        ),
                        "biased_mass_including_diagonal": statistics.mean(
                            per_layer_legacy[window_index]
                        ),
                        "gain_including_diagonal": statistics.mean(per_layer_legacy[window_index])
                        - statistics.mean(zero_layer_mass_legacy[window_index]),
                        "cls_cosine": float(cls_cosines[window_index].item()),
                        "per_layer": per_layer[window_index],
                        "zero_bias_per_layer": zero_layer_mass[window_index],
                        "diagnostics": diagnostics[window_index],
                    }
                )
            grid_rows.append(
                {
                    "parameter": parameter_name,
                    "weight": weight,
                    "windows": windows,
                    "windows_meeting_gate": sum(
                        row["gain"] >= THRESHOLDS["attention_shift_gain"]
                        and row["cls_cosine"] >= THRESHOLDS["attention_shift_cls_cosine"]
                        for row in windows
                    ),
                    "mean_gain": statistics.mean(row["gain"] for row in windows),
                }
            )

    for layer in eager_model.encoder.layer:
        layer.attention.self.sgam_addresses = shuffled_addresses
        module = layer.attention.self.sgam
        with torch.no_grad():
            module.depth_weight.zero_()
            module.parent_weight.fill_(-1.0)
            module.sibling_weight.zero_()
    with torch.inference_mode():
        shuffled_output = eager_model(**window_inputs, output_attentions=True)
    shuffled_layer_mass_legacy = within_section_by_layer(shuffled_output.attentions, window_labels)
    shuffled_layer_mass = within_section_by_layer(
        shuffled_output.attentions, window_labels, exclude_diagonal=True
    )
    shuffled_legacy_gains = [
        statistics.mean(shuffled_layer_mass_legacy[i]) - statistics.mean(zero_layer_mass_legacy[i])
        for i in range(len(WINDOW_SECTIONS))
    ]
    shuffled_gains = [
        statistics.mean(shuffled_layer_mass[i]) - statistics.mean(zero_layer_mass[i])
        for i in range(len(WINDOW_SECTIONS))
    ]
    report["attention_shift"] = {
        "metric": "for each valid text token, sum attention probabilities to non-self valid text tokens in its true section; average queries, heads, then layers",
        "address_source": "fast tokenizer offset_mapping intersected with each handwritten section character span",
        "offset_mapping": offset_records,
        "grid": grid_rows,
        "zero_control_max_difference": zero_control_difference,
        "zero_weight_diagnostics": zero_diagnostics,
        "shuffled_control": {
            "seed": SEED,
            "parent_weight": -1.0,
            "randomization": "uniform random permutation of the true section labels within each window; exact section counts preserved",
            "legacy_including_diagonal_gains": shuffled_legacy_gains,
            "diagonal_excluded_true_section_gains": shuffled_gains,
            "max_gain": max(shuffled_gains),
            "threshold": THRESHOLDS["shuffled_true_section_max_gain"],
            "passed": max(shuffled_gains) <= THRESHOLDS["shuffled_true_section_max_gain"],
        },
    }

    del eager_model, eager_tokenizer, stock_sdpa, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    report["gradients"] = [
        gradient_step(shared=True, model_id=model_id, device=device),
        gradient_step(shared=False, model_id=model_id, device=device),
    ]

    report["overhead"] = {}
    for arm in ("stock_sdpa", "stock_eager", "hooked"):
        impl = None if arm == "stock_sdpa" else "eager"
        bench_tokenizer, bench_model = load_model(model_id, device, implementation=impl)
        if arm == "hooked":
            bench_addresses = torch.zeros((32, 512, 4), dtype=torch.float32, device=device)
            for token in range(512):
                section = min(3, token // 171 + 1)
                bench_addresses[:, token] = torch.tensor([token, 1, section, 0], device=device)
            attach_sgam_to_bert_attention(bench_model, token_addresses=bench_addresses)
            for layer in bench_model.encoder.layer:
                module = layer.attention.self.sgam
                with torch.no_grad():
                    module.depth_weight.zero_()
                    module.parent_weight.zero_()
                    module.sibling_weight.zero_()
        results = []
        for batch_size in (1, 8, 32):
            inputs = full_length_input(bench_tokenizer, batch_size, device)
            if arm == "hooked":
                for layer in bench_model.encoder.layer:
                    layer.attention.self.sgam_addresses = bench_addresses[:batch_size]
            results.append(
                {
                    "batch_size": batch_size,
                    "sequence_length": 512,
                    "measure": measure(bench_model, inputs),
                }
            )
        report["overhead"][arm] = results
        del bench_model, bench_tokenizer
        gc.collect()
        torch.cuda.empty_cache()

    report["gates"] = {
        "unit_tests": report["unit_tests"]["exit_code"] == 0 and report["unit_tests"]["failed"] == 0,
        "notebooks": report["notebooks"]["failed"] == 0 and report["notebooks"]["passed"] == 4,
        "hook_all_layers_first_and_reload": report["hook"]["calls_first_forward"] == 12
        and report["hook"]["calls_after_fresh_reload"] == 12,
        "train_mode_dropout_preserved": report["hook"]["training_dropout"]["attention_dropout_calls"] == 12
        and report["hook"]["training_dropout"]["stock_vs_zero_hook_same_seed_max_abs_diff"] <= 1e-5
        and report["hook"]["training_dropout"]["stock_different_seed_max_output_diff"] > 0,
        "zero_weight_equivalence": report["equivalence"]["hidden_diff"]["max"]
        <= THRESHOLDS["zero_weight_max_abs_diff"]
        and report["equivalence"]["cls_cosine"]["min"] >= THRESHOLDS["zero_weight_cls_cosine"]
        and report["equivalence"]["mean_pool_cosine"]["min"] >= THRESHOLDS["zero_weight_mean_pool_cosine"],
        "attention_shift": max(row["windows_meeting_gate"] for row in grid_rows)
        >= THRESHOLDS["attention_shift_windows"],
        "shuffled_control": report["attention_shift"]["shuffled_control"]["passed"],
        "zero_weight_control": zero_control_difference == 0.0,
        "shared_gradient_step": report["gradients"][0]["unique_parameters"] == 3
        and report["gradients"][0]["nonzero_gradient_count"] == report["gradients"][0]["checked_gradient_count"]
        and report["gradients"][0]["encoder_frozen"]
        and report["gradients"][0]["encoder_trainable_parameter_count"] == 0
        and all(item["before"] == 0.0 for item in report["gradients"][0]["parameters"]),
        "per_layer_gradient_step": report["gradients"][1]["unique_parameters"] == 36
        and report["gradients"][1]["nonzero_gradient_count"] == report["gradients"][1]["checked_gradient_count"]
        and report["gradients"][1]["encoder_frozen"]
        and report["gradients"][1]["encoder_trainable_parameter_count"] == 0
        and all(item["before"] == 0.0 for item in report["gradients"][1]["parameters"]),
        "three_arm_overhead_recorded": set(report["overhead"]) == {"stock_sdpa", "stock_eager", "hooked"}
        and all(len(rows) == 3 for rows in report["overhead"].values()),
    }
    json_path, markdown_path = write_report(report)
    print(f"JSON report: {json_path}")
    print(f"Markdown report: {markdown_path}")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
