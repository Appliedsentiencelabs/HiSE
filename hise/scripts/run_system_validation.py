"""Run empirical HiSE system gates against the cached real BGE checkpoint."""

from __future__ import annotations

import gc
import json
import math
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

os.environ.setdefault("USE_TF", "0")

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hise.bert_sgam import attach_sgam_to_bert_attention  # noqa: E402

MODEL_ID = "BAAI/bge-base-en-v1.5"
SENTENCES = [
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


def load_model(model_id: str, *, device: torch.device) -> tuple[Any, Any]:
    tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
    model = AutoModel.from_pretrained(model_id, local_files_only=True).to(device).eval()
    return tokenizer, model


def hidden_states(model: Any, encoded: dict[str, torch.Tensor], *, output_attentions: bool = False) -> Any:
    with torch.inference_mode():
        return model(**encoded, output_attentions=output_attentions)


def tokenize_windows(
    tokenizer: Any, *, device: torch.device
) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor]:
    rows: list[list[int]] = []
    section_rows: list[list[int]] = []
    for sections in WINDOW_SECTIONS:
        ids = [tokenizer.cls_token_id]
        section_ids = [0]
        for section_index, section in enumerate(sections, start=1):
            section_tokens = tokenizer.encode(section, add_special_tokens=False)
            ids.extend(section_tokens)
            section_ids.extend([section_index] * len(section_tokens))
        ids.append(tokenizer.sep_token_id)
        section_ids.append(0)
        rows.append(ids)
        section_rows.append(section_ids)

    length = max(map(len, rows))
    input_ids = torch.full((len(rows), length), tokenizer.pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros_like(input_ids)
    addresses = torch.zeros((len(rows), length, 4), dtype=torch.float32)
    section_labels = torch.full((len(rows), length), -1, dtype=torch.long)
    for batch_index, (row, labels) in enumerate(zip(rows, section_rows)):
        row_len = len(row)
        input_ids[batch_index, :row_len] = torch.tensor(row)
        attention_mask[batch_index, :row_len] = 1
        section_labels[batch_index, :row_len] = torch.tensor(labels)
        for token_index, section_id in enumerate(labels):
            addresses[batch_index, token_index] = torch.tensor(
                [token_index, 1 if section_id else 0, section_id, 0], dtype=torch.float32
            )
    return (
        {
            "input_ids": input_ids.to(device),
            "attention_mask": attention_mask.to(device),
        },
        addresses.to(device),
        section_labels.to(device),
        attention_mask.to(device),
    )


def within_section_mass(
    attentions: tuple[torch.Tensor, ...],
    section_labels: torch.Tensor,
    attention_mask: torch.Tensor,
) -> list[float]:
    values: list[float] = []
    for batch_index in range(section_labels.shape[0]):
        label_row = section_labels[batch_index]
        query_valid = (label_row > 0) & attention_mask[batch_index].bool()
        key_valid = (label_row > 0) & attention_mask[batch_index].bool()
        same_section = label_row[:, None] == label_row[None, :]
        same_section &= key_valid[None, :]
        layer_values = []
        for layer_attention in attentions:
            probs = layer_attention[batch_index]
            mass = (probs * same_section.to(probs.dtype)[None, :, :]).sum(dim=-1)
            layer_values.append(mass[:, query_valid].mean().item())
        values.append(sum(layer_values) / len(layer_values))
    return values


def install_attention_call_counters(model: Any) -> tuple[dict[str, int], list[Any]]:
    calls = {"count": 0}
    originals = []
    for layer in model.encoder.layer:
        attention = layer.attention.self
        original = attention.forward
        originals.append((attention, original))

        def counted(self: Any, *args: Any, _original: Any = original, **kwargs: Any) -> Any:
            calls["count"] += 1
            return _original(*args, **kwargs)

        attention.forward = counted.__get__(attention, type(attention))
    return calls, originals


def restore_attention_call_counters(originals: list[Any]) -> None:
    for attention, original in originals:
        attention.forward = original


def make_full_length_input(tokenizer: Any, batch_size: int, device: torch.device) -> dict[str, torch.Tensor]:
    seed = tokenizer.encode(
        "The company reported results and reviewed its operating plan.",
        add_special_tokens=False,
    )
    token_ids = [tokenizer.cls_token_id]
    while len(token_ids) < 511:
        token_ids.extend(seed)
    token_ids = token_ids[:511] + [tokenizer.sep_token_id]
    input_ids = torch.tensor(token_ids, dtype=torch.long, device=device).unsqueeze(0).expand(batch_size, -1)
    return {
        "input_ids": input_ids.contiguous(),
        "attention_mask": torch.ones_like(input_ids),
    }


def measure_forward(model: Any, encoded: dict[str, torch.Tensor], *, repeats: int = 3) -> dict[str, float]:
    with torch.inference_mode():
        model(**encoded)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    elapsed: list[float] = []
    for _ in range(repeats):
        torch.cuda.synchronize()
        started = time.perf_counter()
        with torch.inference_mode():
            model(**encoded)
        torch.cuda.synchronize()
        elapsed.append(time.perf_counter() - started)
    return {
        "median_ms": round(1000 * sorted(elapsed)[len(elapsed) // 2], 3),
        "peak_allocated_mb": round(torch.cuda.max_memory_allocated() / (1024**2), 2),
        "peak_reserved_mb": round(torch.cuda.max_memory_reserved() / (1024**2), 2),
    }


def check_gradients(*, shared: bool, model_id: str, device: torch.device) -> dict[str, Any]:
    tokenizer, model = load_model(model_id, device=device)
    encoded, addresses, section_labels, _ = tokenize_windows(tokenizer, device=device)
    encoded = {key: value[:2, :48] for key, value in encoded.items()}
    addresses = addresses[:2, :48]
    section_labels = section_labels[:2, :48]
    for batch_index in range(section_labels.shape[0]):
        section_offsets: dict[int, int] = {}
        for token_index, section_id_tensor in enumerate(section_labels[batch_index]):
            section_id = int(section_id_tensor.item())
            if section_id > 0:
                addresses[batch_index, token_index, 1] = section_id
                addresses[batch_index, token_index, 3] = section_offsets.get(section_id, 0)
                section_offsets[section_id] = section_offsets.get(section_id, 0) + 1
    attach_sgam_to_bert_attention(model, token_addresses=addresses, shared=shared)
    modules = [layer.attention.self.sgam for layer in model.encoder.layer]
    unique_modules = {id(module): module for module in modules}
    with torch.no_grad():
        for module in unique_modules.values():
            module.depth_weight.fill_(-0.2)
            module.parent_weight.fill_(-0.2)
            module.sibling_weight.fill_(-0.2)
    model.zero_grad(set_to_none=True)
    output = model(**encoded).last_hidden_state
    loss = output[:, 1:, :].square().mean()
    loss.backward()
    parameters = []
    visited_modules: set[int] = set()
    for layer_index, module in enumerate(modules):
        if id(module) in visited_modules:
            continue
        visited_modules.add(id(module))
        for name in ("depth_weight", "parent_weight", "sibling_weight"):
            parameter = getattr(module, name)
            grad = None if parameter.grad is None else float(parameter.grad.detach().cpu())
            parameters.append(
                {
                    "layer": "shared" if shared else layer_index,
                    "parameter": name,
                    "value": float(parameter.detach().cpu()),
                    "gradient": grad,
                    "nonzero_gradient": grad is not None and math.isfinite(grad) and abs(grad) > 1e-12,
                }
            )
    unique_parameter_ids = {
        id(parameter)
        for module in unique_modules.values()
        for parameter in module.parameters()
    }
    result = {
        "mode": "shared" if shared else "per_layer",
        "unique_parameter_count": len(unique_parameter_ids),
        "expected_parameter_count": 3 if shared else 36,
        "nonzero_gradient_count": sum(item["nonzero_gradient"] for item in parameters),
        "checked_gradient_count": len(parameters),
        "parameters": parameters,
        "loss": float(loss.detach().cpu()),
    }
    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    return result


def run_project_checks() -> dict[str, Any]:
    package_python = ROOT / ".venv" / "Scripts" / "python.exe"
    tests = subprocess.run(
        [str(package_python), "-m", "pytest", "-v", "-rA", str(ROOT / "tests")],
        capture_output=True,
        text=True,
        check=False,
    )
    if tests.returncode:
        raise RuntimeError(f"Project unit tests failed:\n{tests.stdout}\n{tests.stderr}")

    notebooks = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "validate_notebooks.py")],
        capture_output=True,
        text=True,
        check=False,
    )
    if notebooks.returncode:
        raise RuntimeError(f"Notebook validation failed:\n{notebooks.stdout}\n{notebooks.stderr}")

    pytest_output = f"{tests.stdout}\n{tests.stderr}"
    pytest_match = re.search(r"(\d+) passed(?:,\s*(\d+) failed)?(?:,\s*(\d+) skipped)?", pytest_output)
    pytest_summary = (
        f"{pytest_match.group(1)} passed"
        + (f", {pytest_match.group(2)} failed" if pytest_match.group(2) else "")
        + (f", {pytest_match.group(3)} skipped" if pytest_match.group(3) else "")
        if pytest_match
        else "pytest passed"
    )
    notebook_summary = next(
        (line.strip() for line in notebooks.stdout.splitlines() if line.startswith("Validated ")),
        "notebook validation passed",
    )
    return {
        "unit_tests": pytest_summary,
        "notebooks": notebook_summary,
        "unit_test_exit_code": tests.returncode,
        "notebook_validation_exit_code": notebooks.returncode,
    }


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available; this validation requires the GPU workload.")
    device = torch.device("cuda")
    model_id = os.environ.get("HISE_MODEL_ID", MODEL_ID)
    report: dict[str, Any] = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "project_checks": run_project_checks(),
    }
    tokenizer, model = load_model(model_id, device=device)
    report.update({
        "model_id": model_id,
        "model_class": type(model).__name__,
        "transformers_version": __import__("transformers").__version__,
        "torch_version": torch.__version__,
        "device": torch.cuda.get_device_name(0),
        "cuda_version": torch.version.cuda,
        "default_attention_implementation": getattr(model.config, "_attn_implementation", None),
        "default_attention_class": type(model.encoder.layer[0].attention.self).__name__,
        "layer_count": len(model.encoder.layer),
    })

    encoded = tokenizer(SENTENCES, padding=True, truncation=True, max_length=128, return_tensors="pt")
    encoded = {key: value.to(device) for key, value in encoded.items()}
    torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        stock = model(**encoded).last_hidden_state
    stock_pool = stock[:, 0, :].clone()
    report["fixed_sentence_count"] = len(SENTENCES)
    report["padded_batch_shape"] = list(encoded["input_ids"].shape)
    report["padded_sentence_token_lengths"] = encoded["attention_mask"].sum(dim=1).cpu().tolist()

    stock_benchmark = []
    for batch_size in (1, 8, 32):
        full_input = make_full_length_input(tokenizer, batch_size, device)
        stock_benchmark.append(
            {
                "batch_size": batch_size,
                "sequence_length": full_input["input_ids"].shape[1],
                "measurement": measure_forward(model, full_input),
            }
        )

    window_inputs, addresses, section_labels, section_mask = tokenize_windows(tokenizer, device=device)
    attach_sgam_to_bert_attention(model, token_addresses=addresses, shared=True)
    for layer in model.encoder.layer:
        with torch.no_grad():
            layer.attention.self.sgam.depth_weight.zero_()
            layer.attention.self.sgam.parent_weight.zero_()
            layer.attention.self.sgam.sibling_weight.zero_()
    sentence_addresses = torch.zeros(
        (encoded["input_ids"].shape[0], encoded["input_ids"].shape[1], 4),
        dtype=torch.float32,
        device=device,
    )
    token_positions = torch.arange(encoded["input_ids"].shape[1], device=device)
    sentence_addresses[:, :, 0] = token_positions
    sentence_addresses[:, :, 1] = 1
    sentence_addresses[:, :, 2] = (token_positions // 8).unsqueeze(0)
    for layer in model.encoder.layer:
        layer.attention.self.sgam_addresses = sentence_addresses

    call_counter, originals = install_attention_call_counters(model)
    with torch.inference_mode():
        zero_output = model(**encoded).last_hidden_state
    restore_attention_call_counters(originals)
    report["hook_call_count"] = call_counter["count"]
    report["zero_weight_equivalence"] = []
    for index, sentence in enumerate(SENTENCES):
        valid_length = int(encoded["attention_mask"][index].sum().item())
        stock_item = stock[index, :valid_length]
        hook_item = zero_output[index, :valid_length]
        report["zero_weight_equivalence"].append(
            {
                "sentence": sentence,
                "max_abs_diff": float((stock_item - hook_item).abs().max().item()),
                "cosine": float(F.cosine_similarity(stock_pool[index:index + 1], zero_output[index:index + 1, 0, :]).item()),
            }
        )
    report["zero_weight_max_abs_diff"] = max(item["max_abs_diff"] for item in report["zero_weight_equivalence"])
    report["zero_weight_mean_cosine"] = sum(item["cosine"] for item in report["zero_weight_equivalence"]) / len(SENTENCES)

    for layer in model.encoder.layer:
        layer.attention.self.sgam_addresses = addresses
    with torch.inference_mode():
        before = model(**window_inputs, output_attentions=True)
    before_mass = within_section_mass(before.attentions, section_labels, section_mask)
    for layer in model.encoder.layer:
        with torch.no_grad():
            layer.attention.self.sgam.depth_weight.zero_()
            layer.attention.self.sgam.parent_weight.fill_(-1.0)
            layer.attention.self.sgam.sibling_weight.zero_()
    with torch.inference_mode():
        after = model(**window_inputs, output_attentions=True)
    after_mass = within_section_mass(after.attentions, section_labels, section_mask)
    cosine_after = F.cosine_similarity(before.last_hidden_state[:, 0, :], after.last_hidden_state[:, 0, :], dim=-1)
    window_rows = []
    for index in range(len(WINDOW_SECTIONS)):
        window_rows.append(
            {
                "window": index + 1,
                "tokens": int(section_mask[index].sum().item()),
                "within_section_before": before_mass[index],
                "within_section_after": after_mass[index],
                "gain": after_mass[index] - before_mass[index],
                "cls_embedding_cosine": float(cosine_after[index].item()),
            }
        )
    report["attention_shift"] = {
        "metric": "average fraction of attention probability on same-section tokens, averaged over valid section queries, heads, and layers",
        "nonzero_weights": {"depth": 0.0, "parent": -1.0, "sibling": 0.0},
        "pass_threshold": {"minimum_gain": 0.10, "windows_meeting_gain": 4, "embedding_cosine": 0.90},
        "windows": window_rows,
        "windows_meeting_gain": sum(item["gain"] >= 0.10 for item in window_rows),
        "windows_meeting_both": sum(item["gain"] >= 0.10 and item["cls_embedding_cosine"] >= 0.90 for item in window_rows),
    }

    benchmark_rows = []
    for stock_item in stock_benchmark:
        batch_size = stock_item["batch_size"]
        full_input = make_full_length_input(tokenizer, batch_size, device)
        length = full_input["input_ids"].shape[1]
        sgam_addresses = torch.zeros((batch_size, length, 4), dtype=torch.float32, device=device)
        for token_index in range(length):
            section_id = min(3, token_index // 171 + 1)
            sgam_addresses[:, token_index, :] = torch.tensor(
                [token_index, 1, section_id, 0], dtype=torch.float32, device=device
            )
        for layer in model.encoder.layer:
            layer.attention.self.sgam_addresses = sgam_addresses
            with torch.no_grad():
                layer.attention.self.sgam.depth_weight.zero_()
                layer.attention.self.sgam.parent_weight.zero_()
                layer.attention.self.sgam.sibling_weight.zero_()
        hooked_measurement = measure_forward(model, full_input)
        stock_measurement = stock_item["measurement"]
        benchmark_rows.append(
            {
                "batch_size": batch_size,
                "sequence_length": length,
                "stock": stock_measurement,
                "hooked_zero_sgam": hooked_measurement,
                "latency_overhead_percent": round(
                    100 * (hooked_measurement["median_ms"] / stock_measurement["median_ms"] - 1), 2
                ),
                "peak_memory_delta_mb": round(
                    hooked_measurement["peak_allocated_mb"] - stock_measurement["peak_allocated_mb"], 2
                ),
            }
        )
    report["overhead"] = benchmark_rows

    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    report["gradient_checks"] = [
        check_gradients(shared=True, model_id=model_id, device=device),
        check_gradients(shared=False, model_id=model_id, device=device),
    ]
    report["gate_summary"] = {
        "hook_reached_all_layers": report["hook_call_count"] == report["layer_count"],
        "zero_weight_equivalence_pass": report["zero_weight_max_abs_diff"] <= 1e-5
        and report["zero_weight_mean_cosine"] >= 0.99999,
        "attention_shift_pass": report["attention_shift"]["windows_meeting_both"] >= 4,
        "gradient_parameter_counts_pass": all(
            item["unique_parameter_count"] == item["expected_parameter_count"]
            for item in report["gradient_checks"]
        ),
        "gradient_nonzero_pass": all(
            item["nonzero_gradient_count"] == item["checked_gradient_count"]
            for item in report["gradient_checks"]
        ),
    }

    output_dir = ROOT.parent / "Dcoumentation" / "reports"
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "HiSE_System_Validation_Report.json"
    markdown_path = output_dir / "HiSE_System_Validation_Report.md"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    markdown = [
        "# HiSE System Validation Report",
        "",
        f"- Model: `{report['model_id']}` ({report['model_class']})",
        f"- Device: {report['device']} (CUDA {report['cuda_version']})",
        f"- Transformers: {report['transformers_version']}; PyTorch: {report['torch_version']}",
        f"- Default attention: {report['default_attention_implementation']} / `{report['default_attention_class']}`",
        f"- Layers: {report['layer_count']}; hook calls: {report['hook_call_count']}",
        "",
        "## Project checks",
        "",
        f"- Unit tests: {report['project_checks']['unit_tests']}",
        f"- Notebooks: {report['project_checks']['notebooks']}",
        "",
        "## Zero-weight equivalence (20 fixed sentences; padded batch)",
        "",
        f"- Batch shape: `{report['padded_batch_shape']}`",
        f"- Maximum absolute hidden-state difference: `{report['zero_weight_max_abs_diff']:.9g}`",
        f"- Mean CLS cosine: `{report['zero_weight_mean_cosine']:.9f}`",
        "",
        "| # | Tokens | Max abs diff | CLS cosine | Sentence |",
        "|---:|---:|---:|---:|---|",
    ]
    for index, item in enumerate(report["zero_weight_equivalence"], start=1):
        token_count = report["padded_sentence_token_lengths"][index - 1]
        markdown.append(
            f"| {index} | {token_count} | {item['max_abs_diff']:.8g} | {item['cosine']:.9f} | {item['sentence']} |"
        )
    markdown.extend(
        [
            "",
            "## Attention-shift gate",
            "",
            f"- Metric: {report['attention_shift']['metric']}",
            "- Fixed nonzero weights: depth=0, parent=-1.0, sibling=0",
            "- Pass threshold set before measurement: gain >= 0.10 and CLS cosine >= 0.90 on >=4/5 windows",
            "",
            "| Window | Tokens | Within-section before | After | Gain | CLS cosine |",
            "|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for item in report["attention_shift"]["windows"]:
        markdown.append(
            f"| {item['window']} | {item['tokens']} | {item['within_section_before']:.6f} | "
            f"{item['within_section_after']:.6f} | {item['gain']:.6f} | {item['cls_embedding_cosine']:.6f} |"
        )
    markdown.extend(
        [
            "",
            "## Gradient checks",
            "",
            "| Mode | Unique params | Expected | Nonzero gradients | Checked | Loss |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for item in report["gradient_checks"]:
        markdown.append(
            f"| {item['mode']} | {item['unique_parameter_count']} | {item['expected_parameter_count']} | "
            f"{item['nonzero_gradient_count']} | {item['checked_gradient_count']} | {item['loss']:.8g} |"
        )
    markdown.extend(
        [
            "",
            "Parameter-level values and gradients:",
            "",
            "| Mode | Layer | Parameter | Value | Gradient |",
            "|---|---:|---|---:|---:|",
        ]
    )
    for item in report["gradient_checks"]:
        for parameter in item["parameters"]:
            markdown.append(
                f"| {item['mode']} | {parameter['layer']} | {parameter['parameter']} | "
                f"{parameter['value']:.8g} | {parameter['gradient']:.8g} |"
            )
    markdown.extend(
        [
            "",
            "## 512-token GPU overhead (median of 3 timed forwards after warmup)",
            "",
            "Stock uses the model's default SDPA path; hooked uses eager attention because the SGAM score bias is added before softmax. Timings include that required path change. Inputs are full-length synthetic token-id sequences; this is inference-forward overhead, not end-to-end tokenization or embedding-pooling latency.",
            "",
            "| Batch | Stock ms | Hooked ms | Latency overhead | Stock peak alloc MB | Hooked peak alloc MB | Peak delta MB |",
            "|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for item in report["overhead"]:
        markdown.append(
            f"| {item['batch_size']} | {item['stock']['median_ms']:.3f} | {item['hooked_zero_sgam']['median_ms']:.3f} | "
            f"{item['latency_overhead_percent']:.2f}% | {item['stock']['peak_allocated_mb']:.2f} | "
            f"{item['hooked_zero_sgam']['peak_allocated_mb']:.2f} | {item['peak_memory_delta_mb']:.2f} |"
        )
    markdown.extend(["", "## Gate summary", ""])
    markdown.extend(f"- `{name}`: **{'PASS' if passed else 'FAIL'}**" for name, passed in report["gate_summary"].items())
    markdown.extend(
        [
            "",
            "## Scope and limitations",
            "",
            "The attention-shift windows are hand-authored synthetic examples and use fixed, manually selected SGAM weights; they are not evidence of retrieval improvement or trained behavior. The 512-token timing uses repeated valid token IDs and measures model forward latency and allocator peaks only. Retrieval metrics, training convergence, energy use, and throughput under a production corpus were not evaluated.",
            "",
            f"JSON detail: `{json_path.name}`",
            "",
        ]
    )
    markdown_path.write_text("\n".join(markdown), encoding="utf-8")
    print(markdown_path)
    print("\n".join(markdown))
    print(f"\nDetailed JSON: {json_path}")


if __name__ == "__main__":
    main()
