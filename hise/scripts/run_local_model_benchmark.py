"""Run a batched local GPU embedding benchmark and write JSONL evidence."""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / '_compat'))
os.environ.setdefault('USE_TF', '0')

import torch
from transformers import AutoModel, AutoTokenizer


ROOT = Path(__file__).parents[1]
DATA = ROOT / 'data' / 'benchmark' / 'structeval_v1.jsonl'
OUTPUT = ROOT / 'outputs' / 'local_model_benchmark.jsonl'
MODEL_PATH = os.getenv(
    'HISE_LOCAL_MODEL',
    str(Path.home() / '.cache' / 'huggingface' / 'hub' / 'models--sentence-transformers--all-MiniLM-L6-v2' / 'snapshots' / 'c9745ed1d9f207416be6d2e6f8de32d1f16199bf'),
)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = int(os.getenv('HISE_BATCH_SIZE', '32'))


def mean_pool(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weights = mask.unsqueeze(-1).expand(hidden.shape).float()
    return (hidden * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1e-9)


def cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    return torch.nn.functional.cosine_similarity(left.unsqueeze(0), right.unsqueeze(0)).item()


def embed(texts: list[str], tokenizer: AutoTokenizer, model: AutoModel) -> torch.Tensor:
    encoded = tokenizer(texts, padding=True, truncation=True, max_length=512, return_tensors='pt').to(DEVICE)
    with torch.inference_mode():
        output = model(**encoded)
    vectors = mean_pool(output.last_hidden_state, encoded['attention_mask'])
    return torch.nn.functional.normalize(vectors, p=2, dim=1).cpu()


def main() -> None:
    rows = [json.loads(line) for line in DATA.read_text(encoding='utf-8').splitlines()]
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
    model = AutoModel.from_pretrained(MODEL_PATH, local_files_only=True).to(DEVICE).eval()
    texts = [row['query'] for row in rows]
    texts.extend(candidate['text'] for row in rows for candidate in row['candidates'])
    started = time.perf_counter()
    vectors: list[torch.Tensor] = []
    for offset in range(0, len(texts), BATCH_SIZE):
        vectors.extend(embed(texts[offset:offset + BATCH_SIZE], tokenizer, model))
    embedding_seconds = time.perf_counter() - started
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        'event': 'benchmark_summary',
        'model': os.getenv('HISE_LOCAL_MODEL_NAME', 'sentence-transformers/all-MiniLM-L6-v2'),
        'model_path': MODEL_PATH,
        'device': DEVICE,
        'gpu': torch.cuda.get_device_name(0) if DEVICE == 'cuda' else 'cpu',
        'cases': len(rows),
        'batch_size': BATCH_SIZE,
        'embedding_seconds': round(embedding_seconds, 4),
    }
    false_positives = 0
    format_counts: dict[str, list[int]] = {}
    with OUTPUT.open('w', encoding='utf-8') as stream:
        stream.write(json.dumps(summary) + '\n')
        cursor = 0
        for row in rows:
            query_vector = vectors[cursor]
            cursor += 1
            scored = []
            for candidate in row['candidates']:
                score = cosine(query_vector, vectors[cursor])
                cursor += 1
                scored.append({'id': candidate['id'], 'path': candidate['path'], 'cosine': round(score, 8)})
            ranked = sorted(scored, key=lambda item: (-item['cosine'], item['id']))
            is_false_positive = ranked[0]['id'] != row['relevant_id']
            false_positives += is_false_positive
            counts = format_counts.setdefault(row['format'], [0, 0])
            counts[0] += 1
            counts[1] += int(is_false_positive)
            stream.write(json.dumps({**row, 'scores': scored, 'ranked_ids': [item['id'] for item in ranked], 'top_is_false_positive': is_false_positive}) + '\n')
        for format_name, (count, format_false_positives) in sorted(format_counts.items()):
            stream.write(json.dumps({'event': 'format_summary', 'format': format_name, 'cases': count, 'false_positives': format_false_positives, 'false_positive_rate': round(format_false_positives / count, 4)}) + '\n')
    print(json.dumps({**summary, 'false_positives': false_positives, 'false_positive_rate': round(false_positives / len(rows), 4), 'format_summary': format_counts, 'output': str(OUTPUT)}, indent=2))


if __name__ == '__main__':
    main()
