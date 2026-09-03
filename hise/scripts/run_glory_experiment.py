"""Run and log the first real Glory Nomic structural-retrieval experiment."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


ROOT = Path(__file__).parents[1]
EXAMPLE = ROOT / 'data' / 'examples' / 'adversarial_pairs.json'
OUTPUT = ROOT / 'outputs' / 'glory_nomic_experiment.jsonl'
OLLAMA_URL = os.getenv('HISE_OLLAMA_URL', 'http://100.71.221.47:11434')
EMBED_MODEL = os.getenv('HISE_EMBEDDING_MODEL', 'nomic-embed-text:latest')
LLM_URL = os.getenv('HISE_LLM_URL', 'http://100.71.221.47:8000/v1/models')
LLM_MODEL = os.getenv('HISE_LLM_MODEL', 'asl-chat')


def log(event: dict) -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    event = {'timestamp_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), **event}
    with OUTPUT.open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(event) + '\n')


def embed(inputs: list[str]) -> tuple[list[list[float]], float]:
    started = time.perf_counter()
    payload = json.dumps({'model': EMBED_MODEL, 'input': inputs}).encode('utf-8')
    request = Request(f'{OLLAMA_URL}/api/embed', data=payload, headers={'Content-Type': 'application/json'})
    try:
        with urlopen(request, timeout=60) as response:
            result = json.load(response)
    except (HTTPError, URLError) as error:
        detail = error.read().decode() if isinstance(error, HTTPError) else str(error)
        log({'event': 'embedding_error', 'model': EMBED_MODEL, 'error': detail})
        raise RuntimeError(f'Nomic embedding request failed: {detail}') from error
    latency_ms = (time.perf_counter() - started) * 1000
    vectors = result.get('embeddings')
    if not vectors or len(vectors) != len(inputs):
        raise RuntimeError(f'Unexpected Ollama embedding response: {result}')
    log({'event': 'embedding_request', 'model': EMBED_MODEL, 'count': len(inputs), 'latency_ms': round(latency_ms, 2)})
    return vectors, latency_ms


def cosine(left: list[float], right: list[float]) -> float:
    numerator = sum(a * b for a, b in zip(left, right))
    left_norm = sum(a * a for a in left) ** 0.5
    right_norm = sum(b * b for b in right) ** 0.5
    return numerator / (left_norm * right_norm)


def run_condition(name: str, query: str, candidates: list[tuple[str, str]], relevant_id: str) -> dict:
    vectors, latency_ms = embed([query, *(content for _, content in candidates)])
    scores = [(candidate_id, cosine(vectors[0], vector)) for (candidate_id, _), vector in zip(candidates, vectors[1:])]
    ranked = sorted(scores, key=lambda item: (-item[1], item[0]))
    result = {
        'event': 'condition_result',
        'condition': name,
        'query': query,
        'model': EMBED_MODEL,
        'scores': [{'candidate_id': item[0], 'cosine': round(item[1], 8)} for item in scores],
        'ranked_ids': [item[0] for item in ranked],
        'relevant_id': relevant_id,
        'top_is_false_positive': ranked[0][0] != relevant_id,
        'embedding_latency_ms': round(latency_ms, 2),
    }
    log(result)
    return result


def check_llm_service() -> None:
    started = time.perf_counter()
    try:
        with urlopen(LLM_URL, timeout=15) as response:
            body = json.load(response)
        available = [item.get('id') for item in body.get('data', [])]
        log({'event': 'llm_service_check', 'url': LLM_URL, 'model': LLM_MODEL, 'available_models': available, 'latency_ms': round((time.perf_counter() - started) * 1000, 2)})
    except (HTTPError, URLError) as error:
        detail = error.read().decode() if isinstance(error, HTTPError) else str(error)
        log({'event': 'llm_service_error', 'url': LLM_URL, 'model': LLM_MODEL, 'error': detail})


def check_llm_inference() -> None:
    endpoint = LLM_URL.rsplit('/models', 1)[0] + '/chat/completions'
    payload = {
        'model': LLM_MODEL,
        'messages': [{'role': 'user', 'content': 'Reply with exactly SERVER_OK.'}],
        'max_tokens': 8,
        'temperature': 0.0,
        'chat_template_kwargs': {'enable_thinking': False},
    }
    started = time.perf_counter()
    request = Request(endpoint, data=json.dumps(payload).encode('utf-8'), headers={'Content-Type': 'application/json'})
    try:
        with urlopen(request, timeout=60) as response:
            result = json.load(response)
        content = result['choices'][0]['message']['content']
        log({'event': 'llm_inference_check', 'url': endpoint, 'model': LLM_MODEL, 'response': content, 'latency_ms': round((time.perf_counter() - started) * 1000, 2)})
    except (HTTPError, URLError, KeyError, IndexError) as error:
        detail = error.read().decode() if isinstance(error, HTTPError) else str(error)
        log({'event': 'llm_inference_error', 'url': endpoint, 'model': LLM_MODEL, 'error': detail})


def main() -> None:
    case = json.loads(EXAMPLE.read_text(encoding='utf-8'))
    query = 'What are the financial risks?'
    candidates = [('performance', case['positive']['text']), ('risk', case['negative']['text'])]
    structural_candidates = [
        ('performance', f'Annual Report / Executive Summary / Key Highlights\n{case["positive"]["text"]}'),
        ('risk', f'Annual Report / Risk Factors / Revenue Concentration Risk\n{case["negative"]["text"]}'),
    ]
    log({'event': 'experiment_start', 'embedding_url': f'{OLLAMA_URL}/api/embed', 'embedding_model': EMBED_MODEL, 'llm_model': LLM_MODEL})
    baseline = run_condition('content_only', query, candidates, 'risk')
    structural = run_condition('path_prefixed_control', query, structural_candidates, 'risk')
    check_llm_service()
    check_llm_inference()
    print(json.dumps({'baseline': baseline, 'path_prefixed_control': structural, 'log_file': str(OUTPUT)}, indent=2))


if __name__ == '__main__':
    main()
