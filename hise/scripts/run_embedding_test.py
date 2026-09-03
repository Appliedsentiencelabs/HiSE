r"""Run a baseline false-positive test against an OpenAI-compatible embedding service.

Example:
  $env:HISE_EMBEDDING_URL = 'http://100.71.221.47:8000/v1/embeddings'
  $env:HISE_EMBEDDING_MODEL = 'nomic-embed-text'
  python .\hise\scripts\run_embedding_test.py
"""

import json
import os
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


ROOT = Path(__file__).parents[1]
EXAMPLE = ROOT / 'data' / 'examples' / 'adversarial_pairs.json'
URL = os.getenv('HISE_EMBEDDING_URL', 'http://100.71.221.47:8000/v1/embeddings')
MODEL = os.getenv('HISE_EMBEDDING_MODEL', 'nomic-embed-text')


def request_embeddings(inputs: list[str]) -> list[list[float]]:
    payload = json.dumps({'model': MODEL, 'input': inputs}).encode('utf-8')
    request = Request(URL, data=payload, headers={'Content-Type': 'application/json'})
    try:
        with urlopen(request, timeout=30) as response:
            result = json.load(response)
    except (HTTPError, URLError) as error:
        detail = error.read().decode() if isinstance(error, HTTPError) else str(error)
        raise RuntimeError(f'embedding request failed at {URL}: {detail}') from error
    return [item['embedding'] for item in sorted(result['data'], key=lambda item: item['index'])]


def cosine(left: list[float], right: list[float]) -> float:
    numerator = sum(a * b for a, b in zip(left, right))
    left_norm = sum(a * a for a in left) ** 0.5
    right_norm = sum(b * b for b in right) ** 0.5
    return numerator / (left_norm * right_norm)


def main() -> None:
    case = json.loads(EXAMPLE.read_text(encoding='utf-8'))
    inputs = [case['query'], case['positive']['text'], case['negative']['text']]
    query, positive, negative = request_embeddings(inputs)
    positive_score = cosine(query, positive)
    negative_score = cosine(query, negative)
    print(f'endpoint: {URL}')
    print(f'model: {MODEL}')
    print(f'positive structural path: {" / ".join(case["positive"]["path"])}')
    print(f'negative structural path: {" / ".join(case["negative"]["path"])}')
    print(f'positive cosine: {positive_score:.6f}')
    print(f'negative cosine: {negative_score:.6f}')
    print(f'text-only false positive: {negative_score >= positive_score}')


if __name__ == '__main__':
    main()
