import json
from pathlib import Path


def test_adversarial_example_is_reproducible() -> None:
    example = json.loads((Path(__file__).parents[1] / 'data' / 'examples' / 'adversarial_pairs.json').read_text())
    assert example['positive']['text'] == example['negative']['text']
    assert example['positive']['path'] != example['negative']['path']
