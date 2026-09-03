from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).parents[1]
OUTPUT = ROOT / 'data' / 'benchmark' / 'structeval_v1.jsonl'

SCENARIOS = [
    ('tabular', 'What is the revenue risk?', 'Risk Register', 'Revenue concentration increased to 91% of billings.', 'Executive Summary', 'Revenue increased 47% year over year.'),
    ('json', 'What is the production memory limit?', 'spec.containers[0].resources.limits.memory', '512Mi is the hard memory limit for the main application.', 'spec.initContainers[0].resources.requests.memory', '128Mi is the soft memory request for initialization.'),
    ('html', 'What is the contract exception?', 'agreement > exceptions > intellectual-property', 'The vendor retains rights to pre-existing intellectual property.', 'agreement > obligations > intellectual-property', 'The vendor transfers project intellectual property after completion.'),
    ('markdown', 'What is the operational risk?', 'Quarterly Report > Risks > Operational Risk', 'The current process depends on one operator and has no failover.', 'Quarterly Report > Highlights > Operations', 'Operations achieved a 98% completion rate this quarter.'),
    ('complex', 'What constraint applies to the deployment?', 'Platform > Security > Production Constraints', 'Production deployment requires approval from two independent reviewers.', 'Platform > Overview > Deployment', 'The deployment completed successfully in the staging environment.'),
]


def main() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    with OUTPUT.open('w', encoding='utf-8') as stream:
        for repeat in range(200):
            for kind, query, positive_path, positive_text, negative_path, negative_text in SCENARIOS:
                # The same query is paired with semantically related but structurally distinct chunks.
                row = {
                    'case_id': f'{kind}-{repeat:04d}',
                    'format': kind,
                    'query': query,
                    'relevant_id': 'positive',
                    'candidates': [
                        {'id': 'positive', 'path': positive_path, 'text': positive_text},
                        {'id': 'negative', 'path': negative_path, 'text': negative_text},
                    ],
                }
                stream.write(json.dumps(row) + '\n')
                total += 1
    print(f'Wrote {total} cases to {OUTPUT}')


if __name__ == '__main__':
    main()
