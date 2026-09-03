"""Generate charts and a Markdown evidence report from local benchmark JSONL."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


ROOT = Path(__file__).parents[1]
INPUT = ROOT / 'outputs' / 'local_model_benchmark.jsonl'
FIGURES = ROOT / 'outputs' / 'figures'
REPORT = ROOT / 'outputs' / 'HiSE_Benchmark_Report.md'

sns.set_theme(style='whitegrid', context='talk')
PALETTE = {'positive': '#1b9e77', 'negative': '#d95f02'}


def load_data() -> tuple[dict, pd.DataFrame]:
    records = [json.loads(line) for line in INPUT.read_text(encoding='utf-8').splitlines()]
    summary = next(record for record in records if record.get('event') == 'benchmark_summary')
    cases = [record for record in records if 'case_id' in record]
    score_rows = []
    for case in cases:
        scores = {item['id']: item['cosine'] for item in case['scores']}
        score_rows.append({
            'case_id': case['case_id'],
            'format': case['format'],
            'positive_cosine': scores['positive'],
            'negative_cosine': scores['negative'],
            'margin_positive_minus_negative': scores['positive'] - scores['negative'],
            'false_positive': case['top_is_false_positive'],
        })
    return summary, pd.DataFrame(score_rows)


def save_figures(frame: pd.DataFrame) -> dict[str, Path]:
    FIGURES.mkdir(parents=True, exist_ok=True)
    rates = frame.groupby('format', sort=True)['false_positive'].mean().mul(100).reset_index(name='false_positive_rate')
    order = rates.sort_values('false_positive_rate', ascending=False)['format']

    plt.figure(figsize=(11, 6))
    chart = sns.barplot(data=rates, x='format', y='false_positive_rate', order=order, color='#2878b5')
    chart.set(title='Baseline False-Positive Rate by Structured Data Format', xlabel='Data format', ylabel='False positives (%)')
    chart.set_ylim(0, 100)
    for container in chart.containers:
        chart.bar_label(container, fmt='%.1f%%', padding=4)
    plt.tight_layout()
    rate_path = FIGURES / 'false_positive_rate_by_format.png'
    plt.savefig(rate_path, dpi=180)
    plt.close()

    long = frame.melt(id_vars=['format'], value_vars=['positive_cosine', 'negative_cosine'], var_name='candidate', value_name='cosine')
    long['candidate'] = long['candidate'].map({'positive_cosine': 'Relevant', 'negative_cosine': 'Distractor'})
    plt.figure(figsize=(12, 7))
    chart = sns.boxplot(data=long, x='format', y='cosine', hue='candidate', palette={'Relevant': PALETTE['positive'], 'Distractor': PALETTE['negative']})
    chart.set(title='Cosine Similarity: Relevant Chunk vs Distractor', xlabel='Data format', ylabel='Cosine similarity')
    chart.legend(title='Candidate')
    plt.tight_layout()
    score_path = FIGURES / 'cosine_similarity_by_format.png'
    plt.savefig(score_path, dpi=180)
    plt.close()

    plt.figure(figsize=(12, 6))
    chart = sns.boxplot(data=frame, x='format', y='margin_positive_minus_negative', color='#66c2a5')
    chart.axhline(0, color='#444444', linewidth=1.2)
    chart.set(title='Retrieval Score Margin (Relevant - Distractor)', xlabel='Data format', ylabel='Cosine margin')
    plt.tight_layout()
    margin_path = FIGURES / 'score_margin_by_format.png'
    plt.savefig(margin_path, dpi=180)
    plt.close()
    return {'rate': rate_path, 'score': score_path, 'margin': margin_path}


def write_report(summary: dict, frame: pd.DataFrame, figures: dict[str, Path]) -> None:
    rates = frame.groupby('format', sort=True).agg(cases=('false_positive', 'size'), false_positives=('false_positive', 'sum'))
    rates['false_positive_rate'] = rates['false_positives'] / rates['cases']
    mean_scores = frame[['positive_cosine', 'negative_cosine', 'margin_positive_minus_negative']].mean()
    model_name = Path(summary['model']).name
    report = f'''# HiSE Baseline Benchmark Report

## Purpose

This report records the first volume experiment for HiSE. It tests whether a text embedding baseline can reliably distinguish relevant and distractor content when the candidates occupy different structural roles in tabular data, JSON, HTML/XML, Markdown, and complex nested data.

This is a baseline measurement, not a claim that HiSE has already improved retrieval. The HiSE structural encoder must be evaluated against this same dataset and protocol.

## Experimental Setup

| Item | Value |
|---|---|
| Embedding model | `{model_name}` |
| Device | `{summary['device']}` |
| GPU | `{summary.get('gpu', 'not recorded')}` |
| Cases | {summary['cases']:,} |
| Cases per format | 200 |
| Batch size | {summary['batch_size']} |
| Embedding time | {summary['embedding_seconds']:.4f} seconds |
| Protocol | Query embedding compared with relevant and distractor candidates |

The benchmark contains identical query templates repeated across five data formats. Each case has one relevant candidate and one distractor candidate. The current baseline ranks a case incorrectly when the distractor receives the higher cosine score. Ties are resolved deterministically by candidate id.

## Results

- False positives: **{int(frame['false_positive'].sum()):,} / {len(frame):,}**
- Overall false-positive rate: **{frame['false_positive'].mean() * 100:.1f}%**
- Mean relevant-candidate cosine: **{mean_scores['positive_cosine']:.4f}**
- Mean distractor cosine: **{mean_scores['negative_cosine']:.4f}**
- Mean relevant-minus-distractor margin: **{mean_scores['margin_positive_minus_negative']:.4f}**

### By Format

| Format | Cases | False positives | Rate |
|---|---:|---:|---:|
'''
    for format_name, row in rates.sort_values('false_positive_rate', ascending=False).iterrows():
        report += f"| {format_name} | {int(row['cases'])} | {int(row['false_positives'])} | {row['false_positive_rate'] * 100:.1f}% |\n"
    report += f'''
## Visualizations

### False-Positive Rate

![False-positive rate by format](figures/{figures['rate'].name})

### Cosine Similarity

![Cosine similarity by format](figures/{figures['score'].name})

### Score Margin

![Relevant minus distractor score margin](figures/{figures['margin'].name})

## Interpretation

The baseline produces a substantial false-positive rate across the generated volume, with the highest error rates in the formats where structural role is important. A negative score margin means the distractor was more similar to the query than the relevant candidate. This is the measurable failure mode HiSE is designed to address.

The results do not establish that structural encoding improves retrieval yet. The next controlled experiment must use the same cases, queries, and candidate text while adding structural-address features, then compare false-positive rate, structural discrimination score, and retrieval inversion rate.

## Artifacts

- Raw cases: `data/benchmark/structeval_v1.jsonl`
- Raw embeddings and decisions: `outputs/local_model_benchmark.jsonl`
- Generator: `data/generate_benchmark.py`
- Runner: `scripts/run_local_model_benchmark.py`
'''
    REPORT.write_text(report, encoding='utf-8')


def main() -> None:
    summary, frame = load_data()
    figures = save_figures(frame)
    write_report(summary, frame, figures)
    print(f'Generated {len(figures)} figures and report: {REPORT}')


if __name__ == '__main__':
    main()
