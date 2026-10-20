"""Command-line entry point for the HiSE prototype."""

from __future__ import annotations

import argparse
from pathlib import Path

from .address import StructuralAddress
from .evaluation import RetrievalCase, evaluate_sgam, lexical_rank, top_one_is_false_positive


def _demo() -> None:
    case = RetrievalCase(
        query="What is the revenue situation?",
        relevant_id="risk",
        candidates=(
            ("performance", "Revenue increased 47% year over year."),
            ("risk", "Revenue increased 47% year over year."),
        ),
    )
    ranked = lexical_rank(case.query, case.candidates)
    print("top results:", ranked)
    print("top_one_is_false_positive:", top_one_is_false_positive(case))


def _eval() -> None:
    case = RetrievalCase(
        query="What are the financial risks?",
        relevant_id="risk",
        candidates=(
            ("performance", "Revenue increased 47% year over year."),
            ("risk", "The company faces customer concentration risk in the top 10 accounts."),
        ),
    )
    ranked = lexical_rank(case.query, case.candidates)
    false_positive = top_one_is_false_positive(case)
    addresses = [
        StructuralAddress(0, 1, 10, 0),
        StructuralAddress(1, 1, 10, 1),
        StructuralAddress(2, 2, 20, 0),
        StructuralAddress(3, 2, 21, 1),
    ]
    report = evaluate_sgam(addresses)
    print({
        "query": case.query,
        "relevant_id": case.relevant_id,
        "ranked": ranked,
        "false_positive": false_positive,
        "sgam": report,
    })


def main() -> None:
    parser = argparse.ArgumentParser(description="HiSE prototype CLI")
    parser.add_argument("command", choices=["demo", "eval"], help="Run a demo or evaluation case")
    parser.add_argument("--verbose", action="store_true", help="Print more detail")
    args = parser.parse_args()

    if args.command == "demo":
        _demo()
    else:
        _eval()


if __name__ == "__main__":
    main()
