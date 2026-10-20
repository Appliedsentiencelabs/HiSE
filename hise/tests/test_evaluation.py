from hise.address import StructuralAddress
from hise.evaluation import RetrievalCase, evaluate_sgam, lexical_rank, top_one_is_false_positive


def test_identical_surface_text_can_create_baseline_tie() -> None:
    case = RetrievalCase(
        query='What is the revenue situation?',
        relevant_id='risk',
        candidates=(
            ('performance', 'Revenue increased 47% year over year.'),
            ('risk', 'Revenue increased 47% year over year.'),
        ),
    )
    ranked = lexical_rank(case.query, case.candidates)
    assert ranked[0][1] == ranked[1][1]
    assert top_one_is_false_positive(case)


def test_sgam_evaluation_report_tracks_core_invariants() -> None:
    addresses = [
        StructuralAddress(0, 1, 10, 0),
        StructuralAddress(1, 1, 10, 1),
        StructuralAddress(2, 2, 20, 0),
        StructuralAddress(3, 2, 21, 1),
    ]
    report = evaluate_sgam(addresses)
    assert report["zero_bias"] is True
    assert report["same_structure_zero"] is True
    assert report["symmetric"] is True
    assert report["bounded"] is True
    assert report["gradient_flow"] is True
    assert report["all_pass"] is True
