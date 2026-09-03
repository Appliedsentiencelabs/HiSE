from hise.evaluation import RetrievalCase, lexical_rank, top_one_is_false_positive


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
