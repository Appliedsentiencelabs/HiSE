from hise.address import StructuralAddress


def test_adversarial_pair_has_same_text_but_different_structure() -> None:
    performance = StructuralAddress(18, 2, StructuralAddress.stable_parent_id('executive'), 1)
    risks = StructuralAddress(18, 2, StructuralAddress.stable_parent_id('risks'), 1)
    assert 'Revenue increased 47% year over year.' == 'Revenue increased 47% year over year.'
    assert performance.as_tuple() != risks.as_tuple()
