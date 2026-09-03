from hise.address import StructuralAddress


def test_address_has_unified_four_coordinate_schema() -> None:
    address = StructuralAddress(sequence_position=7, depth=3, parent_id=11, sibling_index=2)
    assert address.as_tuple() == (7, 3, 11, 2)
    assert address.as_vector() == (7.0, 3.0, 11.0, 2.0)


def test_parent_id_is_stable() -> None:
    assert StructuralAddress.stable_parent_id('root.risks') == StructuralAddress.stable_parent_id('root.risks')
    assert StructuralAddress.stable_parent_id('root.risks') != StructuralAddress.stable_parent_id('root.performance')
