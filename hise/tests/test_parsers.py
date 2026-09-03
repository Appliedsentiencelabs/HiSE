from hise.parsers import parse_html, parse_json, parse_markdown, parse_table


def test_markdown_preserves_heading_context() -> None:
    items = parse_markdown('# Report\n## Risks\nRevenue increased 47%')
    assert len(items) == 3
    assert items[1][1].depth == 2
    assert items[2][1].depth == 2
    assert items[1][1].parent_id != items[2][1].parent_id


def test_table_maps_row_and_column() -> None:
    items = parse_table([['metric', 'value'], ['revenue', '47%']])
    assert len(items) == 4
    assert items[0][1].sibling_index == 0
    assert items[3][1].sibling_index == 1001


def test_json_maps_nested_key_path_depth() -> None:
    items = parse_json({'risk': {'revenue': 'concentration'}})
    assert items[0][0] == 'concentration'
    assert items[0][1].depth == 2


def test_html_maps_dom_depth() -> None:
    items = parse_html('<section><h2>Risks</h2><p>Revenue increased</p></section>')
    assert [item[0] for item in items] == ['Risks', 'Revenue increased']
    assert items[1][1].depth == items[0][1].depth
    assert items[1][1].parent_id != items[0][1].parent_id
