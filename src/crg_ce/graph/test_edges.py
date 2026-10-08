from crg_ce.graph.edges import get_edge_label_descriptions


def test_get_edge_label_descriptions_preserves_all_labels_and_filters_subsets() -> None:
    # This verifies the configurable prompt-label interface preserves the legacy four-label mapping exactly while
    # returning only requested labels in caller order.
    assert get_edge_label_descriptions(["proves", "supports", "refutes", "undermines"]) == {
        "proves": "establishes the target claim as true beyond reasonable doubt",
        "supports": "increases confidence in the target claim",
        "refutes": "establishes the target claim as false beyond reasonable doubt",
        "undermines": "decreases confidence in the target claim",
    }
    assert get_edge_label_descriptions(["supports", "undermines"]) == {
        "supports": "increases confidence in the target claim",
        "undermines": "decreases confidence in the target claim",
    }
