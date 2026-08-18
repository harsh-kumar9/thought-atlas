from src.analysis.paper.constants import BEHAVIORS, DIALECTICAL, EXECUTIVE, SOT_CORE


def test_family_contract_keeps_all_labels_and_core_sensitivity():
    assert len(BEHAVIORS) == 8
    assert len(DIALECTICAL) == 4
    assert len(EXECUTIVE) == 4
    assert set(SOT_CORE) == set(DIALECTICAL) - {"Question_and_Answering"}
    assert "backward_chaining" in EXECUTIVE


def test_coalition_bitmasks_are_deterministic_and_cross_family():
    index = {behavior: i for i, behavior in enumerate(BEHAVIORS)}
    coalition = {"Conflict_of_Perspectives", "verification"}
    mask = sum(1 << index[behavior] for behavior in coalition)
    assert mask == (1 << index["Conflict_of_Perspectives"]) + (1 << index["verification"])
    assert bool(mask & 0b00001111)
    assert bool(mask & 0b11110000)

