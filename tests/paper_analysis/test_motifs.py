from src.analysis.paper.final_analysis import MOTIFS, TRIPLES


def test_all_prespecified_motifs_are_registered():
    assert MOTIFS["perspective_to_verification"] == (
        "Perspective_Shift", "verification")
    assert MOTIFS["conflict_to_reconciliation"] == (
        "Conflict_of_Perspectives", "Reconciliation")
    assert TRIPLES["verification_backtracking_subgoal"] == (
        "verification", "backtracking", "subgoal")
    assert len(MOTIFS) == 7
    assert len(TRIPLES) == 4


def test_motif_names_preserve_raw_column_case():
    raw = {behavior for motif in list(MOTIFS.values()) + list(TRIPLES.values()) for behavior in motif}
    assert "Question_and_Answering" in raw
    assert "Reconciliation" in raw
    assert "verification" in raw

