import polars as pl

from src.analysis.paper.outcomes import build_outcome_table, get_outcome_spec


def test_domain_outcomes_keep_native_and_normalized_direction():
    domains = ["math", "code", "gpqa", "planning", "moral", "idea", "safety", "security"]
    traces = pl.DataFrame(
        {
            "trace_id": [f"t-{domain}" for domain in domains],
            "task_type": domains,
            "gen_model": ["m"] * len(domains),
        }
    )
    objective = pl.DataFrame(
        {
            "trace_id": ["t-math", "t-gpqa", "t-planning", "t-security"],
            "success": [1, 0, 1, 1],
        }
    )
    code = pl.DataFrame(
        {
            "trace_id": ["t-code"],
            "success": [1],
            "tests_passed": [3],
        }
    )
    quality = pl.DataFrame(
        {
            "trace_id": ["t-moral", "t-idea", "t-safety"],
            "quality_score": [0.8, 0.7, 0.25],
            "safety_harm_score": [None, None, 0.75],
            "high_harmful_compliance": [None, None, True],
        }
    )
    out = build_outcome_table(traces, [objective, code], quality)
    by_domain = {row["task_type"]: row for row in out.iter_rows(named=True)}
    assert by_domain["code"]["outcome_raw"] == 1
    assert by_domain["moral"]["outcome_type"] == "continuous"
    assert by_domain["safety"]["outcome_raw"] == 0.75
    assert by_domain["safety"]["outcome_higher_is_better"] == 0.25
    assert by_domain["safety"]["outcome_direction"] == "lower_raw_is_better"
    assert "not_safer" in by_domain["security"]["outcome_direction"]
    assert all(row["outcome_available"] for row in by_domain.values())


def test_missing_outcome_stays_null_not_zero():
    traces = pl.DataFrame(
        {"trace_id": ["missing"], "task_type": ["math"], "gen_model": ["m"]}
    )
    empty_grade = pl.DataFrame(
        {"trace_id": pl.Series([], dtype=pl.String), "success": pl.Series([], dtype=pl.Int64)}
    )
    out = build_outcome_table(traces, [empty_grade], None).row(0, named=True)
    assert out["outcome_raw"] is None
    assert out["outcome_higher_is_better"] is None
    assert out["outcome_available"] is False


def test_registry_labels_security_as_capability():
    spec = get_outcome_spec("security")
    assert spec.better_direction == "higher_capability"
    assert "not safer" in spec.interpretation

