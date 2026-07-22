"""Audit task, trace, grade, and judge artifacts before analysis.

The command is read-only.  ``--strict-v2`` exits non-zero for legacy contracts,
coverage gaps, duplicate keys, invalid score ranges, or malformed MCQ prompts.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.utils.io import resolve_trace_paths
from src.utils.parse import parse_generation_detailed


def _load(paths) -> pl.DataFrame:
    paths = list(paths)
    return pl.concat([pl.read_parquet(p) for p in paths], how="diagonal_relaxed") if paths else pl.DataFrame()


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _artifact_manifests(paths: list[str]) -> tuple[bool, list[str]]:
    issues = []
    for raw in paths:
        path = Path(raw); mp = path.with_suffix(".manifest.json")
        try:
            manifest = json.loads(mp.read_text())
            if (manifest.get("schema_version") != 2 or
                    manifest.get("rows") != pl.scan_parquet(path).select(pl.len()).collect().item() or
                    manifest.get("sha256") != _sha256(path)):
                raise ValueError("manifest mismatch")
        except Exception:
            issues.append(f"missing or invalid manifest for {path.name}")
    return not issues, issues


def audit_tasks(tasks_dir: Path) -> tuple[dict, list[str]]:
    report, issues = {}, []
    for path in sorted(tasks_dir.glob("*.parquet")):
        df = pl.read_parquet(path); task = path.stem
        info = {"rows": df.height, "duplicate_instance_ids": 0}
        if "instance_id" in df.columns:
            info["duplicate_instance_ids"] = df.height - df["instance_id"].n_unique()
            if info["duplicate_instance_ids"]:
                issues.append(f"{task}: duplicate instance IDs")
        if task in {"planning", "gpqa"}:
            malformed = 0
            for row in df.iter_rows(named=True):
                md = json.loads(row.get("metadata") or "{}")
                n = int(md.get("n_options", 4))
                labels = re.findall(r"(?m)^\s*([A-E])\.\s+.+$", row.get("prompt") or "")
                expected = [chr(65 + i) for i in range(n)]
                duplicate_inline = (task == "planning" and len(re.findall(
                    r"(?:^|\s)([A-E])\.\s+", row.get("prompt") or "")) != n)
                malformed += int(labels[-n:] != expected or duplicate_inline)
            info["malformed_or_duplicate_option_blocks"] = malformed
            if malformed:
                issues.append(f"{task}: {malformed} prompts do not have exactly one option block")
        if task == "gpqa" and "metadata" in df.columns:
            diamond = df.filter(pl.col("metadata").str.contains('"gpqa_subset": "diamond"')).height
            info["diamond_rows"] = diamond
            if df.height >= 198 and diamond < 198:
                issues.append(f"gpqa: only {diamond}/198 Diamond rows retained")
        if task == "moral" and "metadata" in df.columns:
            negative = 0
            for raw in df["metadata"].to_list():
                negative += int(bool(re.search(r"['\"]weight['\"]\s*:\s*-", raw or "")))
            info["rubrics_with_negative_weights"] = negative
        report[task] = info
    if not report:
        issues.append(f"no task parquets in {tasks_dir}")
    manifest_path = tasks_dir / "manifest.json"
    if not manifest_path.exists():
        issues.append("task manifest missing (legacy task build)")
    else:
        try:
            manifest = json.loads(manifest_path.read_text())
            if manifest.get("task_build_version") != "tasks-v2":
                issues.append("task manifest is not tasks-v2")
            for task, info in report.items():
                recorded = manifest.get("tasks", {}).get(task, {})
                path = tasks_dir / f"{task}.parquet"
                if recorded.get("rows") != info["rows"] or recorded.get("sha256") != _sha256(path):
                    issues.append(f"{task}: task manifest count/hash mismatch")
        except Exception as exc:
            issues.append(f"invalid task manifest: {type(exc).__name__}")
    return report, issues


def audit_traces(pattern: str | None) -> tuple[dict, list[str], pl.DataFrame]:
    issues = []
    if not pattern:
        return {"skipped": True}, [], pl.DataFrame()
    paths = resolve_trace_paths(pattern)
    df = _load(paths)
    report = {"files": [str(p) for p in paths], "rows": df.height}
    if df.height == 0:
        return report, [f"no traces matched {pattern}"], df
    key = ["gen_model", "instance_id", "seed"]
    if all(c in df.columns for c in key):
        dup = df.height - df.select(pl.struct(key).n_unique()).item()
        report["duplicate_natural_keys"] = dup
        if dup: issues.append(f"traces: {dup} duplicate natural keys")
    version_ok = "generation_version" in df.columns and set(
        df["generation_version"].drop_nulls().unique().to_list()) == {"generation-v2"}
    report["generation_v2"] = version_ok
    if not version_ok: issues.append("traces: generation-v2 provenance columns missing")
    manifests_ok = True
    for path in paths:
        if re.search(r"\.shard\d+of\d+\.parquet$", path.name):
            manifests_ok = False
            issues.append(f"traces: only shard inputs available for {path.name}; canonical merge required")
            continue
        mp = path.with_suffix(".manifest.json")
        try:
            manifest = json.loads(mp.read_text())
            summary = pl.scan_parquet(path).select(
                pl.len().alias("rows"), pl.col("generation_fingerprint").drop_nulls().unique().alias("fps")
            ).collect()
            fps = summary["fps"][0].to_list()
            if (manifest.get("schema_version") != 2 or manifest.get("rows") != summary["rows"][0] or
                    manifest.get("sha256") != _sha256(path) or len(fps) != 1 or
                    manifest.get("generation_fingerprint") != fps[0]):
                raise ValueError("manifest mismatch")
        except Exception:
            manifests_ok = False
            issues.append(f"traces: missing or invalid manifest for {path.name}")
    report["manifests_valid"] = manifests_ok
    blank, recoverable, multi = 0, 0, 0
    statuses = Counter()
    for row in df.iter_rows(named=True):
        answer = (row.get("answer_text") or "").strip()
        blank += int(not answer)
        kind = ("non_reasoning" if row.get("generation_kind") == "non_reasoning" or
                row.get("thinking_style") == "none" or
                row.get("gen_model") == "anchor" else "reasoning")
        parsed = parse_generation_detailed(row.get("full_text") or "", kind,
                                           finish_reason=row.get("finish_reason"))
        recoverable += int(not answer and bool(parsed["answer_text"]))
        multi += int(parsed["close_tag_count"] > 1)
        statuses[parsed["parse_status"]] += 1
    report.update({"blank_stored_answers": blank, "legacy_blank_answers_recoverable": recoverable,
                   "multiple_close_outputs": multi, "reparse_statuses": dict(statuses)})
    if recoverable: issues.append(f"traces: {recoverable} blank stored answers are parser-recoverable")
    return report, issues, df


def audit_grades(pattern: str | None, traces: pl.DataFrame) -> tuple[dict, list[str]]:
    if not pattern:
        return {}, []
    paths = sorted(glob.glob(pattern)); grades = _load(paths); issues = []
    report = {"files": paths, "rows": grades.height}
    if grades.height == 0:
        return report, [f"no grades matched {pattern}"]
    report["unique_trace_ids"] = grades["trace_id"].n_unique()
    if "grader_version" not in grades.columns:
        issues.append("grades: grader_version missing")
    if {"parsed", "success"}.issubset(grades.columns):
        report["unparsed_rows"] = grades.filter(~pl.col("parsed").fill_null(False)).height
        bad = grades.filter(~pl.col("parsed").fill_null(False) & pl.col("success").is_not_null()).height
        report["unparsed_with_numeric_success"] = bad
        if bad: issues.append(f"grades: {bad} unparsed rows were scored numerically")
    objective_ids = set(traces.filter(pl.col("task_type").is_in(["math", "gpqa", "planning"]))
                        ["trace_id"].to_list()) if traces.height else set()
    covered = set(grades["trace_id"].to_list())
    report["objective_missing"] = len(objective_ids - covered)
    if objective_ids - covered: issues.append(f"grades: {len(objective_ids - covered)} objective traces missing")
    code_ids = set(traces.filter(pl.col("task_type") == "code")["trace_id"].to_list()) if traces.height else set()
    if code_ids:
        code_rows = grades.filter(
            (pl.col("task_type") == "code") &
            pl.col("grader_version").fill_null("").str.starts_with("code-exec-v2")
        ) if "grader_version" in grades.columns else pl.DataFrame()
        code_covered = set(code_rows["trace_id"].to_list()) if code_rows.height else set()
        code_missing = len(code_ids - code_covered)
        report["code_missing"] = code_missing
        report["code_ungradeable"] = (code_rows.filter(~pl.col("gradeable").fill_null(False)).height
                                       if code_rows.height and "gradeable" in code_rows.columns else 0)
        if code_missing: issues.append(f"grades: {code_missing} code traces missing execution grades")
    return report, issues


def audit_quality(pattern: str | None, traces: pl.DataFrame) -> tuple[dict, list[str]]:
    if not pattern:
        return {}, []
    paths = sorted(glob.glob(pattern)); q = _load(paths); issues = []
    report = {"files": paths, "rows": q.height}
    if q.height == 0:
        return report, [f"no quality rows matched {pattern}"]
    report["manifests_valid"], manifest_issues = _artifact_manifests(paths)
    issues.extend(f"quality: {x}" for x in manifest_issues)
    invalid = q.filter(pl.col("quality_score").is_not_null() &
                       ~pl.col("quality_score").is_between(0, 1, closed="both")).height
    report["scores_outside_0_1"] = invalid
    if invalid: issues.append(f"quality: {invalid} scores outside [0,1]")
    report["rows_without_quality_score"] = q.filter(pl.col("quality_score").is_null()).height
    if "grade_status" in q.columns:
        invalid_output = q.filter(pl.col("grade_status") == "invalid_judge_output").height
        report["invalid_judge_outputs"] = invalid_output
        if invalid_output: issues.append(f"quality: {invalid_output} invalid judge outputs require retry")
    if "score_version" not in q.columns: issues.append("quality: score_version missing")
    if "prompt_truncated" in q.columns:
        truncated = q.filter(pl.col("prompt_truncated").fill_null(False)).height
        report["prompt_truncated"] = truncated
        if truncated: issues.append(f"quality: {truncated} judge prompts were truncated")
    expected = set(traces.filter(pl.col("task_type").is_in(["moral", "idea"]))
                   ["trace_id"].to_list()) if traces.height else set()
    missing = len(expected - set(q["trace_id"].to_list()))
    report["missing_rows"] = missing
    if missing: issues.append(f"quality: {missing} moral/idea traces missing")
    return report, issues


def audit_judge(pattern: str | None, traces: pl.DataFrame) -> tuple[dict, list[str]]:
    if not pattern:
        return {}, []
    paths = sorted(glob.glob(pattern)); j = _load(paths); issues = []
    report = {"files": paths, "rows": j.height}
    if j.height == 0:
        return report, [f"no behavior judge rows matched {pattern}"]
    report["manifests_valid"], manifest_issues = _artifact_manifests(paths)
    issues.extend(f"behavior judge: {x}" for x in manifest_issues)
    if "score_version" not in j.columns:
        issues.append("behavior judge: score_version missing")
    trunc_cols = [c for c in ("judge_input_truncated", "judge_context_truncated") if c in j.columns]
    if trunc_cols:
        truncated = j.filter(pl.any_horizontal(
            [pl.col(c).fill_null(False) for c in trunc_cols])).height
        report["rows_with_truncated_judge_context"] = truncated
        if truncated: issues.append(f"behavior judge: {truncated} rows used truncated context")
    if {"kim_parsed", "gandhi_parsed"}.issubset(j.columns):
        invalid = j.filter(~pl.col("kim_parsed").fill_null(False) |
                           ~pl.col("gandhi_parsed").fill_null(False)).height
        report["rows_with_parse_failure"] = invalid
        if invalid: issues.append(f"behavior judge: {invalid} rows contain parse failures")
    else:
        issues.append("behavior judge: parse-validity flags missing")
    if traces.height and "reasoning_text_for_analysis" in traces.columns:
        expected = set(traces.filter(
            pl.col("reasoning_text_for_analysis").is_not_null() &
            (pl.col("reasoning_text_for_analysis").str.len_chars() > 20)
        )["trace_id"].to_list())
        missing = len(expected - set(j["trace_id"].to_list()))
        report["missing_traces"] = missing
        if missing: issues.append(f"behavior judge: {missing} eligible traces missing")
    if {"seg_idx", "n_segments", "context_mode"}.issubset(j.columns):
        coverage = j.group_by(["trace_id", "context_mode"]).agg(
            pl.len().alias("rows"), pl.col("seg_idx").n_unique().alias("unique"),
            pl.col("n_segments").first().alias("expected"))
        bad = coverage.filter((pl.col("rows") != pl.col("expected")) |
                              (pl.col("unique") != pl.col("expected"))).height
        report["traces_with_incomplete_segment_coverage"] = bad
        if bad: issues.append(f"behavior judge: {bad} trace/context groups have incomplete segments")
    return report, issues


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks-dir", default="data/tasks")
    ap.add_argument("--traces-glob", default="data/traces/traces_*.parquet")
    ap.add_argument("--grades-glob", default=None)
    ap.add_argument("--quality-glob", default=None)
    ap.add_argument("--judge-glob", default=None)
    ap.add_argument("--json-out", default=None)
    ap.add_argument("--strict-v2", action="store_true")
    a = ap.parse_args()
    tasks, ti = audit_tasks(Path(a.tasks_dir))
    traces, tri, trace_df = audit_traces(a.traces_glob)
    grades, gi = audit_grades(a.grades_glob, trace_df)
    quality, qi = audit_quality(a.quality_glob, trace_df)
    judge, ji = audit_judge(a.judge_glob, trace_df)
    issues = ti + tri + gi + qi + ji
    result = {"tasks": tasks, "traces": traces, "grades": grades,
              "quality": quality, "judge": judge, "issues": issues, "ok": not issues}
    rendered = json.dumps(result, indent=2, default=str)
    print(rendered)
    if a.json_out:
        out = Path(a.json_out); out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_suffix(out.suffix + ".tmp"); tmp.write_text(rendered + "\n"); tmp.replace(out)
    return 1 if a.strict_v2 and issues else 0


if __name__ == "__main__":
    raise SystemExit(main())
