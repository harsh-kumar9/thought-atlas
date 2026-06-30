"""src/segment/segmenter.py — Project wrapper around the vendored ThinkARM splitter.

Adds, on top of the byte-identical ThinkARM segmentation (thinkarm_vendored.py):
  * a stable per-trace segment table (trace_id, seg_idx, sentence, section_type)
  * normalized position in [0,1] used for the 0-100% "heartbeat" temporal analysis
    (ThinkARM Fig.3 convention: position = seg_idx / (n_segments - 1))
  * convenience to segment a whole traces parquet -> segments parquet

What it deliberately does NOT do: change tokenization, sentence boundaries, or the
think/answer split. That is the vendored method, kept exact per project requirement.

The annotation taxonomy (Kim conversational + Gandhi cognitive) is applied DOWNSTREAM
by the judge, per segment. ThinkARM assigns one Schoenfeld episode per sentence; how we
map our 8-code dual taxonomy onto per-sentence labels is a judge-schema decision
(see configs/ + DESIGN_NOTES once resolved), not a segmentation decision.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import polars as pl

from .thinkarm_vendored import process_response_to_sentences


SEGMENTS_SCHEMA = {
    "trace_id": pl.Utf8,
    "seg_idx": pl.Int32,        # 0-based order within the trace
    "n_segments": pl.Int32,     # total segments in this trace
    "norm_pos": pl.Float64,     # seg_idx / (n_segments-1) in [0,1]; 0.0 if single seg
    "section_type": pl.Utf8,    # "think" | "answer" (ThinkARM convention)
    "sentence": pl.Utf8,
}


def segment_text(text: str) -> list[dict]:
    """Segment one trace string into ThinkARM sentences with normalized positions."""
    raw = process_response_to_sentences(text or "", apply_merging=True)
    n = len(raw)
    out = []
    for i, s in enumerate(raw):
        out.append({
            "seg_idx": i,
            "n_segments": n,
            "norm_pos": (i / (n - 1)) if n > 1 else 0.0,
            "section_type": s["type"],
            "sentence": s["sentence"],
        })
    return out


def segment_traces_df(
    traces: pl.DataFrame,
    *,
    text_col: str = "full_text",
    id_col: str = "trace_id",
    min_segments: int = 1,
) -> pl.DataFrame:
    """Explode a traces DataFrame into a long segment table (one row per sentence).

    Carries trace_id only; join back to the traces table for task_type/model/etc.
    `min_segments` lets the temporal analysis later restrict to length>=N (ThinkARM
    uses >=300 *tokens*; we expose a segment-count floor and leave token floors to
    the analysis config).
    """
    rows = []
    for r in traces.iter_rows(named=True):
        segs = segment_text(r.get(text_col) or "")
        if len(segs) < min_segments:
            continue
        tid = r[id_col]
        for s in segs:
            rows.append({"trace_id": tid, **s})
    if not rows:
        return pl.DataFrame(schema=SEGMENTS_SCHEMA)
    return pl.DataFrame(rows).select(list(SEGMENTS_SCHEMA.keys()))


def segment_parquet(in_path: Path, out_path: Path, *, text_col: str = "full_text") -> int:
    df = pl.read_parquet(in_path)
    seg = segment_traces_df(df, text_col=text_col)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    seg.write_parquet(out_path)
    return seg.height


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Segment a traces parquet into sentences (ThinkARM method).")
    ap.add_argument("--in", dest="in_path", required=True)
    ap.add_argument("--out", dest="out_path", required=True)
    ap.add_argument("--text-col", default="full_text")
    a = ap.parse_args()
    n = segment_parquet(Path(a.in_path), Path(a.out_path), text_col=a.text_col)
    print(f"wrote {n} segments -> {a.out_path}")
