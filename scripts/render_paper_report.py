#!/usr/bin/env python3
"""Render paper_results/REPORT.md to a standalone, readable HTML shell."""

from __future__ import annotations

import argparse
import html
from pathlib import Path

from markdown_it import MarkdownIt


CSS = """
:root { color-scheme: light; --ink:#17212b; --muted:#526071; --line:#dce3ea; --accent:#6d4aff; }
* { box-sizing: border-box; }
body { margin:0; color:var(--ink); background:#f4f6f8; font:16px/1.62 ui-sans-serif,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; }
main { max-width:1040px; margin:32px auto; padding:56px 72px; background:white; box-shadow:0 10px 36px rgba(28,39,52,.08); border-radius:16px; }
h1 { font-size:2.35rem; line-height:1.12; letter-spacing:-.035em; margin:0 0 1rem; }
h2 { font-size:1.55rem; line-height:1.25; margin:3rem 0 1rem; border-top:1px solid var(--line); padding-top:1.25rem; }
h3 { font-size:1.15rem; margin:2rem 0 .5rem; }
p, li { max-width:82ch; }
blockquote { margin:1.5rem 0; padding:1rem 1.25rem; border-left:4px solid var(--accent); background:#f7f5ff; font-size:1.06rem; }
table { width:100%; border-collapse:collapse; margin:1.25rem 0 2rem; font-size:.9rem; }
th, td { border-bottom:1px solid var(--line); padding:.62rem .7rem; text-align:left; vertical-align:top; }
th { background:#f3f5f8; font-weight:650; }
tr:hover td { background:#fafbfc; }
img { display:block; max-width:100%; height:auto; margin:1.5rem auto 2.25rem; border:1px solid var(--line); border-radius:8px; }
code { background:#f1f3f5; padding:.12rem .32rem; border-radius:4px; font-size:.9em; }
pre { overflow:auto; background:#17212b; color:#edf2f7; padding:1rem 1.2rem; border-radius:8px; }
pre code { background:transparent; padding:0; }
strong { font-weight:680; }
@media (max-width:760px) { main { margin:0; padding:28px 22px; border-radius:0; } h1 { font-size:1.9rem; } table { display:block; overflow-x:auto; } }
@media print { body { background:white; } main { box-shadow:none; margin:0; padding:0; max-width:none; } h2 { break-after:avoid; } img, table { break-inside:avoid; } }
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    source = args.source.read_text()
    rendered = MarkdownIt("commonmark", {"html": True}).enable("table").render(source)
    title = source.splitlines()[0].lstrip("# ") if source else "Report"
    document = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(title)}</title>
<style>{CSS}</style>
</head>
<body><main>{rendered}</main></body>
</html>
"""
    args.output.write_text(document)


if __name__ == "__main__":
    main()
