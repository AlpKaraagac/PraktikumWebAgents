#!/usr/bin/env python3
"""Summarise each failure-category in *failure_corpus.md* with Gemini Flash.

The script expects the markdown file created by *analyze_benchmark_results.py*.
For every `## <Category>` section it sends all examples to Gemini 2 Flash and
retrieves a concise summary plus one actionable improvement rule.
"""
from __future__ import annotations

import argparse
import os
import re
from pathlib import Path
from typing import Dict, List, Sequence

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import HumanMessage, SystemMessage

LLM = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.2,
    timeout=60.0,
)

MAX_TOK_PER_CALL = 6000  # conservative safety margin

def load_corpus(md_path: Path) -> Dict[str, List[str]]:
    """Return {category: [blob, …]} extracted from markdown sections."""
    raw = md_path.read_text(encoding="utf-8")
    sections = re.split(r"^## ", raw, flags=re.M)
    blobs_by_cat: Dict[str, List[str]] = {}
    for sec in sections:
        if not sec or sec.startswith("# "):
            continue
        header, *body_lines = sec.splitlines()
        cat = header.split("  ")[0].strip()
        body = [ln for ln in body_lines if ln.strip()]
        blobs_by_cat.setdefault(cat, []).extend(body)
    return blobs_by_cat


def approx_tokens(text: str) -> int:
    # Gemini tokenisation 0.75 words/token in English.
    return int(len(text.split()) / 0.75)

def summarise_blobs(category: str, blobs: Sequence[str]) -> str:
    """Return Gemini summary for *category* (handles chunking if huge)."""

    system = SystemMessage(
        content="""You are an expert QA analyst helping to improve an autonomous web-navigation agent. \
Summarise patterns in failure traces and propose concise improvement rules."""
    )

    def call_llm(lines: Sequence[str]) -> str:
        prompt = (
            f"### Category: {category}\n"
            f"There are {len(lines)} anonymised failure traces below. "
            f"Return up to four bullet points describing common root causes, then ONE new instruction (<120 chars) for the agent prompt.\n\n"
            + "\n".join(lines)
        )
        return LLM([system, HumanMessage(content=prompt)]).content.strip()

    all_text = "\n".join(blobs)
    if approx_tokens(all_text) < MAX_TOK_PER_CALL:
        return call_llm(blobs)

    # chunk-and-reduce when category is very large
    chunk_tokens = MAX_TOK_PER_CALL // 2
    current, bucket, partial_summaries = 0, [], []
    for b in blobs:
        t = approx_tokens(b)
        if current + t > chunk_tokens and bucket:
            partial_summaries.append(call_llm(bucket))
            bucket, current = [], 0
        bucket.append(b)
        current += t
    if bucket:
        partial_summaries.append(call_llm(bucket))

    return call_llm(partial_summaries)


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate Gemini summaries for each failure category.")
    ap.add_argument("--corpus_md", type=Path, help="Path to failure_corpus.md")
    ap.add_argument("--out", type=Path, default=Path("category_summaries.md"), help="Output markdown file path")
    args = ap.parse_args()

    corpus = load_corpus(args.corpus_md)
    out_lines: List[str] = ["# Failure Category Summaries\n"]

    for cat, blobs in corpus.items():
        print(f"→ Summarising {cat}  ({len(blobs)} traces)…")
        out_lines.append(f"## {cat}\n")
        out_lines.append(summarise_blobs(cat, blobs))
        out_lines.append("")

    args.out.write_text("\n".join(out_lines), encoding="utf-8")
    print(f"✔ Summaries written to {args.out}")


if __name__ == "__main__":
    if not os.getenv("GOOGLE_API_KEY"):
        raise SystemExit("GOOGLE_API_KEY environment variable not set.")
    main()
