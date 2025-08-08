#!/usr/bin/env python3
"""Group **failed** benchmark records by failure-category and dump their
anonymised free-text responses verbatim.

For each JSON object in the input file we:
  - keep only failures (``predicted_label == 0`` or final_response contains
    "failure"),
  - pull together the agent's *thoughts*
  - anonymise URLs, domain names (- ``<DOMAIN>``), e-mails, and glaring
    proper nouns, then
  - append the cleaned text to its category bucket.

The output is a markdown file whose sections are the categories; within
each section the examples are simply concatenated in the order found.  This
raw corpus can feed a prompt-optimiser or manual review.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

# full URLs
_URL_RE = re.compile(r"https?://\S+|www\.[A-Za-z0-9./_%+-]+", re.I)
# bare domain names like example.com or gov.some.gov
_DOMAIN_RE = re.compile(r"\b[\w.-]+\.(?:com|gov|net|org|edu|co)\b", re.I)
# e-mails
_EMAIL_RE = re.compile(r"[\w.+-]+@[\w.-]+", re.I)
# loose proper-noun detector (one-to-two capitalised words)
_ENTITY_RE = re.compile(r"\b(?:[A-Z][a-z]{2,}\s+){0,2}[A-Z][a-z]{2,}\b")


def anonymize(text: str) -> str:
    """Strip URLs, *.com / *.gov domains, emails, and proper nouns."""
    text = _URL_RE.sub("<URL>", text)
    text = _DOMAIN_RE.sub("<DOMAIN>", text)
    text = _EMAIL_RE.sub("<EMAIL>", text)
    text = _ENTITY_RE.sub("<ENTITY>", text)
    return text

_CATEGORY_RULES: List[Tuple[str, re.Pattern[str]]] = [
    (
        "sort_not_applied",
        re.compile(
            r"(?i)(?:\bsort(?:ed|ing)?|order).*?(?:not|unable|fail|remained).*?(?:apply|select|set)",
            re.S,
        ),
    ),
    (
        "filter_wrong_value_or_range",
        re.compile(
            r"(?i)(?:filter|range|max|min|price|year|bed|day|size|rating).*?(?:deviation|too|incorrect|not\s+met|include|broader|different).*?(?:filter|applied)",
            re.S,
        ),
    ),
    (
        "location_mismatch",
        re.compile(
            r"(?i)(?:zip|location|city).*?(?:changed|shows|incorrect|mismatch|different)",
            re.S,
        ),
    ),
    (
        "action_not_completed",
        re.compile(
            r"(?i)(?:unable|failed|could\s+not|cannot).*?(?:add\s+to\s+cart|set\s+.*home\s+store|subscribe|upload|quote|submit|complete|click)",
            re.S,
        ),
    ),
    (
        "element_not_found",
        re.compile(
            r"(?i)(?:checkbox|button|option|element|filter).*?(?:not.*?interactive|not.*?found|unavailable|hidden)",
            re.S,
        ),
    ),
    (
        "site_blocked",
        re.compile(
            r"(?i)(?:blocked|captcha|access\s+denied|sign\s*up\s+required|login\s+required)",
            re.S,
        ),
    ),
    (
        "manual_filtering",
        re.compile(r"(?i)manual(?:ly)?\s+(?:filter|scan|check)", re.S),
    ),
    (
        "wrong_object_type",
        re.compile(
            r"(?i)(?:house|land|apartment|room|couch|shoe|listing).*?(?:instead|not).*?(?:house|land|apartment|room|shoe)",
            re.S,
        ),
    ),
    (
        "first_not_confirmed",
        re.compile(
            r"(?i)(?:first|earliest|cheapest|oldest).*?(?:not|unable|fail).*?(?:reach|confirm)",
            re.S,
        ),
    ),
    (
        "time_range_not_matched",
        re.compile(
            r"(?i)(?:past\s+\d+\s*(?:days|months)|first\s+available|departure|next\s+month).*?(?:not|incorrect|fail).*?(?:apply|set)",
            re.S,
        ),
    ),
    (
        "calculation_wrong_inputs",
        re.compile(
            r"(?i)(?:calculator|input|zip).*?(?:incorrect|wrong|not).*?(?:applied|used)",
            re.S,
        ),
    ),
    (
        "external_dependency_failed",
        re.compile(
            r"(?i)(?:validation\s+error|form\s+.*error|quote\s+.*failed|reloads?|reloading)",
            re.S,
        ),
    ),
    (
        "filter_not_applied",
        re.compile(
            r"(?i)(?:filter|sort).*?(?:not|unable|fail).*?(?:apply|select|click)",
            re.S,
        ),
    ),
    # fallback / guard‑rail – must be last
    ("other", re.compile(r".*", re.S)),
]




def categorize(text: str) -> str:
    for name, pat in _CATEGORY_RULES:
        if pat.search(text):
            return name
    return "other"

def iter_failed_blobs(jsonl_path: Path) -> Iterable[Tuple[str, str]]:
    """Yield (clean_text, category) for every failed record in *jsonl_path*."""
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            failed = (
                rec.get("predicted_label") == 0 or
                (isinstance(rec.get("final_result_response"), str) and
                 "failure" in rec["final_result_response"].lower())
            )
            if not failed:
                continue

            response = ""
            if rec.get("evaluation_details"):
                response = rec["evaluation_details"].get("response", "")
            if not response:
                response = ""
            blob_clean = anonymize(response)
            yield blob_clean, categorize(blob_clean)

def make_report(blobs_by_cat: Dict[str, List[str]]) -> str:
    """Return a simple markdown string with all blobs concatenated per category."""
    lines: List[str] = ["# Failure Corpus (anonymised)\n"]
    for cat, entries in sorted(blobs_by_cat.items(), key=lambda kv: (-len(kv[1]), kv[0])):
        lines.append(f"## {cat.replace('_', ' ').title()}  ({len(entries)} examples)\n")
        lines.extend(entries)
        lines.append("")  # blank line between categories
    return "\n".join(lines)

def main() -> None:
    p = argparse.ArgumentParser(description="Group failed tasks into categories and dump all anonymised responses verbatim.")
    p.add_argument("--jsonl", type=Path, help="JSONL benchmark file")
    p.add_argument("--out", type=Path, default=Path("failure_corpus.md"), help="Output markdown file")
    args = p.parse_args()

    blobs_by_cat: Dict[str, List[str]] = defaultdict(list)
    for blob, cat in iter_failed_blobs(args.jsonl):
        blobs_by_cat[cat].append(blob)

    args.out.write_text(make_report(blobs_by_cat), encoding="utf-8")
    print(f"▶ Saved {sum(len(v) for v in blobs_by_cat.values())} blobs into {len(blobs_by_cat)} categories → {args.out}")


if __name__ == "__main__":
    main()
