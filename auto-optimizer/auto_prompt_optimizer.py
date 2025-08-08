#!/usr/bin/env python3
"""Automatic prompt optimiser driven by Gemini Flash.

Inputs
------
1. *base_prompt.txt* (the current system-/setup-prompt used by your agent)
2. *category_summaries.md* (output from `summarize_failure_corpus.py`)

The script sends both to Gemini-2-Flash with explicit instructions:
- analyse weaknesses, 
- return an **edited prompt** ready for drop-in replacement.

It then prints a unified **diff** of original - proposed prompt so you can
see exactly what would change.
"""
from __future__ import annotations

import argparse
import difflib
import os
from pathlib import Path
from typing import List

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import SystemMessage, HumanMessage

# --------------------------- LLM setup --------------------------- #
LLM = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0.2,
    timeout=90.0,
)

# --------------------------- helpers ----------------------------- #

def make_llm_prompt(base_prompt: str, summary_md: str) -> str:
    return (
        "You are a senior prompt-engineer. Improve the following prompt so "
        "that an autonomous browser-navigation agent avoids its most common "
        "failure categories listed afterwards.\n\n"
        "--- CURRENT PROMPT ---\n" + base_prompt.strip() + "\n"
        "--- FAILURE THEMES ---\n" + summary_md.strip() + "\n\n"
        "--- TASK ---\n"
        "Return the **full, edited prompt**.\n"
        "Keep style and structure; only add/re‑phrase lines needed to mitigate "
        "failures. Mark your output exactly like:\n"
        "Keep all the important parts of the prompt for correct output formatting.\n"
        "<BEGIN_OPTIMISED_PROMPT>\n" "...prompt here..." "\n<END_OPTIMISED_PROMPT>\n"
    )


def extract_new_prompt(response: str) -> str:
    if "<BEGIN_OPTIMISED_PROMPT>" in response:
        return response.split("<BEGIN_OPTIMISED_PROMPT>")[1].split("<END_OPTIMISED_PROMPT>")[0].strip()
    # fallback - assume whole content is the prompt
    return response.strip()

# --------------------------- main ----------------------------- #

def main() -> None:
    ap = argparse.ArgumentParser(description="Optimise an agent prompt using Gemini Flash and failure summaries.")
    ap.add_argument("--prompt_file", type=Path, help="Path to current prompt text/md file")
    ap.add_argument("--summary_md", type=Path, help="Path to category_summaries.md")
    ap.add_argument("--out", type=Path, default=Path("optimised_prompt.txt"), help="Write optimised prompt here")
    args = ap.parse_args()

    base_prompt = args.prompt_file.read_text(encoding="utf-8")
    summaries = args.summary_md.read_text(encoding="utf-8")

    sys_msg = SystemMessage(content="You are ChatGPT acting as the LLM front‑end.")
    user_msg = HumanMessage(content=make_llm_prompt(base_prompt, summaries))

    print("→ Contacting Gemini Flash …")
    response = LLM([sys_msg, user_msg]).content
    new_prompt = extract_new_prompt(response)

    # save
    args.out.write_text(new_prompt, encoding="utf-8")
    print(f"✔ Optimised prompt written to {args.out}\n")

    # show diff
    diff: List[str] = list(difflib.unified_diff(
        base_prompt.splitlines(),
        new_prompt.splitlines(),
        fromfile="current_prompt",
        tofile="optimised_prompt",
        lineterm="",
    ))
    print("\n=== Prompt Diff (unified) ===")
    if diff:
        for ln in diff:
            print(ln)
    else:
        print("<no changes proposed>")


if __name__ == "__main__":
    if not os.getenv("GOOGLE_API_KEY"):
        raise SystemExit("GOOGLE_API_KEY env var not set.")
    main()
