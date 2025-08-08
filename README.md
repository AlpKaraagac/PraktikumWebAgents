# Browser-Use x Online-Mind2Web - Enhanced Web-Agent Framework

> **TL;DR** Automated browsing powered by LLMs (Planner -> Executor -> Validator) with prompt-optimisation tooling and evaluation on the **Online-Mind2Web** benchmark.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Key Components](#key-components)
3. [Setup - Quick Start](#Setup - Quick Start)
4. [Running Benchmarks](#Running Benchmarks)
---

## Project Overview

Large-language-model (LLM) **GUI agents** can imitate human clicks & typing and thus automate arbitrary web tasks.
This repo contains our *improved* fork of the open-source **Browser-Use** agent plus:

* a **hierarchical Planner -> Executor -> Validator** loop that decomposes tasks, executes subtasks, and sanity-checks outcomes.
* an **automatic prompt-optimiser** that analyses failed benchmark traces and refines the system prompt.
* evaluation scripts for the **Online-Mind2Web** benchmark (live websites, harder than Mind2Web-offline).

---

## Key Components

| Layer                 | Role                                                                              | Highlights                                           |
| --------------------- | --------------------------------------------------------------------------------- | ---------------------------------------------------- |
| **Planner**           | Decomposes a natural-language task into ordered subtasks.                         | Uses a lightweight LLM call; replans adaptively.     |
| **Executor**          | Runs each subtask via **Browser-Use** in a Playwright browser context.            | Single-action per DOM snapshot to avoid index drift. |
| **Validator**         | Inspects state after each subtask to confirm success or trigger recovery.         | Catches missing filters, wrong pages, etc.           |
| **Prompt-Optimiser**  | After a benchmark run, clusters and summarises failures - proposes prompt tweaks. | Fully automated; anonymous rule extraction.          |
| **Benchmark Harness** | Interfaces with **Online-Mind2Web** for scoring.                                  | Handles CAPTCHA blocks & pop-ups heuristically.      |

---

## Setup - Quick Start

> Tested on **Ubuntu 22.04 / Python 3.11**.
> **Playwright** needs Chromium; ensure you can download browser binaries.

```bash
# 1 / Clone the code bases
$ git clone https://github.com/browser-use/browser-use.git
$ git clone https://github.com/OSU-NLP-Group/Online-Mind2Web.git

# 2 / Checkout the exact commit used in the paper
$ cd browser-use && \
  git switch --detach 4e1266fbece75724894957baeba022dbf94b4b02 && cd ..

# 3 / Set-up Browser-Use (creates an isolated venv at browser-use/.venv)
$ ./browser-use/bin/setup.sh

# 4 / Activate the virtualenv
$ source browser-use/.venv/bin/activate

# 5 / Install extra Python deps
(browser-use) $ uv pip install datasets google.generativeai

# 6 / Install Playwright & browsers
(browser-use) $ playwright install

# 7/Patch Online‑Mind2Web with our modified files
(browser-use) $ cp -r Online-Mind2web-replace/* Online-Mind2Web/
```

That's it. The environment now contains *Browser-Use + our additions*, *Online-Mind2Web* harness, and all runtime deps.

> **Heads-up:**  You need valid API keys (e.g. `OPENAI_API_KEY`, `GOOGLE_API_KEY`) in the shell environment for LLM calls.

---

## Running Benchmarks

```bash
# Activate venv first (if not already)
$ source browser-use/.venv/bin/activate

# Running the benchmark (edit if you want hierarchical or not)
(browser-use) $ ./full_pipeline.sh

# Auto optimize prompt
(browser-use) $ ./auto_optimize.sh
```