#!/usr/bin/env python3
import asyncio
import json
import argparse
import sys
import os
import base64
import csv
from pathlib import Path
import random

from datasets import load_dataset
from langchain_google_genai import ChatGoogleGenerativeAI
from browser_use import Agent
from browser_use.browser.context import BrowserSession
from agent.hierarchical_agent import HierarchicalAgent
from typing import Any, List, Dict
from dataclasses import asdict, is_dataclass

async def save_mind2web_trace(
    history: Any,
    task_desc: str,
    output_dir: Path,
    hierarchical: bool = False,
) -> None:
    """
    Convert an AgentHistoryList *or* HierarchicalAgent output into Mind2Web
    format:

      output_dir/trajectory/0.png, 1.png, ...
      output_dir/result.json

    Parameters
    ----------
    history : AgentHistoryList | dict
        The object returned by `agent.run()`.  When `hierarchical` is True,
        this is the dict returned by `HierarchicalAgent.run()`.
    task_desc : str
        Natural-language description of the overall task.
    output_dir : pathlib.Path
        Destination directory; created if missing.
    hierarchical : bool, default False
        Pass True when the history came from a HierarchicalAgent so that
        its `"subtask_history"` field (if present) is copied through.
    """
    data: Dict[str, Any] = (
        history.model_dump() if hasattr(history, "model_dump") else history
    )
    hist: List[dict] = data.get("history", [])

    traj_dir = output_dir / "trajectory"
    traj_dir.mkdir(parents=True, exist_ok=True)

    screenshot_paths: List[str] = []
    action_history: List[str] = []
    thoughts: List[str] = []
    final_response: str | None = None
    for i, step in enumerate(hist):
        state = step.get("state", {})
        shot = state.get("screenshot", "")
        dst_rel: str | None = None

        if shot:
            try:
                # a) file path on disk
                if os.path.exists(shot):
                    dst = traj_dir / f"{i}.png"
                    os.link(os.path.abspath(shot), dst)
                    dst_rel = f"trajectory/{i}.png"
                # b) base-64 encoded bytes
                else:
                    img_data = base64.b64decode(shot)
                    with open(traj_dir / f"{i}.png", "wb") as imgf:
                        imgf.write(img_data)
                    dst_rel = f"trajectory/{i}.png"
            except Exception:
                # ignore anything we fail to decode / copy
                pass

        if dst_rel:
            screenshot_paths.append(dst_rel)

        mo = step.get("model_output")
        if mo is not None:
            text = mo if isinstance(mo, str) else mo.get("text", mo.get("content", str(mo)))
            thoughts.append(text)
            action_history.append(text)

        for r in step.get("result", []):
            if r.get("is_done") or r.get("error"):
                final_response = r.get("error") or "<done>"

    result: Dict[str, Any] = {
        "task": task_desc,
        "action_history": action_history,
        "thoughts": "\n---\n".join(thoughts) if thoughts else None,
        "final_result_response": final_response,
        "input_image_paths": screenshot_paths[:1] if screenshot_paths else [],
    }

    # When called for HierarchicalAgent, copy its sub-task status list through
    if hierarchical and "subtask_history" in data:
        sub_h = data["subtask_history"]
        # Convert dataclass instances - dicts just in case
        result["subtask_history"] = [
            asdict(s) if is_dataclass(s) else s for s in sub_h
        ]
    with open(output_dir / "result.json", "w") as rf:
        json.dump(result, rf, indent=2)

async def run_task(row, base_out: Path, LLM, hierarchical: bool):
    """
    Run a single task and emit Mind2Web trace directly.
    """
    # prepare directories
    level_dir = base_out / row['level']
    task_dir = level_dir / row['task_id']
    task_dir.mkdir(parents=True, exist_ok=True)

    # 1. launch browser
    browser_session = BrowserSession(
        headless=True,
        browser_args=[
            "--no-sandbox",
            "--disable-blink-features=AutomationControlled",
            "--disable-web-security",
        ],
        user_agent=(
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/115.0.0.0 Safari/537.36"
        ),
    )
    # This might result in a warning by browser_use as it expects allowed URLs to be set in the case of sensitive data. For us not important.
    sensitive_data = {
        'https://amazon.de': {
            'username': 'alp59371@gmail.com',
            'password': 'testAccount123'
        },
    }
    initial = [{"go_to_url": {"url": row['website']}}]
    if hierarchical:
        agent = HierarchicalAgent(
            llm=LLM,
            show_browser=False,
            max_actions_per_subtask=30,
            initial_actions=initial,
            sensitive_data=sensitive_data,
        )
    else:
        agent = Agent(
            task=row['confirmed_task'],
            llm=LLM,
            browser_session=browser_session,
            initial_actions=initial,
            sensitive_data=sensitive_data,
            max_actions_per_step=1, 
        )

    # 2. run agent
    try:
        if hierarchical:
            history = await agent.run(row['confirmed_task'], row['website'])
        else:
            history = await agent.run(max_steps=75)
    except Exception as e:
        history = None
        error = str(e)

    # 3. save Mind2Web format
    if history:
        await save_mind2web_trace(history, row['confirmed_task'], task_dir, hierarchical)
    else:
        # write minimal error result.json
        with open(task_dir / 'result.json', 'w') as rf:
            json.dump({
                'task': row['confirmed_task'],
                'action_history': [],
                'thoughts': None,
                'final_result_response': f"Exception: {error}",
                'input_image_paths': []
            }, rf, indent=2)

async def main(args):
    # - 1. LLM setup (from CLI args)
    LLM = ChatGoogleGenerativeAI(
        model=args.model,
        temperature=args.temperature,
        convert_system_message_to_human=True,
    )

    # - 2. Load tasks (from CLI args)
    TASKS = load_dataset(
        "osunlp/Online-Mind2Web",
        name="default",
        split="test",
        token=True,
    )

    base_out = Path(args.output_dir)
    levels = ['medium']
    start_id = args.start_id

    for level in levels:
        # all rows for this level
        rows = [r for r in TASKS if r['level'] == level]

        # respect --start_id, if given
        if start_id:
            try:
                start_index = next(i for i, r in enumerate(rows) if r['task_id'] == start_id)
            except StopIteration:
                start_index = 0
        else:
            start_index = 0

        # slice off everything before start_index
        rows = rows[start_index:]

        print(f"Checking {len(rows)} {level} tasks…")

        for row in rows:
            task_dir = base_out / row["level"] / row["task_id"]
            if task_dir.exists() and any(task_dir.iterdir()):
                print(f"↪︎  Skipping task {row['task_id']} – output already exists.")
                continue
            await run_task(row, base_out, LLM, args.hierarchical)

if __name__=='__main__':
    parser = argparse.ArgumentParser(
        description="Run Online-Mind2Web traces end-to-end"
    )
    parser.add_argument(
        "--output_dir", "-o",
        type=str,
        default="trajectories",
        help="Where to write trajectory/* and result.json"
    )
    parser.add_argument(
        "--start_id",
        type=str,
        default=None,
        help="If set, skip ahead to this task_id"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gemini-2.5-flash",
        help="Gemini/LLM model to use"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        help="LLM temperature"
    )
    parser.add_argument(
        "--hierarchical",
        action="store_true",
        help="Run in hierarchical mode (default: False)",
    )

    args = parser.parse_args()
    orig_err = sys.stderr
    try:
        sys.stderr = open(os.devnull, 'w')
        asyncio.run(main(args))
    finally:
        sys.stderr.close()
        sys.stderr = orig_err