from __future__ import annotations

import asyncio
from typing import Any, Optional

from playwright.async_api import async_playwright, BrowserContext, Playwright
from langchain_core.language_models import BaseChatModel
from browser_use.browser.session import BrowserSession
from browser_use.browser.profile import BrowserProfile
from browser_use import Agent as BrowserUseAgent
from .planner import ChecklistPlanner

from dataclasses import dataclass

@dataclass
class SubtaskStatus:
    name: str
    done: bool = False
    error: str | None = None

    def mark_done(self) -> None:
        self.done = True

    def mark_error(self, err: str) -> None:
        self.error = err

async def _keep_last_page(context):
    """
    Close every open tab but the newest one and return that last page.
    """
    # Filter out already-closed handles
    pages = [p for p in context.pages if not p.is_closed()]
    if not pages:
        return None

    # Last opened (index -1) is the page we want to keep
    last = pages[-1]
    # Close every earlier page
    for p in pages[:-1]:
        try:
            await p.close()
        except Exception:
            pass  # ignore if it vanished meanwhile
    return last

class HierarchicalAgent:
    """
    Plans a high-level task into a checklist of subtasks, then runs a BrowserUseAgent
    on each subtask while preserving a single browser session.  The full, step-by-step
    action history from *all* subtasks is returned in Mind2Web format so callers
    (e.g. save_mind2web_trace) can dump screenshots, thoughts, etc.
    """

    def __init__(
        self,
        llm: BaseChatModel,
        show_browser: bool = False,
        max_actions_per_subtask: int = 25,
        initial_actions: list[dict[str, dict[str, Any]]] | None = None,
        subtask_timeout: float = 60.0,
        sensitive_data: Optional[dict[str, Any]] = None,
    ) -> None:
        self.llm = llm
        self.planner = ChecklistPlanner(llm)
        self.show_browser = show_browser
        self.max_actions_per_subtask = max_actions_per_subtask
        self.initial_actions = initial_actions
        self.subtask_timeout = subtask_timeout
        self.playwright: Playwright | None = None
        self.context: BrowserContext | None = None
        self.sensitive_data = sensitive_data or {}

    async def _init_browser(self) -> None:
        self.playwright = await async_playwright().start()
        browser = await self.playwright.chromium.launch(headless=not self.show_browser)
        self.context = await browser.new_context()

    async def run(self, task: str, website: str) -> dict[str, Any]:
        """
        Execute the full high-level *task* on *website*.

        Returns
        -------
        dict
            A Mind2Web-compatible dict with a single key `"history"`, whose
            value is a list of step dictionaries taken from every subtask in
            order.  This can be fed directly to `save_mind2web_trace`.
        """
        if not self.context:
            await self._init_browser()

        #Plan the checklist
        checklist = await self.planner.plan(task, website)

        profile = BrowserProfile(
            # Vision / DOM capture
            use_vision=True,
            viewport_expansion=0,
            wait_for_fonts_ready=False,
            wait_for_network_idle_page_load_time=0.0,
            maximum_wait_screenshot_time=15_000, 
            post_action_wait_ms=750,
        )

        session = BrowserSession(
            playwright=self.playwright,
            browser_context=self.context,
            browser_profile=profile,
            keep_alive=True,
            headless=True,
            browser_args=[
                "--no-sandbox",
                "--disable-blink-features=AutomationControlled",
                "--disable-web-security",
                "--disable-notifications",
            ],
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/115.0.0.0 Safari/537.36"
            ),
        )
        await session.start()

        statuses: list[SubtaskStatus] = [SubtaskStatus(name=s) for s in checklist]
        combined_history: list[dict[str, Any]] = []
        page = None  # will hold the active Playwright page across subtasks
        
        try:
            #Iterate over subtasks
            for idx, subtask in enumerate(checklist, start=1):
                print(f"[Agent] Subtask {idx}/{len(checklist)}  ➜  {subtask}")

                session.agent_current_page  = page
                session.human_current_page  = page 
                
                params: dict[str, Any] = {
                    "task": subtask,
                    "llm": self.llm,
                    "browser_session": session,
                    "page": page,
                    "max_actions_per_step": 1,
                    "validate_output": True,
                }
                if idx == 1 and self.initial_actions:
                    params["initial_actions"] = self.initial_actions
                    
                if self.sensitive_data:
                    params["sensitive_data"] = self.sensitive_data
                    
                progress_lines = []
                for j, st in enumerate(statuses, start=1):
                    if j < idx:
                        if st.done:
                            progress_lines.append(f"✔️ Step {j}: {st.name}")
                        elif st.error:
                            progress_lines.append(f"❌ Step {j}: {st.name} (error: {st.error})")
                    else:
                        progress_lines.append(f"🔄 Step {j}: {st.name} (COMING UP NEXT)")
                formatted_progress = "\n".join(progress_lines)

                extra = f"""
                ### Context
                You are an **autonomous browser agent** executing one checklist item at a time inside a
                shared browser session.

                • **High-level goal (for reference only)**  
                {task}

                ### Checklist status
                {formatted_progress}

                ### How to work on the *current* subtask
                1. Start in the active tab.  
                2. **If** you need new information, the element you need is missing, or you have taken
                two actions without progress – **open a NEW tab** and run a Google query whose
                keywords come from the subtask description.  
                3. Close any tabs you no longer need so that **only one tab is left** before you finish.  
                4. When the subtask is complete, return `DONE` and any required result data.

                Stay strictly inside the scope of the current subtask; do not begin future subtasks early.
                """
                
                print(extra)
                
                params["extend_system_message"] = extra

                agent = BrowserUseAgent(**params)

                #Enforce an overall timeout for this subtask
                try:
                    sub_history = await asyncio.wait_for(
                        agent.run(max_steps=self.max_actions_per_subtask),
                        timeout=self.subtask_timeout,
                    )
                    statuses[idx - 1].mark_done()
                except asyncio.TimeoutError:
                    print(f"[Agent] ⚠️  Subtask {idx} timed out after {self.subtask_timeout}s")
                    err = f"timed out after {self.subtask_timeout}s"
                    statuses[idx - 1].mark_error(err)
                    raise
                except Exception as e:
                    print(f"[Agent] ❌  Error in subtask {idx}: {e}")
                    statuses[idx - 1].mark_error(str(e))
                    raise

                #Accumulate step-level history
                sub_dump = (
                    sub_history.model_dump()
                    if hasattr(sub_history, "model_dump")
                    else sub_history
                )
                combined_history.extend(sub_dump.get("history", []))

                # 5️⃣  Hand off the current page to the next subtask
                page = await _keep_last_page(session.browser_context)
                
        finally:
            try:
                await session.kill()
            finally:
                if self.playwright:
                    await self.playwright.stop()

        #Clean up browser resources
        await session.kill()
        if self.playwright:
            await self.playwright.stop()

        #Return a Mind2Web-style history dict
        return {"history": combined_history,
                "subtask_history": [s.__dict__ for s in statuses]}
