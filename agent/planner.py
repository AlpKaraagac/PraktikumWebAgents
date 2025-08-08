
from __future__ import annotations

import os
from typing import List, Literal

from langchain_core.prompts import PromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.language_models import BaseChatModel

__all__ = ["ChecklistPlanner"]

# ---------------------------------------------------------------------------#
# Internal structured schema                                                 #
# ---------------------------------------------------------------------------#
class _Plan(BaseModel):
    steps: List[str] = Field(
        description="Different steps to follow, in sorted order."
    )

# ---------------------------------------------------------------------------#
# Prompt templates                                                            #
# ---------------------------------------------------------------------------#
_PLANNER_PROMPT = PromptTemplate(
    template=(
        "For the given task {task}, produce a concise, step-by-step plan that "
        "a web-automation agent could follow to obtain the final answer from "
        "the specified website. Each step must be explicit, self-contained and "
        "necessary. The last step must yield the answer."
    ),
    input_variables=["task"],
)

_REFINE_PROMPT = PromptTemplate(
    template=(
        "You are given an initial plan: {plan}\n\n"
        "Rewrite it so that every step corresponds to exactly one of these "
        "primitive browser actions:\n"
        "1. open a URL\n"
        "2. click an element\n"
        "3. sort results\n"
        "4. filter results\n"
        "5. change page\n"
        "6. close page\n\n"
        "Each step must include all information required to perform the action. "
        "Return the improved plan in the same JSON schema."
    ),
    input_variables=["plan"],
)

# ---------------------------------------------------------------------------#
# Public class with the *same* signature as ChecklistPlanner                 #
# ---------------------------------------------------------------------------#
class ChecklistPlanner:
    """
    Drop-in replacement for checklist_planner.ChecklistPlanner.

    Parameters
    ----------
    llm_backend : {'openai', 'google'}, optional
        Which chat model family to use.  Default is 'openai'.
    model_name : str, optional
        Model identifier passed to the underlying LangChain chat model.
        Defaults to 'gpt-4o' for OpenAI and 'gemini-2.5-flash' for Google.
    temperature : float, optional
        Sampling temperature for the LLM.  Default 0 (deterministic).
    """

    def __init__(
        self,
        llm: BaseChatModel | None = None,
    ):
        self.llm = llm

        # Build chains once and cache
        self._planner_chain = _PLANNER_PROMPT | self.llm.with_structured_output(_Plan)
        self._refine_chain = _REFINE_PROMPT | self.llm.with_structured_output(_Plan)

    # -------------------------------------------------------------------#
    #  💡  THE ONE PUBLIC METHOD — same signature & return as before     #
    # -------------------------------------------------------------------#
    async def plan(self, task: str, website: str) -> List[str]:
        """
        Generate a checklist plan.

        Parameters
        ----------
        task : str
            High-level task description (e.g. "Find the latest NVIDIA driver").
        website : str
            Base website domain (e.g. "www.nvidia.com").

        Returns
        -------
        List[str]
            Ordered, atomic browser actions — ready for downstream code.
        """
        # 1) Initial coarse plan
        full_task = f"{task.strip()}. Website: {website.strip()}"
        raw_plan: _Plan = await self._planner_chain.ainvoke({"task": full_task})

        # 2) Refine into primitive browser actions
        refined: _Plan = await self._refine_chain.ainvoke({"plan": raw_plan.steps})

        # 3) Defensive validation
        if not all(isinstance(step, str) for step in refined.steps):
            raise TypeError("Planner must return List[str]")

        return refined.steps
