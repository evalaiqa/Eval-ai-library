# calibration/budget.py
"""
Budget and cost control for calibration LLM work.

A single :class:`Budget` object is shared across all the parallel LLM calls of
one task (analyst / re-evaluation / optimizer). It is asyncio-safe: the check
and the counter increment happen under one lock so two coroutines cannot both
slip past the last allowed call.

The call that *would* exceed ``max_llm_calls`` is never made — the task stops
cleanly and returns a partial result with ``stopped_by_budget=True``. Cost is
only known after a call returns (from ``chat_complete``'s cost), so the cost cap
stops the task once the accumulated cost has reached the limit; if the model
returns no cost (platform credit calls), only ``max_llm_calls`` is enforced.

:class:`_BudgetedModel` is the single chokepoint: wrap any model spec in it and
every ``chat_complete`` through it is counted and costed against the budget.
"""

import asyncio
from typing import Any, List, Optional

from eval_lib.llm_client import CustomLLMClient, chat_complete


class BudgetExceeded(Exception):
    """Raised when the next LLM call would exceed the budget."""


class InsufficientLabels(Exception):
    """Raised when a calibration task does not have enough labelled cases."""


class Budget:
    """Shared, asyncio-safe call/cost cap for one calibration task."""

    def __init__(
        self,
        max_llm_calls: Optional[int] = None,
        max_cost_usd: Optional[float] = None,
    ):
        self.max_llm_calls = max_llm_calls
        self.max_cost_usd = max_cost_usd
        self.llm_calls = 0
        self.cost_usd = 0.0
        self._lock = asyncio.Lock()

    async def check_and_reserve(self) -> None:
        """Reserve one call slot. Raises :class:`BudgetExceeded` if the call
        would exceed the call cap, or if accumulated cost already reached the
        cost cap."""
        async with self._lock:
            if self.max_llm_calls is not None and self.llm_calls >= self.max_llm_calls:
                raise BudgetExceeded(f"max_llm_calls={self.max_llm_calls} reached")
            if self.max_cost_usd is not None and self.cost_usd >= self.max_cost_usd:
                raise BudgetExceeded(
                    f"max_cost_usd={self.max_cost_usd} reached (cost so far {self.cost_usd})"
                )
            self.llm_calls += 1

    async def add_cost(self, cost: Optional[float]) -> None:
        """Record the cost of a completed call."""
        if not cost:
            return
        async with self._lock:
            self.cost_usd += float(cost)

    def snapshot(self) -> dict:
        return {"llm_calls": self.llm_calls, "cost_usd": round(self.cost_usd, 6)}


class _BudgetedModel(CustomLLMClient):
    """Wrap a model spec (string / descriptor / CustomLLMClient) so every call
    goes through a shared :class:`Budget`. Delegates to the real
    ``chat_complete`` and records the returned cost.
    """

    def __init__(self, inner: Any, budget: Optional[Budget] = None):
        self._inner = inner
        self._budget = budget

    async def chat_complete(self, messages: List[dict], temperature: float = 0.0):
        if self._budget is not None:
            await self._budget.check_and_reserve()
        text, cost = await chat_complete(self._inner, messages=messages, temperature=temperature)
        if self._budget is not None:
            await self._budget.add_cost(cost)
        return text, cost

    def get_model_name(self) -> str:
        if isinstance(self._inner, CustomLLMClient):
            return str(self._inner.get_model_name())
        return str(self._inner)
