'''
Tool Correctness Metric: Evaluates whether the correct tools were called
during the execution of an AI agent.
Score calculation: Proportion of expected tools correctly called
'''

from typing import Dict, Any, List
from eval_lib.metric_pattern import MetricPattern
from eval_lib.testcases_schema import EvalTestCase


class ToolCorrectnessMetric(MetricPattern):
    name = "toolCorrectnessMetric"

    def __init__(
        self,
        threshold: float = 0.5,
        verbose: bool = False,
        evaluation_params: List[str] = [],
        exact_match: bool = False,
        check_ordering: bool = False
    ):
        super().__init__(model=None, threshold=threshold, verbose=verbose)
        self.evaluation_params = evaluation_params
        self.exact_match = exact_match
        self.check_ordering = check_ordering

    async def evaluate(self, test_case: EvalTestCase) -> Dict[str, Any]:
        self.tools_called = test_case.tools_called or []
        self.expected_tools = test_case.expected_tools or []

        score = self.calculate_score()
        reason = self.generate_reason()

        result = {
            "name": self.name,
            "score": score,
            "success": score >= self.threshold,
            "reason": reason,
            "evaluation_cost": 0.0  # No LLM cost for this metric
        }

        self.print_result(result)

        return result

    def generate_reason(self) -> str:
        called_names = self.tools_called
        expected_names = self.expected_tools

        if self.exact_match:
            if self.calculate_exact_match_score() == 1.0:
                return f"Exact match: all expected tools {expected_names} were called exactly."
            else:
                return f"Mismatch: expected {expected_names}, called {called_names}."
        elif self.check_ordering:
            lcs, weighted = self.compute_weighted_lcs()
            if weighted == len(self.expected_tools):
                return "Correct tool usage and order."
            else:
                return f"Incomplete or unordered: expected {expected_names}, got {called_names}."
        else:
            used_expected = set(self.tools_called) & set(self.expected_tools)
            missing = set(self.expected_tools) - used_expected
            if not missing:
                return f"All expected tools {expected_names} were called."
            else:
                return f"Missing tools {list(missing)}. Expected {expected_names}, got {called_names}."

    def calculate_score(self) -> float:
        if self.exact_match:
            return self.calculate_exact_match_score()
        elif self.check_ordering:
            _, score = self.compute_weighted_lcs()
            # No tools expected => nothing to order, requirement trivially met.
            return score / len(self.expected_tools) if self.expected_tools else 1.0
        else:
            return self.calculate_non_exact_match_score()

    def calculate_exact_match_score(self) -> float:
        if len(self.tools_called) != len(self.expected_tools):
            return 0.0
        for i in range(len(self.tools_called)):
            if self.tools_called[i] != self.expected_tools[i]:
                return 0.0
        return 1.0

    def calculate_non_exact_match_score(self) -> float:
        match_count = 0
        used = set()
        for expected in self.expected_tools:
            for i, called in enumerate(self.tools_called):
                if i in used:
                    continue
                if expected == called:
                    match_count += 1
                    used.add(i)
                    break
        # No tools were expected => the agent correctly needed none, score 1.0.
        # (Extra calls are not penalised in non-exact mode; use exact_match=True
        # if calling tools when none are expected should fail.)
        return match_count / len(self.expected_tools) if self.expected_tools else 1.0

    def compute_weighted_lcs(self):
        m, n = len(self.expected_tools), len(self.tools_called)
        dp = [[0.0] * (n + 1) for _ in range(m + 1)]

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if self.expected_tools[i - 1] == self.tools_called[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

        score = dp[m][n]
        return [], score
