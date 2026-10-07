# contextual_relevancy_llm.py
'''
Contextual Relevancy Metric: Evaluates how well the retrieved context supports
the user's question and inferred intent.

Score calculation: Softmax aggregation of relevancy verdicts

'''
from typing import List, Dict, Tuple, Any
import json
import re
from math import exp
import numpy as np
from eval_lib.testcases_schema import EvalTestCase
from eval_lib.metric_pattern import MetricPattern
from eval_lib.llm_client import chat_complete
from eval_lib.utils import score_agg, extract_json_block

# weights for each verdict category
VERDICT_WEIGHTS = {
    "fully": 1.0,
    "mostly": 0.9,
    "partial": 0.7,
    "minor": 0.3,
    "none": 0.0,
}


class ContextualRelevancyMetric(MetricPattern):
    name = "contextualRelevancyMetric"

    def __init__(
            self,
            model: str,
            threshold: float = 0.6,
            temperature: float = 0.5,
            verbose: bool = False
    ):
        super().__init__(model=model, threshold=threshold, verbose=verbose)
        self.temperature = temperature

    async def _infer_user_intent(self, question: str) -> Tuple[str, float]:
        """
        Ask the LLM to summarize the user's intent in one sentence.
        """
        prompt = (
            "Determine the user's intent behind this question.\n"
            "Answer in one concise sentence.\n\n"
            f"Question: {question}"
        )
        resp, cost = await chat_complete(
            self.model,
            [{"role": "user", "content": prompt}],
            temperature=0.0
        )
        return resp.strip(), cost or 0.0

    async def _generate_verdicts(
        self,
        intent: str,
        context: List[str],
        question: str
    ) -> Tuple[List[Dict[str, str]], float, float]:
        """
        For each context segment, ask the LLM to classify its relevance 
        to the inferred intent with a 5-level verdict and a brief reason.
        """
        prompt = (
            "You are evaluating how well each CONTEXT segment serves both the user's explicit question and underlying intent.\n\n"
            f"USER QUESTION: {question}\n\n"
            f"USER INTENT: {intent}\n\n"
            "CONTEXT SEGMENTS (JSON array):\n"
            f"{json.dumps(context, ensure_ascii=False)}\n\n"
            "For each segment, evaluate its relevance to BOTH the specific question asked AND the user's broader intent.\n"
            "Return an object for each segment:\n"
            '{"verdict": "fully|mostly|partial|minor|none", "reason": "<one-sentence explaining relevance to question and intent>"}\n'
            "Respond with a JSON array ONLY.\n\n"
            "Verdict levels:\n"
            "- fully: directly answers the question and completely addresses the user's intent\n"
            "- mostly: addresses the question well and covers most of the user's intent with minor gaps\n"
            "- partial: partially relevant to the question or intent but missing key information\n"
            "- minor: tangentially related to either the question or intent\n"
            "- none: not relevant to the question or user's intent\n\n"
            "Key rule:\n"
            "- A segment that is about the SAME entity the user asked about but does not directly "
            "answer the question (e.g. the order's delivery note when the question is that order's "
            "status) is at least \"minor\" - never \"none\".\n"
            "- Use \"none\" ONLY for a segment about a DIFFERENT entity or completely unrelated to the question."
        )
        resp, cost = await chat_complete(
            self.model,
            [{"role": "user", "content": prompt}],
            temperature=0.0
        )
        raw = extract_json_block(resp)
        verdicts = json.loads(raw)
        # compute weights list
        scores = [VERDICT_WEIGHTS.get(v["verdict"].lower(), 0.0)
                  for v in verdicts]
        agg = score_agg(scores, temperature=self.temperature)
        return verdicts, round(agg, 4), cost or 0.0

    async def _summarize_reasons(
        self,
        verdicts: List[Dict[str, str]]
    ) -> Tuple[str, float]:
        """
        Take the top two and bottom one verdict reasons and ask the LLM
        to write a unified 1–2 sentence summary of context relevancy.
        """
        # sort by weight
        sorted_by_weight = sorted(
            verdicts,
            key=lambda v: VERDICT_WEIGHTS.get(v["verdict"].lower(), 0.0),
            reverse=True
        )
        top_reasons = [v["reason"] for v in sorted_by_weight[:2]]
        bottom_reasons = [v["reason"] for v in sorted_by_weight[-1:]]
        bullets = "\n".join(f"- {r}" for r in top_reasons + bottom_reasons)

        prompt = (
            "You are an expert evaluator."
            "Below are key points about how context segments matched the user's question and intent:\n\n"
            f"{bullets}\n\n"
            "Write a concise 1–2 sentence summary explaining overall how relevant "
            "the retrieved context is to answering the user's question and meeting their needs."
        )
        resp, cost = await chat_complete(
            self.model,
            [{"role": "user",   "content": prompt}],
            temperature=0.0
        )
        return resp.strip(), cost or 0.0

    async def evaluate(self, test_case: EvalTestCase) -> Dict[str, Any]:
        llm_cost = 0.0
        question = test_case.input
        context = test_case.retrieval_context or []

        # 1) Infer intent
        intent, cost = await self._infer_user_intent(question)
        llm_cost += cost

        # 2) Generate verdicts for each context segment
        verdicts, score, cost = await self._generate_verdicts(intent, context, question)
        llm_cost += cost

        # 3) Summarize reasons
        summary, cost = await self._summarize_reasons(verdicts)
        llm_cost += cost

        success = score >= self.threshold

        evaluation_log = {
            "input_question": question,
            "user_intent": intent,
            "retrieval_context": context,
            "comment_verdicts": "Each shows if a statement is grounded in the context.",
            "verdicts": verdicts,
            "final_score": score,
            "comment_final_score": "Weighted support score from context.",
            "threshold": self.threshold,
            "temperature": self.temperature,
            "success": success,
            "comment_success": "Whether the score exceeds the threshold.",
            "final_reason": summary,
            "comment_reasoning": "LLM-generated explanation based on verdict rationales."
        }

        result = {
            "name": self.name,
            "score": score,
            "success": success,
            "reason": summary,
            "evaluation_cost": round(llm_cost, 6),
            "evaluation_log": evaluation_log
        }

        self.print_result(result)

        return result
