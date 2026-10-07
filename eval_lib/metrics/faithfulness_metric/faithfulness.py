# faithfulness_metric.py
'''
Faithfulness Metric: Evaluates the factual consistency of a chatbot's answer
with respect to the retrieved context.
Score calculation: Softmax aggregation of verdicts on factual statements
'''
from typing import List, Dict, Tuple, Any
import json
import re
import numpy as np
from math import exp
from eval_lib.testcases_schema import EvalTestCase
from eval_lib.metric_pattern import MetricPattern
from eval_lib.llm_client import chat_complete
from eval_lib.utils import score_agg, extract_json_block

VERDICT_WEIGHTS = {
    "fully": 1.0,
    "mostly": 0.9,
    "partial": 0.7,
    "minor": 0.3,
    "none": 0.0,
}

# Statement types produced by the verdict step. Only "fact" and "inference"
# count toward the faithfulness score; pure restatements of the user's
# question and non-claims (offers, questions, calls to action, pleasantries)
# are kept in the log but excluded from scoring. A missing/unknown type
# defaults to a scored statement, preserving backward compatibility with
# judge replies that do not emit a type.
EXCLUDED_TYPES = {"input_restatement", "non_claim"}


def _statement_type(verdict: Dict[str, Any]) -> str:
    return (verdict.get("type") or "fact").strip().lower()


def _is_scored(verdict: Dict[str, Any]) -> bool:
    return _statement_type(verdict) not in EXCLUDED_TYPES


class FaithfulnessMetric(MetricPattern):
    name = "faithfulnessMetric"
    requires_actual_output = True

    def __init__(
            self,
            model: str,
            threshold: float = 0.7,
            temperature: float = 0.5,
            verbose: bool = False
    ):
        super().__init__(model=model, threshold=threshold, verbose=verbose)
        self.temperature = temperature

    async def _generate_statements(self, answer: str) -> Tuple[List[str], float]:
        prompt = (
            "Extract the key factual claims from the following answer.\n\n"
            "Rules:\n"
            "- Each claim must be a single, verifiable factual statement.\n"
            "- Ignore greetings, meta-comments (\"Sure!\", \"Here's...\"), stylistic phrases, "
            "and offers/questions/calls to action (\"share your order ID and I can check\").\n"
            "- Do NOT split one sentence into micro-facts. Keep claims at sentence-level granularity.\n"
            "- Combine closely related details into one claim rather than listing separately.\n"
            "- Maximum 8 claims. Focus on the most important facts.\n\n"
            f"Answer:\n{answer}\n\n"
            "Return a JSON array of strings."
        )
        text, cost = await chat_complete(self.model, [{"role": "user", "content": prompt}], temperature=0.0)
        raw_json = extract_json_block(text)
        statements = json.loads(raw_json)
        assert isinstance(statements, list)
        return statements, cost or 0.0

    async def _generate_verdicts(self, context: str, question: str, statements: List[str]) -> Tuple[List[Dict[str, str]], float, float]:
        """Single-step classify-and-verdict.

        Key points:
        - The verdict step sees the user's QUESTION, not only the context, so
          statements that restate the question or combine a question fact with
          a context fact are judged correctly instead of as unsupported.
        - Each statement is classified (fact / inference / input_restatement /
          non_claim). Only facts and inferences are scored; restatements and
          non-claims are excluded (but kept in the log).
        - Paraphrasing = "fully"; "none" only for contradictions or zero info.
        """
        prompt = (
            "Evaluate how well each statement is supported.\n\n"
            "You are given the retrieval CONTEXT (the only knowledge source) and the "
            "user's QUESTION (reference only, NOT a source of truth).\n\n"
            "Step 1 - classify each statement:\n"
            "- fact: a factual claim about the domain/knowledge base. Judge it against the CONTEXT.\n"
            "- inference: a conclusion that combines a fact stated in the QUESTION with a fact in "
            "the CONTEXT (e.g. the question says \"20 days\", the context says \"within 14 days\", "
            "so \"outside the return window\"). Judge \"fully\" when the CONTEXT premise it relies "
            "on is present; do NOT mark it \"none\" merely because the context does not repeat the "
            "question's fact.\n"
            "- input_restatement: only repeats information from the QUESTION and asserts nothing "
            "about the CONTEXT. (Excluded from scoring.)\n"
            "- non_claim: an offer, question, call to action, or pleasantry "
            "(e.g. \"share your order ID and I can check the details\"). (Excluded from scoring.)\n\n"
            "Step 2 - for fact and inference, find the relevant context passage, then assign a verdict:\n"
            "- fully: The core meaning is clearly present in the context (exact wording NOT required).\n"
            "- mostly: The main idea is supported but with minor differences in details "
            "(e.g., approximate numbers, paraphrased names).\n"
            "- partial: Some parts are supported but key information is missing or incomplete.\n"
            "- minor: Only tangentially related; the context mentions the topic but not the specific claim.\n"
            "- none: The claim directly contradicts the context, OR the context contains "
            "absolutely no related information.\n"
            "For input_restatement and non_claim use the verdict \"n/a\".\n\n"
            "Key distinctions:\n"
            "- Paraphrasing or using synonyms = \"fully\" (not \"mostly\").\n"
            "- Missing exact numbers/dates but correct overall = \"mostly\" (not \"partial\" or \"none\").\n"
            "- Use \"none\" ONLY when the context contradicts the claim or has zero relevant information.\n\n"
            f"CONTEXT:\n{context}\n\n"
            f"QUESTION:\n{question}\n\n"
            f"STATEMENTS (JSON array):\n{json.dumps(statements, ensure_ascii=False)}\n\n"
            "Return only a JSON array of objects like:\n"
            '[{"type": "fact|inference|input_restatement|non_claim", '
            '"verdict": "fully|mostly|partial|minor|none|n/a", '
            '"reason": "<brief explanation>", '
            '"support": "<exact context sentence(s) that support/contradict this claim> or \'none\'"}]'
        )
        text, cost = await chat_complete(self.model, [{"role": "user", "content": prompt}], temperature=0.0)
        raw_json = extract_json_block(text)
        verdicts: List[Dict[str, Any]] = json.loads(raw_json)

        # Safety check: a plain fact claimed as fully/mostly but with no supporting
        # passage is downgraded. Inferences legitimately may not quote a single
        # passage, so they are exempt.
        for v in verdicts:
            if _statement_type(v) == "fact":
                supp = (v.get("support") or "").strip().lower()
                if supp in ("none", "") and v.get("verdict") in ("fully", "mostly"):
                    v["verdict"] = "partial"

        scored = [v for v in verdicts if _is_scored(v)]
        scores = [VERDICT_WEIGHTS.get((v.get("verdict") or "").strip().lower(), 0.0) for v in scored]
        # No scorable factual claims (e.g. the answer only asks for an order ID)
        # => nothing can be unfaithful to the context.
        score = round(score_agg(scores, temperature=self.temperature), 4) if scores else 1.0
        return verdicts, score, cost or 0.0

    async def _summarize_reasons_via_llm(self, verdicts: List[Dict[str, str]]) -> Tuple[str, float]:
        grouped: Dict[str, List[str]] = {}
        for v in verdicts:
            grouped.setdefault(v.get("verdict", ""), []).append(v.get("reason", ""))
        bullets = []
        for tag in ("fully", "mostly", "partial", "none"):
            bullets.extend(f"- {r}" for r in grouped.get(tag, [])[:2])
        prompt = (
            "Summarize the following points from a factual consistency evaluation.\n"
            "Give one short paragraph (1-2 sentences) that explains whether the answer "
            "was well supported by the context, mentioning both strong and weak parts.\n\n"
            f"{chr(10).join(bullets)}\n\n"
            "Summary:"
        )
        text, cost = await chat_complete(self.model, [{"role": "user", "content": prompt}], temperature=0.0)
        return text.strip(), cost or 0.0

    async def evaluate(self, test_case: EvalTestCase) -> Dict[str, any]:
        llm_cost = 0.0
        answer = test_case.actual_output
        context = "\n".join(test_case.retrieval_context or [])
        question = test_case.input

        # 1. Statements from answer
        statements, cost = await self._generate_statements(answer)
        llm_cost += cost

        # 2. Verdicts against context (question passed so restatements/inferences
        #    are judged correctly rather than as unsupported)
        verdicts, verdict_score, cost = await self._generate_verdicts(context, question, statements)
        llm_cost += cost

        # 3. Reason summary
        summary_reason, cost = await self._summarize_reasons_via_llm(verdicts)
        llm_cost += cost

        success = verdict_score >= self.threshold

        evaluation_log = {
            "input_question": question,
            "retrieval_context": test_case.retrieval_context,
            "answer": answer,
            "statements": statements,
            "comment_statements": "Factual assertions extracted from the answer.",
            "verdicts": verdicts,
            "comment_verdicts": (
                "Each verdict shows the statement type and how well a fact/inference is "
                "supported by the context. Statements typed input_restatement or non_claim "
                "are excluded from the score."
            ),
            "scored_statement_count": sum(1 for v in verdicts if _is_scored(v)),
            "comment_scored_statement_count": "Number of fact/inference statements that counted toward the score.",
            "final_score": verdict_score,
            "comment_final_score": "Final score based on faithfulness of statements.",
            "threshold": self.threshold,
            "success": success,
            "comment_success": "Whether the score meets the required threshold.",
            "final_reason": summary_reason,
            "comment_reasoning": "Summary explanation based on all verdicts."
        }

        result = {
            "name": self.name,
            "score": verdict_score,
            "success": success,
            "reason": summary_reason,
            "evaluation_cost": round(llm_cost, 6),
            "evaluation_log": evaluation_log
        }
        self.print_result(result)

        return result
