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
from eval_lib.utils import score_agg, extract_json_block, split_into_statements

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


_TOKEN = re.compile(r"[a-z0-9]{3,}")


def _support_grounded(support: str, context: str, min_overlap: float = 0.6) -> bool:
    """True when the judge's support quote is actually drawn from the context.

    The prompt asks for exact context sentence(s), but a drifting judge
    occasionally backs an unsupported claim with an invented or paraphrased
    quote and labels it "fully". Requiring that most of the quote's content
    tokens occur in the context catches invented support while tolerating
    light rewording (case, punctuation, dashes). This keeps the score robust
    to judge drift instead of relying on the judge being deterministic.
    """
    quote_tokens = set(_TOKEN.findall(support.lower()))
    if not quote_tokens:
        return False
    context_tokens = set(_TOKEN.findall(context.lower()))
    return len(quote_tokens & context_tokens) / len(quote_tokens) >= min_overlap


# What a genuine non-claim looks like: an offer by the assistant, a question or
# request addressed to the user, or a pleasantry. Exclusion from scoring is the
# risky direction (a hallucinated procedure that slips out as "non_claim" is
# never checked), so the judge's non_claim label is only honoured when the
# sentence matches one of these shapes; otherwise it is demoted to a scored
# fact and goes through the normal support check.
_NON_CLAIM_PATTERNS = re.compile(
    r"(?:"
    r"\?\s*$"                                                   # a question to the user
    r"|\bi(?:'ll|'d|'m)\b|\b(?:i|we)\s+(?:can|could|will|would|am|are)\b"  # assistant offer
    r"|\blet me\b|\bfeel free\b|\bhappy to help\b|\bglad to help\b"
    r"|\b(?:share|send|tell|give|provide|confirm|let)\s+(?:me|us)\b"      # give the assistant something
    r"|\b(?:share|send|provide|enter|confirm)\s+(?:your|the)\s+"
    r"(?:order\s+id|order\s+number|email|account|card|last\s+4|details?)\b"
    r"|\bif you\s+(?:tell|share|send|give|provide)\b"
    r"|\bplease\s+(?:share|send|provide|tell|let|confirm|enter)\b"
    r"|\b(?:thanks|thank you|you're welcome)\b"
    r")",
    re.IGNORECASE,
)


def _looks_like_non_claim(statement: str) -> bool:
    return bool(_NON_CLAIM_PATTERNS.search((statement or "").strip()))


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
        """Deterministic sentence-level statements (no LLM call).

        The answer is split into sentences by `split_into_statements`, so the
        statement set is identical on every run. This removes the extraction
        step's run-to-run variance (different counts/splits => different
        denominators and none-fractions), which was the dominant source of
        score spread. Non-claims and restatements are still filtered by type
        in the verdict step.
        """
        return split_into_statements(answer), 0.0

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
            "- non_claim: ONLY an offer by the assistant, a question or request addressed to the user, "
            "or a pleasantry - a sentence that asserts nothing about the domain "
            "(e.g. \"share your order ID and I can check the details\"). (Excluded from scoring.)\n"
            "  An instruction or description of how a process works (e.g. \"start the return by "
            "confirming the item and giving a reason\", \"follow the shipping instructions\") "
            "asserts facts about that process: it is a fact, NOT a non_claim, and must be checked "
            "against the CONTEXT like any other claim.\n\n"
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
            "- Use \"none\" when the context contradicts the claim or has zero relevant information.\n"
            "- A verdict of fully/mostly/partial/minor REQUIRES a supporting quote from the CONTEXT "
            "(for an inference, quote the CONTEXT premise it relies on). If you cannot quote supporting "
            "context, the verdict MUST be \"none\".\n"
            "- Keep reason and verdict consistent: never say the context does not mention or support the "
            "claim while giving a verdict other than \"none\".\n\n"
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

        # Conservative exclusion: honour a non_claim label only for sentences
        # that look like an offer / question / pleasantry. A procedural
        # instruction the judge tags non_claim (it flips on this even with a
        # seed) is demoted to a scored fact; with no context support it then
        # becomes "none" below, instead of silently dropping out of the score.
        if len(verdicts) == len(statements):
            for stmt, v in zip(statements, verdicts):
                if _statement_type(v) == "non_claim" and not _looks_like_non_claim(stmt):
                    v["type"] = "fact"
                    if (v.get("verdict") or "").strip().lower() in ("n/a", ""):
                        v["verdict"] = "none"

        # Support-quote enforcement: a scored statement cannot carry a verdict
        # better than "none" without a supporting quote that is actually drawn
        # from the context. This removes the reason/verdict contradiction (a
        # "partial" whose reason says the context does not mention the claim),
        # the none/partial flip on the same unsupported claim, and the drift
        # mode where the judge backs an unsupported claim with an invented
        # quote. Excluded statements (restatements, non-claims) are untouched.
        for v in verdicts:
            if not _is_scored(v):
                continue
            supp = (v.get("support") or "").strip()
            verdict = (v.get("verdict") or "").strip().lower()
            if verdict not in ("fully", "mostly", "partial", "minor"):
                continue
            if supp.lower() in ("none", "") or not _support_grounded(supp, context):
                v["verdict"] = "none"

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
