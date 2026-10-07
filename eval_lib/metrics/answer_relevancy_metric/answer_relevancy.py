# answer_relevancy.py
'''
AnswerRelevancyMetric: Evaluates how well a chatbot's answer addresses the user's intent by extracting 
factual statements from the answer and assessing their relevance to the inferred intent using an LLM.

Score is based on the proportion of relevant statements, with detailed verdicts and reasoning provided.
'''

from typing import List, Dict, Any, Tuple
import json
from math import exp
from eval_lib.testcases_schema import EvalTestCase
from eval_lib.metric_pattern import MetricPattern
from eval_lib.llm_client import chat_complete
from eval_lib.utils import score_agg, extract_json_block

# Constants for verdict weights
VERDICT_WEIGHTS = {
    "fully": 1.0,    # Fully related
    "mostly": 0.9,   # Mostly related
    "partial": 0.7,  # Partially related
    "minor": 0.3,    # Weekly related
    "none": 0.0      # Not related
}

# Statement types produced by the verdict step. Non-claims (offers, questions,
# calls to action, pleasantries) are kept in the log but excluded from the
# score. A missing/unknown type defaults to a scored claim, preserving
# backward compatibility with judge replies that do not emit a type.
EXCLUDED_TYPES = {"non_claim"}


def _is_scored(verdict: Dict[str, Any]) -> bool:
    return (verdict.get("type") or "claim").strip().lower() not in EXCLUDED_TYPES


class AnswerRelevancyMetric(MetricPattern):
    name = "answerRelevancyMetric"
    requires_actual_output = True

    def __init__(
        self,
        model: str,
        threshold: float = 0.6,
        temperature: float = 0.5,
        verbose: bool = False
    ):
        super().__init__(model=model, threshold=threshold, verbose=verbose)
        self.temperature = temperature

    async def _infer_user_intent(self, question: str) -> str:
        prompt = (
            "Determine the user's intent behind the following question.\n"
            "Answer in ONE concise sentence without adding extra details.\n\n"
            f"Question: {question}"
        )
        response, cost = await chat_complete(self.model, [{"role": "user", "content": prompt}], temperature=0.0)
        return response.strip(), cost or 0.0

    async def _generate_statements(self, intent: str, answer: str) -> Tuple[List[str], float]:
        prompt = (
            "Extract the key factual claims from the following answer.\n\n"
            f"User intent: {intent}\n\n"
            f"Answer:\n{answer}\n\n"
            "Rules:\n"
            "- Each claim must be a single, verifiable statement.\n"
            "- Ignore greetings, meta-comments, disclaimers, and offers to help.\n"
            "- Do NOT split one sentence into micro-facts. Keep claims at sentence-level.\n"
            "- Include both relevant AND irrelevant statements.\n"
            "- Maximum 8 claims. Focus on the most substantive facts.\n"
            "- Output as a JSON array of strings."
        )
        text, cost = await chat_complete(self.model, [{"role": "user", "content": prompt}], temperature=0.0)
        try:
            raw_json = extract_json_block(text)
            statements = json.loads(raw_json)
            assert isinstance(statements, list)
            return statements, cost or 0.0
        except Exception as e:
            raise RuntimeError(f"Failed to parse statements: {e}\n{text}")

    async def _generate_verdicts(self, question: str, intent: str, statements: List[str]) -> Tuple[List[Dict[str, str]], float, float]:
        """Single-step verdict with improved prompt for relevancy."""
        prompt = (
            "You are an impartial evaluator.\n\n"
            "Step 1 - classify each statement:\n"
            "- non_claim: an offer, question, call to action, or pleasantry "
            "(e.g. \"share your order ID and I can check\"). Use the verdict \"n/a\"; "
            "it is excluded from scoring.\n"
            "- claim: anything else. Give it a verdict below.\n\n"
            "Step 2 - for each claim, decide how directly it fulfils the user intent "
            "using the 5-level scale:\n"
            "- fully: Explicitly answers the intent with concrete information (examples count as fully).\n"
            "- mostly: Clearly supports the intent; small details may be missing.\n"
            "- partial: Related to the topic but only partially addresses the intent.\n"
            "- minor: Weak or tangential relation.\n"
            "- none: Irrelevant or completely off-topic.\n\n"
            "Key distinctions:\n"
            "- Examples and concrete details that illustrate the answer = \"fully\" (not \"mostly\").\n"
            "- Background context that helps understand the answer = \"mostly\" (not \"partial\").\n"
            "- A true, on-topic detail about the SAME entity the user asked about "
            "(e.g. the specific order's items, total, or note when the user asked about that order) "
            "is at least \"minor\" - never \"none\" - even if it does not address the exact aspect asked.\n"
            "- Use \"none\" ONLY for statements about a DIFFERENT entity or completely unrelated to the question.\n\n"
            f"USER INTENT: {intent}\n\n"
            f"USER QUESTION:\n{question}\n\n"
            f"STATEMENTS (JSON array):\n{json.dumps(statements, ensure_ascii=False)}\n\n"
            "Return only a JSON array of objects:\n"
            '[{"type": "claim|non_claim", "verdict": "fully|mostly|partial|minor|none|n/a", "reason": "<one sentence>"}]'
        )
        text, cost = await chat_complete(self.model, [{"role": "user", "content": prompt}], temperature=0.0)
        raw_json = extract_json_block(text)
        verdicts = json.loads(raw_json)

        scored = [v for v in verdicts if _is_scored(v)]
        scores = [VERDICT_WEIGHTS.get((v.get("verdict") or "").strip().lower(), 0.0) for v in scored]
        # No scorable claims (e.g. the answer is only an offer to help)
        # => nothing substantive addressed the question.
        verdict_score = round(score_agg(scores, temperature=self.temperature), 4) if scores else 0.0
        return verdicts, verdict_score, cost or 0.0

    async def _summarize_reasons_via_llm(
        self,
        verdicts: List[Dict[str, str]],
    ) -> Tuple[str, float]:

        grouped: Dict[str, List[str]] = {}
        for v in verdicts:
            grouped.setdefault(v.get("verdict", ""), []).append(v.get("reason", ""))

        bullets: List[str] = []
        for tag in ("fully", "mostly", "partial", "minor", "none"):
            if tag in grouped:
                examples = grouped[tag][:2]
                bullets.extend(f"- {r}" for r in examples)

        reasons_block = "\n".join(bullets)

        prompt = (
            "You are an expert evaluator who writes crisp 1-2-sentence summaries."
            "Below are bulleted findings from an answer-relevancy check. "
            "Write a single concise explanation (max two sentences) that sums up "
            "how well the answer met the user's request, mentioning the main strengths "
            "and the biggest gap. Do not enumerate bullets, just a unified summary.\n\n"
            f"{reasons_block}\n\n"
            "Unified explanation:"
        )

        text, cost = await chat_complete(
            self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0
        )
        return text.strip(), cost or 0.0

    async def evaluate(self, test_case: EvalTestCase) -> Dict[str, Any]:
        llm_cost = 0.0
        question = test_case.input
        answer = test_case.actual_output

        # Step 1: Infer the user's intent from the question
        intent, cost = await self._infer_user_intent(question)
        llm_cost += cost

        # Step 2: Generate statements that the answer would fully address
        statements, cost = await self._generate_statements(intent, answer)
        llm_cost += cost

        # Step 3: Generate verdicts for each statement (non-claims excluded,
        #         on-topic entity details floored at "minor" inside the step)
        verdicts, verdict_score, cost = await self._generate_verdicts(question, intent, statements)
        llm_cost += cost

        # Step 4: Summarize the verdict reasons
        summary_reason, cost = await self._summarize_reasons_via_llm(verdicts)
        llm_cost += cost

        # Step 5: Count final score based on verdicts
        final_score = verdict_score
        success = final_score >= self.threshold

        # Step 6: Verbose log
        evaluation_log = {
            "input_question": question,
            "answer": answer,
            "user_intent": intent,
            "comment_user_intent": "Inferred goal of the question.",
            "statements": statements,
            "comment_statements": "Atomic facts extracted from the answer.",
            "verdicts": verdicts,
            "comment_verdicts": (
                "Each verdict gives the statement type and its relevance to the question. "
                "Statements typed non_claim (offers, questions, calls to action) are excluded "
                "from the score; on-topic details about the asked-about entity are floored at 'minor'."
            ),
            "scored_statement_count": sum(1 for v in verdicts if _is_scored(v)),
            "comment_scored_statement_count": "Number of claim statements that counted toward the score.",
            "verdict_score": verdict_score,
            "comment_verdict_score": "Proportion of relevant statements in the answer.",
            "final_score": final_score,
            "comment_final_score": "Score based on the proportion of relevant statements.",
            "threshold": self.threshold,
            "temperature": self.temperature,
            "success": success,
            "comment_success": "Whether the score exceeds the pass threshold.",
            "final_reason": summary_reason,
            "comment_reasoning": "Compressed explanation of the key verdict rationales."
        }

        result = {
            "name": self.name,
            "score": final_score,
            "success": success,
            "reason": summary_reason,
            "evaluation_cost": round(llm_cost, 6),
            "evaluation_log": evaluation_log
        }
        self.print_result(result)

        return result
