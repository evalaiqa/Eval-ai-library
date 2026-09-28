# role_adherence.py
"""
Role Adherence Metric: Evaluates how well the AI assistant maintains its assigned
role and character throughout a multi-turn conversation.

Score calculation: Softmax aggregation of adherence verdicts
"""
import json
from typing import List, Dict, Any, Tuple
from eval_lib.testcases_schema import ConversationalEvalTestCase
from eval_lib.metric_pattern import ConversationalMetricPattern
from eval_lib.llm_client import chat_complete
from eval_lib.utils import score_agg, extract_json_block


# Verdict weights for role adherence levels
VERDICT_WEIGHTS = {
    "fully": 1.0,      # Perfectly maintains role throughout
    "mostly": 0.9,     # Minor deviations but stays in character
    "partial": 0.7,    # Some role breaks but generally consistent
    "minor": 0.3,      # Frequently breaks character
    "none": 0.0        # Completely ignores assigned role
}


class RoleAdherenceMetric(ConversationalMetricPattern):
    """
    Evaluates how consistently an AI assistant adheres to its assigned role
    across multiple conversation turns.
    """

    name = "roleAdherenceMetric"

    def __init__(
        self,
        model: str,
        threshold: float = 0.7,
        temperature: float = 0.5,
        verbose: bool = False,
        chatbot_role: str = ""
    ):
        """
        Initialize Role Adherence metric.

        Args:
            model: LLM model name
            threshold: Success threshold (0.0-1.0)
            temperature: Score aggregation temperature for softmax
        """
        super().__init__(model=model, threshold=threshold, verbose=verbose)
        self.temperature = temperature
        # Store on the same attribute evaluate() reads (and the parent defines),
        # so a role set via metric settings is actually used instead of falling
        # back to "No role specified".
        self.chatbot_role = chatbot_role

    # ==================== HELPER METHODS ====================

    @staticmethod
    def _render_dialogue(turns) -> str:
        """Convert conversation turns into readable format"""
        return "\n".join(
            f"{i+1}. User: {t.input}\n   Assistant: {t.actual_output}"
            for i, t in enumerate(turns)
        )

    @staticmethod
    def _prompt_label_help() -> str:
        """Explanation of role adherence verdict levels"""
        return """Rate role adherence (worst → best):

none    – completely ignores assigned role
minor   – frequently breaks character
partial – some role breaks but generally consistent
mostly  – minor deviations but stays in character
fully   – perfectly maintains role throughout"""

    @staticmethod
    def _prompt_few_shot() -> str:
        """Few-shot examples for verdict generation"""
        return """Example GOOD:
Role: You are a professional medical advisor. Be formal and evidence-based.
Conversation:
1. User: What causes headaches?
   Assistant: Headaches can result from various factors including dehydration, stress, or underlying medical conditions. I recommend consulting a healthcare provider for persistent symptoms.
Verdicts:
[{"verdict":"fully","reason":"Maintained formal medical tone and evidence-based approach"}]

Example BAD:
Role: You are a professional financial advisor. Use formal language.
Conversation:
1. User: Should I invest in stocks?
   Assistant: Yo dude! Stocks are totally rad! Just YOLO into them lol 🚀
Verdicts:
[{"verdict":"none","reason":"Completely abandoned professional tone and formal language requirement"}]"""

    # ==================== CORE EVALUATION STEPS ====================

    async def _generate_verdicts(
        self,
        role_description: str,
        dialogue_text: str
    ) -> Tuple[List[Dict[str, str]], float, float]:
        """
        Generate role adherence verdicts for each conversation turn.

        Args:
            role_description: The assigned role/character description
            dialogue_text: Formatted conversation text

        Returns:
            Tuple of (verdicts_list, aggregated_score, llm_cost)
        """
        prompt = f"""{self._prompt_label_help()}

{self._prompt_few_shot()}

Now evaluate the following conversation.

ASSIGNED ROLE:
{role_description}

DIALOGUE:
{dialogue_text}

Task: Judge how well the assistant stays in character throughout the conversation.

For each assistant reply, assign a verdict: "fully", "mostly", "partial", "minor", or "none".

Return JSON array:
[{{"verdict": "fully|mostly|partial|minor|none", "reason": "<explanation>"}}, ...]"""

        text, cost = await chat_complete(
            self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0
        )

        try:
            raw_json = extract_json_block(text)
            verdicts = json.loads(raw_json)

            if not isinstance(verdicts, list):
                raise ValueError("Expected JSON array of verdicts")

            # Calculate aggregated score from verdicts
            weights = [VERDICT_WEIGHTS.get(
                v.get("verdict", "none"), 0.0) for v in verdicts]
            score = round(score_agg(weights, temperature=self.temperature), 4)

            return verdicts, score, cost or 0.0

        except Exception as e:
            raise RuntimeError(f"Failed to parse verdicts: {e}\n{text}")

    async def _summarize_verdicts(
        self,
        verdicts: List[Dict[str, str]]
    ) -> Tuple[str, float]:
        """
        Generate concise summary of role adherence assessment.

        Args:
            verdicts: List of verdict objects with reasons

        Returns:
            Tuple of (summary_text, llm_cost)
        """
        # Take up to 6 most relevant verdicts for summary
        bullets = "\n".join(f"- {v['reason']}" for v in verdicts[:6])

        prompt = (
            "Write a concise (max 2 sentences) summary of how well the chatbot stayed in character, "
            "based on these observations:\n\n"
            f"{bullets}\n\n"
            "Summary:"
        )

        text, cost = await chat_complete(
            self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0
        )

        return text.strip(), cost or 0.0

    # ==================== MAIN EVALUATION ====================

    async def evaluate(self, test_case: ConversationalEvalTestCase) -> Dict[str, Any]:
        """
        Evaluate role adherence across conversation turns.

        Steps:
        1. Extract chatbot role from test case
        2. Format dialogue into readable text
        3. Generate adherence verdicts (fully/mostly/partial/minor/none)
        4. Aggregate verdicts into final score using softmax
        5. Generate summary explanation
        6. Build comprehensive evaluation log

        Args:
            test_case: Conversational test case with multiple turns and chatbot role

        Returns:
            Evaluation results with score, success, reason, cost, and detailed log
        """
        total_cost = 0.0

        # Step 1: Extract role
        role_description = test_case.chatbot_role or self.chatbot_role or "No role specified"

        # Step 2: Format dialogue
        dialogue_text = self._render_dialogue(test_case.turns)

        # Step 3: Generate role adherence verdicts
        verdicts, verdict_score, cost = await self._generate_verdicts(
            role_description,
            dialogue_text
        )
        total_cost += cost

        # Step 4: Generate summary explanation
        summary, cost = await self._summarize_verdicts(verdicts)
        total_cost += cost

        # Step 5: Determine success
        final_score = verdict_score
        success = final_score >= self.threshold

        # Step 6: Build evaluation log
        evaluation_log = {
            "chatbot_role": role_description,
            "comment_chatbot_role": "The assigned role/character the assistant should maintain.",
            "dialogue": dialogue_text,
            "comment_dialogue": "Full conversation text used for role adherence evaluation.",
            "number_of_turns": len(test_case.turns),
            "comment_number_of_turns": "Total conversation turns analyzed.",
            "verdicts": verdicts,
            "comment_verdicts": "LLM-generated verdicts assessing adherence level per turn (fully/mostly/partial/minor/none).",
            "verdict_weights": {v["verdict"]: VERDICT_WEIGHTS.get(v["verdict"], 0.0) for v in verdicts},
            "comment_verdict_weights": "Numeric weights assigned to each verdict for score calculation.",
            "final_score": final_score,
            "comment_final_score": f"Weighted average of verdict scores using softmax aggregation (temperature={self.temperature}).",
            "threshold": self.threshold,
            "temperature": self.temperature,
            "success": success,
            "comment_success": "Whether the role adherence score meets the required threshold.",
            "final_reason": summary,
            "comment_reasoning": "Concise explanation of how well the assistant maintained its assigned role."
        }

        result = {
            "name": self.name,
            "score": final_score,
            "success": success,
            "reason": summary,
            "evaluation_cost": round(total_cost, 6),
            "evaluation_log": evaluation_log
        }

        self.print_result(result)

        return result
