# few_shot.py
"""
Few-shot example support shared across metrics.

A *few-shot example* is a human-judged case the team shows the judge so it
calibrates to how strictly the team grades. Examples are reference material:
the judge must not copy them and must not score them in place of the current
DATA. They live here (not inside any one metric) so that the base
``MetricPattern`` can render them for every metric that opts in — currently
only ``CustomEvalMetric`` is wired up, but the mechanism is reusable.

Design contract (keep stable):

* With no examples, ``render_few_shot_block`` returns ``""`` so a metric's
  prompt stays byte-for-byte identical to the pre-few-shot version.
* Placeholders ``{{...}}`` inside example fields are rendered **literally** —
  examples are data, not templates, and never affect criterion filtering.
* Order is preserved; examples are never shuffled.
* Hard limits: at most ``FEW_SHOT_MAX_EXAMPLES`` examples (``ValueError`` in the
  constructor, via :func:`normalize_examples`), each field truncated at
  ``FEW_SHOT_FIELD_MAX_CHARS``, and the whole block capped at a caller-supplied
  ``max_chars`` by dropping trailing examples.
"""

from typing import Any, Dict, List, Optional, Tuple, Union

from pydantic import BaseModel, Field

# ── Limits (module constants, not user knobs) ────────────────────────────────
FEW_SHOT_MAX_EXAMPLES = 20
FEW_SHOT_FIELD_MAX_CHARS = 2000
FEW_SHOT_DEFAULT_MAX_CHARS = 12000

_TRUNC_MARKER = " …[truncated]"


class FewShotExample(BaseModel):
    """One human-judged example shown to the judge for calibration.

    ``verdict`` is the overall team decision (pass / fail). ``criterion_verdicts``
    (optional) carries the per-criterion labels on the verdict scale
    (``fully``/``mostly``/``partial``/``minor``/``none``) — shown only in the
    ``verdict`` strategy, index-aligned to the metric's criteria.
    """

    input: str
    actual_output: str
    expected_output: Optional[str] = None
    data: Dict[str, str] = Field(default_factory=dict)
    verdict: str  # "pass" | "fail"
    reason: str = ""
    criterion_verdicts: Optional[List[str]] = None
    source_case_id: Optional[str] = None


def normalize_examples(
    examples: Optional[List[Union["FewShotExample", dict]]],
) -> List[FewShotExample]:
    """Coerce a mixed list of FewShotExample / dict into FewShotExample objects.

    Raises ``ValueError`` when more than ``FEW_SHOT_MAX_EXAMPLES`` are supplied
    (checked at construction time, per spec).
    """
    if not examples:
        return []
    out: List[FewShotExample] = []
    for ex in examples:
        if isinstance(ex, FewShotExample):
            out.append(ex)
        elif isinstance(ex, dict):
            out.append(FewShotExample(**ex))
        else:
            raise TypeError(
                f"few_shot_examples items must be FewShotExample or dict, got {type(ex)!r}"
            )
    if len(out) > FEW_SHOT_MAX_EXAMPLES:
        raise ValueError(
            f"few_shot_examples has {len(out)} items; the maximum is " f"{FEW_SHOT_MAX_EXAMPLES}."
        )
    return out


def _truncate_field(value: Any) -> Tuple[str, bool]:
    """Render a single field value as text, truncating past the field cap."""
    text = value if isinstance(value, str) else str(value)
    if len(text) > FEW_SHOT_FIELD_MAX_CHARS:
        return text[:FEW_SHOT_FIELD_MAX_CHARS] + _TRUNC_MARKER, True
    return text, False


def _render_one(example: FewShotExample, index: int, strategy: str) -> Tuple[str, bool]:
    """Render one example to text. Returns (text, any_field_truncated)."""
    truncated = False
    lines: List[str] = [f"Example {index} (verdict: {example.verdict}):"]

    def add(label: str, value: Any) -> None:
        nonlocal truncated
        rendered, was_trunc = _truncate_field(value)
        truncated = truncated or was_trunc
        if "\n" in rendered:
            lines.append(f"  {label}:\n{rendered}")
        else:
            lines.append(f"  {label}: {rendered}")

    add("input", example.input)
    add("actual_output", example.actual_output)
    if example.expected_output is not None:
        add("expected_output", example.expected_output)
    for key, val in example.data.items():
        add(key, val)
    if example.reason:
        add("reason", example.reason)

    # Per-criterion labels only make sense for the verdict strategy.
    if strategy == "verdict" and example.criterion_verdicts:
        labelled = ", ".join(f"{i + 1}={lbl}" for i, lbl in enumerate(example.criterion_verdicts))
        lines.append(f"  criteria: {labelled}")

    return "\n".join(lines), truncated


_HEADER = (
    "EXAMPLES OF HUMAN JUDGEMENTS\n"
    "These examples show how strictly the team grades; they are reference only. "
    "Do not copy them, and do not score them in place of the DATA below."
)


def render_few_shot_block(
    examples: Optional[List[FewShotExample]],
    strategy: str = "verdict",
    max_chars: int = FEW_SHOT_DEFAULT_MAX_CHARS,
) -> Tuple[str, Dict[str, Any]]:
    """Render the EXAMPLES block and a log dict.

    Returns ``("", log)`` when there are no examples so the caller's prompt is
    unchanged. When non-empty the returned text is prefixed with ``\\n\\n`` and
    carries no trailing newline, so a caller can splice it as
    ``f"{preamble}{block}\\n\\nDATA:..."`` and keep the exact pre-few-shot
    spacing when the block is empty.

    log = {count, used, source_case_ids, truncated[, warning]}.
    """
    examples = examples or []
    count = len(examples)
    if count == 0:
        return "", {"count": 0, "used": 0, "source_case_ids": [], "truncated": False}

    rendered: List[str] = []
    source_ids: List[Optional[str]] = []
    any_field_truncated = False
    used = 0
    running_len = len(_HEADER)

    for i, ex in enumerate(examples):
        text, was_trunc = _render_one(ex, i + 1, strategy)
        # +2 for the "\n\n" separator between header/previous example and this one.
        projected = running_len + 2 + len(text)
        if used > 0 and projected > max_chars:
            break
        if used == 0 and (len(_HEADER) + 2 + len(text)) > max_chars:
            # Even the first example does not fit; drop everything.
            break
        rendered.append(text)
        source_ids.append(ex.source_case_id)
        any_field_truncated = any_field_truncated or was_trunc
        running_len = projected
        used += 1

    dropped = count - used
    if used == 0:
        return "", {
            "count": count,
            "used": 0,
            "source_case_ids": [],
            "truncated": True,
            "warning": (
                f"No few-shot example fit within few_shot_max_chars={max_chars}; "
                f"all {count} dropped."
            ),
        }

    block = "\n\n" + _HEADER + "\n\n" + "\n\n".join(rendered)
    log: Dict[str, Any] = {
        "count": count,
        "used": used,
        "source_case_ids": source_ids,
        "truncated": any_field_truncated or dropped > 0,
    }
    if dropped > 0:
        log["warning"] = (
            f"{dropped} of {count} few-shot examples dropped to stay within "
            f"few_shot_max_chars={max_chars}."
        )
    return block, log
