# calibration/analyst.py
"""
AI calibration analyst.

Reads every judge-vs-human disagreement, diagnoses each case in batches (map),
consolidates the diagnoses into findings (reduce), and proposes concrete config
edits (propose). The LLM only *diagnoses and phrases*; all counts come from
code, the threshold is proposed by ``threshold_sweep`` (not the LLM), and every
proposal is validated in code before it is returned.

The analyst sees only the cases it is given (train). Holdout is never passed in
and therefore never reaches a prompt — the optimizer relies on this.
"""

import asyncio
import hashlib
import json
import re
from typing import Any, Callable, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

from eval_lib.calibration.budget import Budget, BudgetExceeded, _BudgetedModel
from eval_lib.calibration.models import CalibrationSet, CustomEvalConfig, LabeledCase, Proposal
from eval_lib.calibration.stats import (
    _judge_verdict,
    criterion_attribution,
    inter_annotator,
    threshold_sweep,
)
from eval_lib.utils import extract_json_block

# Fixed diagnosis vocabulary (the LLM must choose from these).
CAUSES = {
    "criterion_too_strict",
    "criterion_too_lenient",
    "criterion_ambiguous",
    "missing_criterion",
    "irrelevant_criterion",
    "aggregation_too_lenient",
    "aggregation_too_strict",
    "data_missing",
    "judge_misread",
    "label_suspect",
}

_FIELD_MAX = 1500
_REDUCE_FANOUT = 8
_IA_KAPPA_FLOOR = 0.4
_SMALL_CLASS = 5
_BUILTIN_PLACEHOLDERS = {
    "input",
    "actual_output",
    "expected_output",
    "retrieval_context",
    "system_prompt",
}

_PLACEHOLDER_KINDS = {"rewrite_criterion", "add_criterion", "split_criterion"}
_INDEX_KINDS = {"rewrite_criterion", "remove_criterion", "split_criterion"}


# ── Result models ────────────────────────────────────────────────────────────
class Finding(BaseModel):
    schema_version: int = 1
    id: str
    title: str
    cause: str
    criterion_index: Optional[int] = None
    count: int = 0
    share: float = 0.0
    case_ids: List[str] = Field(default_factory=list)
    explanation: str = ""
    example_case_ids: List[str] = Field(default_factory=list)


class AnalysisResult(BaseModel):
    schema_version: int = 1
    findings: List[Finding] = Field(default_factory=list)
    proposals: List[Proposal] = Field(default_factory=list)
    rejected: List[Dict[str, Any]] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    cases_analyzed: int = 0
    sampled: bool = False
    llm_calls: int = 0
    cost_usd: float = 0.0
    stopped_by_budget: bool = False


def _hash01(seed: int, key: str) -> float:
    digest = hashlib.sha256(f"{seed}:{key}".encode("utf-8")).hexdigest()
    return int(digest, 16) / float(1 << 256)


class CalibrationAnalyst:
    def __init__(
        self,
        model: Any,
        batch_size: int = 25,
        max_cases: int = 300,
        contrast_ratio: float = 0.2,
        concurrency: int = 4,
        language: str = "en",
    ):
        self.model = model
        self.batch_size = max(1, batch_size)
        self.max_cases = max(1, max_cases)
        self.contrast_ratio = contrast_ratio
        self.concurrency = max(1, concurrency)
        self.language = language

    async def analyze(
        self,
        calibration_set: CalibrationSet,
        cases: Optional[List[str]] = None,
        budget: Optional[Budget] = None,
        on_progress: Optional[Callable[[str, int, int], None]] = None,
    ) -> AnalysisResult:
        budget = budget if budget is not None else Budget()
        budgeted = _BudgetedModel(self.model, budget)
        calls_before, cost_before = budget.llm_calls, budget.cost_usd
        config = calibration_set.config
        warnings: List[str] = []
        stopped = False

        id_set = set(cases) if cases is not None else None
        train = [c for c in calibration_set.cases if id_set is None or c.id in id_set]

        # Pre-flight warnings (cheap, no LLM).
        warnings.extend(self._pre_warnings(train))

        # 1. Selection (deterministic).
        selected, sampled, contrast_ids = self._select(train)
        if on_progress:
            on_progress("select", len(selected), len(selected))
        if not selected:
            return self._result(
                [], [], [], warnings, 0, False, budget, calls_before, cost_before, False
            )

        attribution = criterion_attribution(selected)
        context = self._map_context(config, attribution, contrast_ids)

        # 2. Map: diagnose in batches, bounded concurrency.
        batches = [
            selected[i : i + self.batch_size] for i in range(0, len(selected), self.batch_size)
        ]
        sem = asyncio.Semaphore(self.concurrency)
        done = 0

        async def run_batch(batch: List[LabeledCase]):
            nonlocal done, stopped
            async with sem:
                try:
                    prompt = context + "\n\n" + self._cases_block(batch)
                    data = await self._ask_json(prompt, _validate_map, budgeted)
                    out = data["diagnoses"]
                except BudgetExceeded:
                    stopped = True
                    out = None
                except Exception as e:  # a bad batch becomes a warning, never fatal
                    warnings.append(f"map batch failed: {type(e).__name__}: {e}")
                    out = []
            done += 1
            if on_progress:
                on_progress("map", done, len(batches))
            return out

        batch_results = await asyncio.gather(*[run_batch(b) for b in batches])
        diagnoses = self._collect_diagnoses(batch_results, selected)

        # 3. Reduce: group in code, LLM phrases findings (hierarchical if many groups).
        groups = self._group(diagnoses, selected)
        if on_progress:
            on_progress("reduce", 0, len(groups))
        try:
            findings = await self._reduce(groups, len(selected), budgeted)
        except BudgetExceeded:
            stopped = True
            findings = self._fallback_findings(groups, len(selected))
        if on_progress:
            on_progress("reduce", len(groups), len(groups))

        # 4. Propose (LLM) + threshold from sweep (code).
        proposals: List[Proposal] = []
        rejected: List[Dict[str, Any]] = []
        if not stopped and findings:
            try:
                raw = await self._propose(findings, config, budgeted)
            except BudgetExceeded:
                stopped = True
                raw = []
            except Exception as e:
                warnings.append(f"propose step failed: {type(e).__name__}: {e}")
                raw = []
            proposals, rejected = self._validate_proposals(raw, config, calibration_set, train)
        proposals += self._threshold_proposal(selected, config)
        if on_progress:
            on_progress("propose", len(proposals), len(proposals))

        return self._result(
            findings,
            proposals,
            rejected,
            warnings,
            len(selected),
            sampled,
            budget,
            calls_before,
            cost_before,
            stopped,
        )

    # ── pre-flight warnings ──────────────────────────────────────────────────
    def _pre_warnings(self, train: List[LabeledCase]) -> List[str]:
        w: List[str] = []
        ia = inter_annotator(train)
        if ia.cohen_kappa is not None and ia.cohen_kappa < _IA_KAPPA_FLOOR:
            w.append(
                f"inter-annotator κ={ia.cohen_kappa} < {_IA_KAPPA_FLOOR}: reviewers disagree — "
                "agree on a labelling standard before trusting calibration."
            )
        if ia.fleiss_kappa is not None and ia.fleiss_kappa < _IA_KAPPA_FLOOR:
            w.append(f"inter-annotator Fleiss κ={ia.fleiss_kappa} < {_IA_KAPPA_FLOOR}.")
        labelled = [c for c in train if _judge_verdict(c) is not None]
        if len(labelled) < 20:
            w.append(f"only {len(labelled)} usable labels (< 20): findings may be unreliable.")
        n_fail = sum(1 for c in train if c.label.verdict == "fail")
        n_pass = sum(1 for c in train if c.label.verdict == "pass")
        if n_fail < _SMALL_CLASS:
            w.append(f"only {n_fail} fail-labelled cases (< {_SMALL_CLASS}).")
        if n_pass < _SMALL_CLASS:
            w.append(f"only {n_pass} pass-labelled cases (< {_SMALL_CLASS}).")
        return w

    # ── selection ──────────────────────────────────────────────────────────────
    def _select(self, train: List[LabeledCase]) -> Tuple[List[LabeledCase], bool, List[str]]:
        included = [c for c in train if _judge_verdict(c) is not None]
        disagree = [c for c in included if _judge_verdict(c) != c.label.verdict]
        agree = [c for c in included if _judge_verdict(c) == c.label.verdict]

        sampled = False
        if len(disagree) > self.max_cases:
            disagree = self._stratified(disagree, self.max_cases)
            sampled = True

        n_contrast = int(round(self.contrast_ratio * len(agree)))
        contrast = sorted(agree, key=lambda c: _hash01(0, c.id))[:n_contrast]
        contrast_ids = [c.id for c in contrast]

        selected = sorted(disagree + contrast, key=lambda c: c.id)
        return selected, sampled, contrast_ids

    def _stratified(self, cases: List[LabeledCase], k: int) -> List[LabeledCase]:
        strata: Dict[Tuple[str, Optional[int]], List[LabeledCase]] = {}
        for c in cases:
            strata.setdefault(self._stratum_key(c), []).append(c)
        keys = sorted(strata.keys(), key=lambda t: (t[0], -1 if t[1] is None else t[1]))
        picked: List[LabeledCase] = []
        total = len(cases)
        for key in keys:
            members = sorted(strata[key], key=lambda c: _hash01(0, c.id))
            share = max(1, round(k * len(members) / total))
            picked.extend(members[:share])
        picked = sorted(picked, key=lambda c: _hash01(0, c.id))[:k]
        return picked

    def _stratum_key(self, case: LabeledCase) -> Tuple[str, Optional[int]]:
        et = self._error_type(case)
        return (et, self._dominant_criterion(case, et))

    @staticmethod
    def _error_type(case: LabeledCase) -> str:
        j, h = _judge_verdict(case), case.label.verdict
        if h == "fail" and j == "pass":
            return "false_pass"
        if h == "pass" and j == "fail":
            return "false_fail"
        return "agree"

    @staticmethod
    def _dominant_criterion(case: LabeledCase, error_type: str) -> Optional[int]:
        log = (case.judge_result or {}).get("evaluation_log") or {}
        raw = log.get("verdict_weights")
        if not isinstance(raw, dict) or not raw:
            return None
        items = [(int(k), float(v)) for k, v in raw.items()]
        if not items:
            return None
        # false_fail → the lowest-weight criterion dragged it down;
        # false_pass → the highest-weight criterion propped it up.
        if error_type == "false_fail":
            return min(items, key=lambda kv: (kv[1], kv[0]))[0]
        return max(items, key=lambda kv: (kv[1], -kv[0]))[0]

    # ── prompts ──────────────────────────────────────────────────────────────
    def _map_context(self, config: CustomEvalConfig, attribution, contrast_ids: List[str]) -> str:
        crit = "\n".join(f"{i}. {c}" for i, c in enumerate(config.evaluation_criteria))
        stats_lines = []
        for cs in attribution:
            stats_lines.append(
                f"- criterion {cs.index}: skip_rate={cs.skip_rate}, "
                f"false_fail_culprit={len(cs.false_fail_culprit_case_ids)}, "
                f"fully_on_false_pass={len(cs.fully_on_false_pass_case_ids)}"
            )
        stats = "\n".join(stats_lines) if stats_lines else "(no per-criterion stats)"
        causes = ", ".join(sorted(CAUSES))
        return (
            "You are calibrating an LLM judge against human labels. For each case below, say WHY the "
            "judge disagreed (or agreed) with the human, choosing exactly one cause.\n\n"
            f"CONFIG: strategy={config.strategy}, temperature={config.temperature}, "
            f"threshold={config.threshold}\n"
            f"CRITERIA (0-indexed):\n{crit}\n\n"
            f"PER-CRITERION STATS:\n{stats}\n\n"
            f"CAUSES (choose one per case): {causes}\n"
            "Cases whose id is marked (contrast) agreed with the human — use them to calibrate, not to fix.\n"
            f"Write every note in language: {self.language}.\n"
            'Return ONLY JSON: {"diagnoses": [{"case_id": "<id>", "cause": "<one cause>", '
            '"criterion_index": <int or null>, "note": "<short>"}]} with one entry per case.'
        )

    def _cases_block(self, batch: List[LabeledCase]) -> str:
        blocks = []
        for c in batch:
            log = (c.judge_result or {}).get("evaluation_log") or {}
            tc = c.test_case
            data_lines = [
                f"    input: {_trunc(tc.input)}",
                f"    actual_output: {_trunc(tc.actual_output)}",
            ]
            if tc.expected_output:
                data_lines.append(f"    expected_output: {_trunc(tc.expected_output)}")
            for k, v in (tc.extra_fields or {}).items():
                data_lines.append(f"    {k}: {_trunc(v)}")
            kept = log.get("kept_criteria") or []
            verdicts = log.get("verdicts") or []
            vc_lines = []
            for i, crit in enumerate(kept):
                vd = verdicts[i] if i < len(verdicts) else {}
                vc_lines.append(
                    f"    [{crit}] -> {vd.get('verdict', '?')} ({_trunc(vd.get('reason', ''), 200)})"
                )
            jr = c.judge_result or {}
            blocks.append(
                f"Case {c.id}:\n"
                + "\n".join(data_lines)
                + f"\n  judge: score={jr.get('score')}, verdict={_judge_verdict(c)}\n"
                + ("  judge_criteria:\n" + "\n".join(vc_lines) + "\n" if vc_lines else "")
                + f"  human: verdict={c.label.verdict}, reason={_trunc(c.label.reason or '', 300)}"
            )
        return "CASES:\n" + "\n\n".join(blocks)

    async def _reduce(self, groups: List[dict], n_cases: int, budgeted) -> List[Finding]:
        if not groups:
            return []
        chunks = [groups[i : i + _REDUCE_FANOUT] for i in range(0, len(groups), _REDUCE_FANOUT)]
        findings: List[Finding] = []
        fid = 1
        for chunk in chunks:
            summary = "\n".join(
                f'g{g["gid"]}: cause={g["cause"]}, criterion_index={g["criterion_index"]}, '
                f'count={g["count"]}, example_notes={g["notes"][:3]}'
                for g in chunk
            )
            prompt = (
                "Consolidate these diagnosis groups into findings. A finding may cover one or more "
                "groups. Do NOT invent counts.\n"
                f"Write titles and explanations in language: {self.language}.\n\n"
                f"GROUPS:\n{summary}\n\n"
                'Return ONLY JSON: {"findings": [{"group_ids": ["g0", ...], "title": "...", '
                '"explanation": "..."}]}'
            )
            try:
                data = await self._ask_json(prompt, _validate_reduce, budgeted)
                findings += self._findings_from_llm(data["findings"], chunk, n_cases, fid)
            except BudgetExceeded:
                raise
            except Exception:
                findings += self._fallback_findings(chunk, n_cases, fid)
            fid = len(findings) + 1
        return findings

    async def _propose(
        self, findings: List[Finding], config: CustomEvalConfig, budgeted
    ) -> List[dict]:
        crit = "\n".join(f"{i}. {c}" for i, c in enumerate(config.evaluation_criteria))
        fblock = "\n".join(
            f"{f.id}: cause={f.cause}, criterion_index={f.criterion_index}, "
            f"count={f.count}, title={f.title}"
            for f in findings
        )
        prompt = (
            "Propose concrete edits to make the judge match the humans. Keep every criterion wording "
            "in the SAME language as the existing criteria. Each new/rewritten criterion MUST reference "
            "data via {{placeholders}}. Do NOT propose a threshold (it is tuned separately).\n\n"
            f"CRITERIA (0-indexed):\n{crit}\n\nFINDINGS:\n{fblock}\n\n"
            'Return ONLY JSON: {"proposals": [{"kind": "rewrite_criterion|add_criterion|remove_criterion'
            '|split_criterion|set_temperature|set_strategy|add_few_shot|review_labels", '
            '"criterion_index": <int or null>, "text": "<criterion or null>", "texts": [<for split>], '
            '"temperature": <float or null>, "strategy": "<verdict|direct or null>", '
            '"case_ids": [<ids or null>], "rationale": "...", "finding_ids": ["F1", ...]}]}'
        )
        data = await self._ask_json(prompt, _validate_propose, budgeted)
        return list(data["proposals"])

    # ── reduce / grouping helpers ──────────────────────────────────────────────
    def _group(self, diagnoses: List[dict], selected: List[LabeledCase]) -> List[dict]:
        by_key: Dict[Tuple[str, Optional[int]], dict] = {}
        for d in diagnoses:
            key = (d["cause"], d.get("criterion_index"))
            g = by_key.setdefault(
                key, {"cause": key[0], "criterion_index": key[1], "case_ids": [], "notes": []}
            )
            g["case_ids"].append(d["case_id"])
            if d.get("note"):
                g["notes"].append(d["note"])
        groups = list(by_key.values())
        # deterministic order: biggest first, then cause, then criterion index
        groups.sort(
            key=lambda g: (
                -len(g["case_ids"]),
                g["cause"],
                -1 if g["criterion_index"] is None else g["criterion_index"],
            )
        )
        for i, g in enumerate(groups):
            g["gid"] = i
            g["count"] = len(g["case_ids"])
        return groups

    def _findings_from_llm(
        self, raw: List[dict], chunk: List[dict], n_cases: int, start_id: int
    ) -> List[Finding]:
        by_gid = {f"g{g['gid']}": g for g in chunk}
        findings: List[Finding] = []
        covered = set()
        fid = start_id
        for item in raw:
            gids = [gid for gid in (item.get("group_ids") or []) if gid in by_gid]
            if not gids:
                continue
            gs = [by_gid[gid] for gid in gids]
            covered.update(gids)
            findings.append(
                self._make_finding(
                    fid,
                    gs,
                    n_cases,
                    title=str(item.get("title", "")).strip(),
                    explanation=str(item.get("explanation", "")).strip(),
                )
            )
            fid += 1
        # any group the LLM ignored still becomes a finding (no data lost).
        for g in chunk:
            if f"g{g['gid']}" not in covered:
                findings.append(self._make_finding(fid, [g], n_cases))
                fid += 1
        return findings

    def _fallback_findings(
        self, groups: List[dict], n_cases: int, start_id: int = 1
    ) -> List[Finding]:
        return [self._make_finding(start_id + i, [g], n_cases) for i, g in enumerate(groups)]

    @staticmethod
    def _make_finding(
        fid: int, groups: List[dict], n_cases: int, title: str = "", explanation: str = ""
    ) -> Finding:
        case_ids: List[str] = []
        for g in groups:
            case_ids.extend(g["case_ids"])
        # preserve order, dedupe
        seen: Dict[str, None] = {}
        for cid in case_ids:
            seen.setdefault(cid, None)
        case_ids = list(seen.keys())
        primary = max(groups, key=lambda g: g["count"])
        count = len(case_ids)
        return Finding(
            id=f"F{fid}",
            title=title or f"{primary['cause']} on criterion {primary['criterion_index']}",
            cause=primary["cause"],
            criterion_index=primary["criterion_index"],
            count=count,
            share=round(count / n_cases, 4) if n_cases else 0.0,
            case_ids=case_ids,
            explanation=explanation,
            example_case_ids=case_ids[:5],
        )

    # ── proposal validation ────────────────────────────────────────────────────
    def _validate_proposals(
        self,
        raw: List[dict],
        config: CustomEvalConfig,
        cs: CalibrationSet,
        train: List[LabeledCase],
    ) -> Tuple[List[Proposal], List[Dict[str, Any]]]:
        available = self._available_placeholders(cs, train)
        train_ids = {c.id for c in train}
        accepted: List[Proposal] = []
        rejected: List[Dict[str, Any]] = []
        n_criteria = len(config.evaluation_criteria)
        existing = set(config.evaluation_criteria)
        pid = 1
        skip_fields = {"id", "schema_version"}
        for item in raw:
            kind = item.get("kind")
            try:
                kwargs = {
                    k: v
                    for k, v in item.items()
                    if k in Proposal.model_fields and k not in skip_fields
                }
                p = Proposal(id=f"P{pid}", **kwargs)
            except Exception as e:
                rejected.append({"proposal": item, "reason": f"malformed: {e}"})
                continue
            if kind == "set_threshold":
                rejected.append(
                    {
                        "proposal": item,
                        "reason": "threshold is tuned by threshold_sweep, not the LLM",
                    }
                )
                continue
            reason = self._reject_reason(p, available, existing, n_criteria, train_ids)
            if reason:
                rejected.append({"proposal": p.model_dump(mode="json"), "reason": reason})
                continue
            # track growth / dedupe across accepted proposals (net criteria delta)
            for t in self._new_texts(p):
                existing.add(t)
            if p.kind == "add_criterion":
                n_criteria += 1
            elif p.kind == "split_criterion":
                n_criteria += max(0, len(p.texts or []) - 1)
            elif p.kind == "remove_criterion":
                n_criteria -= 1
            accepted.append(p)
            pid += 1
        return accepted, rejected

    def _reject_reason(
        self, p: Proposal, available, existing, n_criteria, train_ids
    ) -> Optional[str]:
        if p.kind in _INDEX_KINDS and (p.criterion_index is None):
            return "missing criterion_index"
        for text in self._new_texts(p):
            ph = set(_placeholders(text))
            if not ph:
                return f"criterion has no {{placeholder}}: {text!r}"
            missing = ph - available
            if missing:
                return f"criterion references unknown placeholders {sorted(missing)}"
            if text in existing:
                return f"duplicate criterion: {text!r}"
        added = len(self._new_texts(p)) - (1 if p.kind == "split_criterion" else 0)
        if p.kind in ("add_criterion", "split_criterion") and n_criteria + added > 50:
            return "would exceed the 50-criteria limit"
        if p.kind == "add_few_shot":
            unknown = [cid for cid in (p.case_ids or []) if cid not in train_ids]
            if unknown:
                return f"add_few_shot references non-train cases {unknown}"
        return None

    @staticmethod
    def _new_texts(p: Proposal) -> List[str]:
        if p.kind in ("rewrite_criterion", "add_criterion"):
            return [p.text] if p.text else []
        if p.kind == "split_criterion":
            return list(p.texts or [])
        return []

    def _available_placeholders(self, cs: CalibrationSet, train: List[LabeledCase]) -> set:
        avail = set(cs.placeholders or set()) | set(_BUILTIN_PLACEHOLDERS)
        for c in cs.config.evaluation_criteria:
            avail.update(_placeholders(c))
        for c in train:
            if c.test_case.extra_fields:
                avail.update(c.test_case.extra_fields.keys())
        return avail

    def _threshold_proposal(
        self, selected: List[LabeledCase], config: CustomEvalConfig
    ) -> List[Proposal]:
        try:
            sweep = threshold_sweep(selected, objective="kappa")
        except Exception:
            return []
        if sweep.best is None:
            return []
        if abs(sweep.best.threshold - config.threshold) < 1e-6:
            return []
        return [
            Proposal(
                id="P-threshold",
                kind="set_threshold",
                threshold=sweep.best.threshold,
                rationale=f"threshold_sweep: κ={sweep.best.kappa} at {sweep.best.threshold} "
                f"vs current {config.threshold}.",
            )
        ]

    # ── LLM helper ─────────────────────────────────────────────────────────────
    async def _ask_json(self, prompt: str, validate: Callable[[Any], None], budgeted) -> Any:
        last: Optional[Exception] = None
        for attempt in range(2):
            msg = prompt
            if attempt == 1:
                msg = (
                    prompt
                    + "\n\nYour previous reply was not valid JSON for this task. Reply with ONLY the JSON."
                )
            text, _ = await budgeted.chat_complete(
                [{"role": "user", "content": msg}], temperature=0.0
            )
            try:
                data = json.loads(extract_json_block(text))
                validate(data)
                return data
            except Exception as e:  # noqa: BLE001
                last = e
        raise ValueError(f"invalid JSON after one retry: {last}")

    def _collect_diagnoses(self, batch_results, selected: List[LabeledCase]) -> List[dict]:
        valid_ids = {c.id for c in selected}
        out: List[dict] = []
        for res in batch_results:
            if not res:
                continue
            for d in res:
                if not isinstance(d, dict):
                    continue
                cid = d.get("case_id")
                cause = d.get("cause")
                if cid not in valid_ids:
                    continue
                if cause not in CAUSES:
                    cause = "judge_misread"
                ci = d.get("criterion_index")
                out.append(
                    {
                        "case_id": cid,
                        "cause": cause,
                        "criterion_index": ci if isinstance(ci, int) else None,
                        "note": str(d.get("note", "")).strip(),
                    }
                )
        # deterministic order by case id
        out.sort(key=lambda d: d["case_id"])
        return out

    def _result(
        self, findings, proposals, rejected, warnings, n, sampled, budget, calls0, cost0, stopped
    ) -> AnalysisResult:
        return AnalysisResult(
            findings=findings,
            proposals=proposals,
            rejected=rejected,
            warnings=warnings,
            cases_analyzed=n,
            sampled=sampled,
            llm_calls=budget.llm_calls - calls0,
            cost_usd=round(budget.cost_usd - cost0, 6),
            stopped_by_budget=stopped,
        )


# ── module helpers ───────────────────────────────────────────────────────────
_PLACEHOLDER_RE = re.compile(r"\{\{(\w+)\}\}")


def _placeholders(text: str) -> List[str]:
    return _PLACEHOLDER_RE.findall(text or "")


def _trunc(value: Any, limit: int = _FIELD_MAX) -> str:
    text = value if isinstance(value, str) else str(value)
    return text if len(text) <= limit else text[:limit] + " …[truncated]"


def _validate_map(data: Any) -> None:
    if not isinstance(data, dict) or not isinstance(data.get("diagnoses"), list):
        raise ValueError("expected {'diagnoses': [...]}")


def _validate_reduce(data: Any) -> None:
    if not isinstance(data, dict) or not isinstance(data.get("findings"), list):
        raise ValueError("expected {'findings': [...]}")


def _validate_propose(data: Any) -> None:
    if not isinstance(data, dict) or not isinstance(data.get("proposals"), list):
        raise ValueError("expected {'proposals': [...]}")
