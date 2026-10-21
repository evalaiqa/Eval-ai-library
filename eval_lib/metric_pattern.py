# metric_pattern.py
"""
Base classes for evaluation metrics with beautiful console logging.
"""
import functools
import json
import time
from typing import Type, Dict, Any, Union, Optional

from eval_lib.testcases_schema import EvalTestCase, ConversationalEvalTestCase
from eval_lib.llm_client import chat_complete
from eval_lib.few_shot import (
    FEW_SHOT_DEFAULT_MAX_CHARS,
    normalize_examples,
    render_few_shot_block,
)


def _is_empty_output(value: Any) -> bool:
    """Return True for None, empty strings and whitespace-only strings."""
    if value is None:
        return True
    if isinstance(value, str) and not value.strip():
        return True
    return False


def _wrap_evaluate_with_empty_guard(original):
    """Wrap a metric's async evaluate() with a guard for empty actual_output.

    When the wrapped metric has `requires_actual_output = True` (the default)
    and `test_case.actual_output` is empty/whitespace, the wrapper returns the
    canonical zero-score result via `_empty_actual_output_result` instead of
    delegating to the original evaluate — so the LLM is never given an empty
    response to score. See issue #1.
    """
    @functools.wraps(original)
    async def wrapper(self, test_case, *args, **kwargs):
        if getattr(self, "requires_actual_output", True):
            actual = getattr(test_case, "actual_output", None)
            if _is_empty_output(actual):
                result = self._empty_actual_output_result(test_case)
                self.print_result(result)
                return result
        return await original(self, test_case, *args, **kwargs)

    wrapper._empty_guard_wrapped = True
    return wrapper


class Colors:
    """ANSI color codes for beautiful console output"""
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    DIM = '\033[2m'


class MetricPattern:
    """
    Base class for metrics that use a pattern-based approach to evaluation.
    This class is designed to be subclassed for specific metrics.

    Subclasses set `requires_actual_output = True` to opt into a guard that
    short-circuits evaluation to a zero score when `test_case.actual_output`
    is empty or whitespace-only. This prevents the "LLM hallucinates relevance
    from an empty answer" failure mode (issue #1) for RAG-style metrics.

    The default is False because most non-RAG metric families have their own
    semantics for empty output (security detection metrics judge the input;
    resistance metrics treat an empty output as a successful refusal;
    deterministic checks like NonEmptyMetric *want* to fail on empty answers
    with their own message). Opt in only where the LLM would otherwise be
    handed an empty answer to score.
    """
    name: str  # name of the metric
    requires_actual_output: bool = False

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        # Wrap the subclass's evaluate() so every metric gets the empty-output
        # guard for free, without each subclass having to remember to add it.
        # If a subclass doesn't define its own evaluate() we leave it alone.
        if "evaluate" in cls.__dict__:
            original = cls.__dict__["evaluate"]
            if not getattr(original, "_empty_guard_wrapped", False):
                cls.evaluate = _wrap_evaluate_with_empty_guard(original)

    def __init__(
        self,
        model: str,
        threshold: float,
        verbose: bool = False,
        *,
        few_shot_examples: Optional[list] = None,
        few_shot_max_chars: int = FEW_SHOT_DEFAULT_MAX_CHARS,
    ):
        self.model = model
        self.threshold = threshold
        self.verbose = verbose
        # Few-shot calibration examples. Opt-in and reusable by any metric:
        # an empty list means no behaviour change (prompts stay identical).
        # The count limit is enforced here (ValueError) at construction time.
        self.few_shot_examples = normalize_examples(few_shot_examples)
        self.few_shot_max_chars = few_shot_max_chars

    def _render_few_shot(self, strategy: str = "verdict") -> "tuple[str, Dict[str, Any]]":
        """Render this metric's few-shot EXAMPLES block (+ log).

        Returns ``("", log)`` when no examples are configured, so a prompt that
        splices the block keeps its exact pre-few-shot spacing. Reusable from
        any subclass that wants to show calibration examples to its judge.
        """
        return render_few_shot_block(
            self.few_shot_examples, strategy, self.few_shot_max_chars
        )

    def _empty_actual_output_result(self, test_case: "EvalTestCase") -> Dict[str, Any]:
        """Canonical zero-score result returned when actual_output is empty.

        Centralised here so the shape stays consistent with what each metric
        normally returns (same top-level keys + a marker in evaluation_log).
        """
        reason = (
            "Assistant returned an empty response, so the metric cannot be "
            "evaluated; defaulting to score 0."
        )
        result = {
            "name": self.name,
            "score": 0.0,
            "success": False,
            "reason": reason,
            "evaluation_cost": 0.0,
            "evaluation_log": {
                "input_question": getattr(test_case, "input", None),
                "answer": getattr(test_case, "actual_output", ""),
                "skipped": True,
                "skip_reason": "empty_actual_output",
                "final_score": 0.0,
                "threshold": self.threshold,
                "success": False,
                "final_reason": reason,
            },
        }
        return result

    def _log(self, message: str, color: str = Colors.CYAN):
        """Log message with color if verbose mode is enabled"""
        if self.verbose:
            print(f"{color}{message}{Colors.ENDC}")

    def _log_step(self, step_name: str, step_num: int = None):
        """Log evaluation step"""
        if self.verbose:
            prefix = f"[{step_num}] " if step_num else ""
            print(f"{Colors.DIM}  {prefix}{step_name}...{Colors.ENDC}")

    def print_result(self, result: Dict[str, Any]):
        """
        Print evaluation result based on verbose setting.
        If verbose=False: simple dict print
        If verbose=True: beautiful formatted output with colors
        """
        if not self.verbose:
            print(result)
            return

        import shutil
        import textwrap
        import re
        import json

        # Получаем ширину терминала и делим пополам
        terminal_width = shutil.get_terminal_size().columns
        WIDTH = terminal_width // 2
        WIDTH = max(WIDTH, 60)  # Минимум 60 символов

        # Функция для переноса длинного текста
        def wrap_text(text, width, indent=0):
            """Переносит текст на несколько строк с отступом"""
            wrapper = textwrap.TextWrapper(
                width=width - indent,
                initial_indent=' ' * indent,
                subsequent_indent=' ' * indent,
                break_long_words=True,
                break_on_hyphens=False
            )
            return wrapper.fill(text)

        success = result.get('success', False)
        score = result.get('score', 0.0)
        reason = result.get('reason', 'N/A')
        cost = result.get('evaluation_cost', 0.0)
        evaluation_log = result.get('evaluation_log', None)

        status_icon = "✅" if success else "❌"
        status_color = Colors.GREEN if success else Colors.RED
        status_text = "PASSED" if success else "FAILED"

        bar_length = min(30, WIDTH - 30)  # Адаптивная длина прогресс-бара
        filled = int(bar_length * score)
        bar = '█' * filled + '░' * (bar_length - filled)

        metric_name = result.get('name', self.name)
        formatted_name = f"📊 {metric_name}"

        # Центрируем заголовок
        name_len = len(formatted_name)
        if name_len > WIDTH:
            formatted_name = formatted_name[:WIDTH-3] + "..."
            centered_name = formatted_name
        else:
            padding = WIDTH - name_len
            left_pad = padding // 2
            right_pad = padding - left_pad
            centered_name = " " * left_pad + formatted_name + " " * right_pad

        # Рамка заголовка
        border = "═" * WIDTH

        print(f"\n{Colors.BOLD}{Colors.CYAN}╔{border}╗{Colors.ENDC}")
        print(f"{Colors.BOLD}{Colors.CYAN}║{Colors.ENDC}{centered_name}{Colors.BOLD}{Colors.CYAN}║{Colors.ENDC}")
        print(f"{Colors.BOLD}{Colors.CYAN}╚{border}╝{Colors.ENDC}\n")

        print(f"{Colors.BOLD}Status:{Colors.ENDC} {status_icon} {status_color}{Colors.BOLD}{status_text}{Colors.ENDC}")
        print(
            f"{Colors.BOLD}Score:{Colors.ENDC}  {Colors.YELLOW}{score:.2f}{Colors.ENDC} [{bar}] {score*100:.0f}%")
        print(
            f"{Colors.BOLD}Cost:{Colors.ENDC}   {Colors.BLUE}💰 ${cost:.6f}{Colors.ENDC}")

        # Переносим Reason на несколько строк если нужно
        print(f"{Colors.BOLD}Reason:{Colors.ENDC}")
        wrapped_reason = wrap_text(reason, WIDTH, indent=2)
        print(f"{Colors.DIM}{wrapped_reason}{Colors.ENDC}\n")

        if evaluation_log:
            log_json = json.dumps(evaluation_log, indent=2, ensure_ascii=False)
            log_lines = log_json.split('\n')

            print(f"{Colors.BOLD}Evaluation Log:{Colors.ENDC}")
            log_border = "─" * WIDTH
            print(f"{Colors.DIM}╭{log_border}╮{Colors.ENDC}")

            for line in log_lines:
                # Если строка длиннее WIDTH, переносим
                if len(line) > WIDTH - 4:
                    # Разбиваем длинную строку
                    wrapped_lines = textwrap.wrap(line, width=WIDTH - 4,
                                                  break_long_words=True,
                                                  break_on_hyphens=False)
                    for wrapped_line in wrapped_lines:
                        spaces_needed = WIDTH - len(wrapped_line) - 2
                        print(
                            f"{Colors.DIM}│{Colors.ENDC} {wrapped_line}{' ' * spaces_needed}{Colors.DIM}│{Colors.ENDC}")
                else:
                    spaces_needed = WIDTH - len(line) - 2
                    print(
                        f"{Colors.DIM}│{Colors.ENDC} {line}{' ' * spaces_needed}{Colors.DIM}│{Colors.ENDC}")

            print(f"{Colors.DIM}╰{log_border}╯{Colors.ENDC}")

        print(f"{Colors.DIM}{'─' * WIDTH}{Colors.ENDC}\n")


class ConversationalMetricPattern:
    """
    Base class for conversational metrics (evaluating full dialogues).
    Used for metrics like RoleAdherence, DialogueCoherence, etc.
    """
    name: str

    def __init__(self, model: str, threshold: float, verbose: bool = False):
        self.model = model
        self.threshold = threshold
        self.verbose = verbose
        self.chatbot_role: Optional[str] = None

    def _log(self, message: str, color: str = Colors.CYAN):
        """Log message with color if verbose mode is enabled"""
        if self.verbose:
            print(f"{color}{message}{Colors.ENDC}")

    def _log_step(self, step_name: str, step_num: int = None):
        """Log evaluation step"""
        if self.verbose:
            prefix = f"[{step_num}] " if step_num else ""
            print(f"{Colors.DIM}  {prefix}{step_name}...{Colors.ENDC}")

    def print_result(self, result: Dict[str, Any]):
        """
        Print evaluation result based on verbose setting.
        If verbose=False: simple dict print
        If verbose=True: beautiful formatted output with colors
        """
        if not self.verbose:
            print(result)
            return

        import shutil
        import textwrap
        import re
        import json

        # Получаем ширину терминала и делим пополам
        terminal_width = shutil.get_terminal_size().columns
        WIDTH = terminal_width // 2
        WIDTH = max(WIDTH, 60)  # Минимум 60 символов

        # Функция для переноса длинного текста
        def wrap_text(text, width, indent=0):
            """Переносит текст на несколько строк с отступом"""
            wrapper = textwrap.TextWrapper(
                width=width - indent,
                initial_indent=' ' * indent,
                subsequent_indent=' ' * indent,
                break_long_words=True,
                break_on_hyphens=False
            )
            return wrapper.fill(text)

        success = result.get('success', False)
        score = result.get('score', 0.0)
        reason = result.get('reason', 'N/A')
        cost = result.get('evaluation_cost', 0.0)
        evaluation_log = result.get('evaluation_log', None)

        status_icon = "✅" if success else "❌"
        status_color = Colors.GREEN if success else Colors.RED
        status_text = "PASSED" if success else "FAILED"

        bar_length = min(30, WIDTH - 30)  # Адаптивная длина прогресс-бара
        filled = int(bar_length * score)
        bar = '█' * filled + '░' * (bar_length - filled)

        metric_name = result.get('name', self.name)
        formatted_name = f"📊 {metric_name}"

        # Центрируем заголовок
        name_len = len(formatted_name)
        if name_len > WIDTH:
            formatted_name = formatted_name[:WIDTH-3] + "..."
            centered_name = formatted_name
        else:
            padding = WIDTH - name_len
            left_pad = padding // 2
            right_pad = padding - left_pad
            centered_name = " " * left_pad + formatted_name + " " * right_pad

        # Рамка заголовка
        border = "═" * WIDTH

        print(f"\n{Colors.BOLD}{Colors.CYAN}╔{border}╗{Colors.ENDC}")
        print(f"{Colors.BOLD}{Colors.CYAN}║{Colors.ENDC}{centered_name}{Colors.BOLD}{Colors.CYAN}║{Colors.ENDC}")
        print(f"{Colors.BOLD}{Colors.CYAN}╚{border}╝{Colors.ENDC}\n")

        print(f"{Colors.BOLD}Status:{Colors.ENDC} {status_icon} {status_color}{Colors.BOLD}{status_text}{Colors.ENDC}")
        print(
            f"{Colors.BOLD}Score:{Colors.ENDC}  {Colors.YELLOW}{score:.2f}{Colors.ENDC} [{bar}] {score*100:.0f}%")
        print(
            f"{Colors.BOLD}Cost:{Colors.ENDC}   {Colors.BLUE}💰 ${cost:.6f}{Colors.ENDC}")

        # Переносим Reason на несколько строк если нужно
        print(f"{Colors.BOLD}Reason:{Colors.ENDC}")
        wrapped_reason = wrap_text(reason, WIDTH, indent=2)
        print(f"{Colors.DIM}{wrapped_reason}{Colors.ENDC}\n")

        if evaluation_log:
            log_json = json.dumps(evaluation_log, indent=2, ensure_ascii=False)
            log_lines = log_json.split('\n')

            print(f"{Colors.BOLD}Evaluation Log:{Colors.ENDC}")
            log_border = "─" * WIDTH
            print(f"{Colors.DIM}╭{log_border}╮{Colors.ENDC}")

            for line in log_lines:
                # Если строка длиннее WIDTH, переносим
                if len(line) > WIDTH - 4:
                    # Разбиваем длинную строку
                    wrapped_lines = textwrap.wrap(line, width=WIDTH - 4,
                                                  break_long_words=True,
                                                  break_on_hyphens=False)
                    for wrapped_line in wrapped_lines:
                        spaces_needed = WIDTH - len(wrapped_line) - 2
                        print(
                            f"{Colors.DIM}│{Colors.ENDC} {wrapped_line}{' ' * spaces_needed}{Colors.DIM}│{Colors.ENDC}")
                else:
                    spaces_needed = WIDTH - len(line) - 2
                    print(
                        f"{Colors.DIM}│{Colors.ENDC} {line}{' ' * spaces_needed}{Colors.DIM}│{Colors.ENDC}")

            print(f"{Colors.DIM}╰{log_border}╯{Colors.ENDC}")

        print(f"{Colors.DIM}{'─' * WIDTH}{Colors.ENDC}\n")
