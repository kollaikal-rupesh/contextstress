"""
Evaluation harness for ContextStress.

Handles exact-match scoring, reasoning chain validation,
confidence calibration tracking, and result aggregation.
"""

import re
from dataclasses import dataclass, field
from typing import Optional

from contextstress.tasks.generator import TaskInstance
from contextstress.adapters import ModelResponse


@dataclass
class StepResult:
    """Result for a single reasoning step."""
    step: int
    expected: str
    found_in_response: bool


@dataclass
class TrialResult:
    """Result for a single evaluation trial."""
    task_id: str
    family: str
    depth: int
    cu: float
    snr: float
    seed: int
    correct: bool
    answer_given: str
    answer_expected: str
    confidence: Optional[float]
    calibration_error: Optional[float]
    step_results: list[StepResult]
    latency_ms: float


class Evaluator:
    """Evaluates model responses against ContextStress tasks."""

    def evaluate_trial(
        self,
        task: TaskInstance,
        response: ModelResponse,
        cu: float,
        snr: float,
        seed: int,
    ) -> TrialResult:
        """Evaluate a single trial."""
        correct = self._exact_match(response.answer, task.answer, task.aliases)
        step_results = self._validate_chain(task, response.raw_text)

        cal_error = None
        if response.confidence is not None:
            cal_error = abs(response.confidence - (100.0 if correct else 0.0))

        return TrialResult(
            task_id=task.id,
            family=task.family,
            depth=task.depth,
            cu=cu,
            snr=snr,
            seed=seed,
            correct=correct,
            answer_given=response.answer,
            answer_expected=task.answer,
            confidence=response.confidence,
            calibration_error=cal_error,
            step_results=step_results,
            latency_ms=response.latency_ms,
        )

    def _exact_match(
        self, given: str, expected: str, aliases: list[str]
    ) -> bool:
        """Case-insensitive match with alias resolution.

        Checks both exact match and containment — models often embed
        the answer in a full sentence rather than returning it bare.

        For containment matches, also checks that the expected answer
        appears as the *conclusion* (last mentioned) when multiple
        entity names are present, to avoid false positives in
        comparative tasks (T4) where all candidates are listed.
        """
        given_norm = self._normalize(given)
        candidates = [expected] + aliases
        for c in candidates:
            c_norm = self._normalize(c)
            # Exact match
            if given_norm == c_norm:
                return True
            # Containment: expected answer appears in the response
            if c_norm in given_norm:
                # Check if this is likely a multi-entity listing where
                # the expected answer might not be the concluded one.
                # Use the last occurrence position as a heuristic:
                # the concluded answer should appear at or near the end.
                # If the answer starts the response, it's clearly the answer
                if given_norm.startswith(c_norm):
                    return True
                last_pos = given_norm.rfind(c_norm)
                total_len = len(given_norm)
                # If the answer is in the last 60% of the text, accept it
                if total_len == 0 or last_pos >= total_len * 0.4:
                    return True
                # Even if early, accept if it's the ONLY candidate-like
                # match (i.e., no other long proper-noun-like words after it)
                remainder = given_norm[last_pos + len(c_norm):]
                # Accept if remainder is short (just punctuation/filler)
                if len(remainder.strip()) < 30:
                    return True
                # Otherwise, reject — the expected answer was mentioned
                # early in a list but something else was concluded.
                continue
            # Containment: response appears in expected (short answer in long)
            if given_norm in c_norm:
                return True
        return False

    @staticmethod
    def _normalize(text: str) -> str:
        """Normalize text for comparison."""
        text = text.lower().strip()
        text = re.sub(r"[^\w\s]", "", text)
        text = re.sub(r"\s+", " ", text)
        return text

    def _validate_chain(
        self, task: TaskInstance, raw_response: str
    ) -> list[StepResult]:
        """Validate each step of the reasoning chain."""
        results = []
        response_lower = raw_response.lower()

        for i, chain_step in enumerate(task.reasoning_chain):
            # Extract key entities from the chain step
            # Look for them in the response
            key_terms = self._extract_key_terms(chain_step)
            found = any(term.lower() in response_lower for term in key_terms)

            results.append(StepResult(
                step=i + 1,
                expected=chain_step,
                found_in_response=found,
            ))

        return results

    @staticmethod
    def _extract_key_terms(chain_step: str) -> list[str]:
        """Extract key terms from a reasoning chain step for validation."""
        # Remove common prefixes like "Step 1:", "→ Answer:"
        cleaned = re.sub(r"Step \d+:", "", chain_step)
        cleaned = re.sub(r"→.*$", "", cleaned)
        cleaned = re.sub(r"\(from passage\)", "", cleaned)

        # Extract capitalized multi-word terms (likely entity names)
        terms = re.findall(r"[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*", cleaned)

        # Also extract quoted terms
        quoted = re.findall(r"'([^']+)'", cleaned)
        terms.extend(quoted)

        return [t for t in terms if len(t) > 2]


@dataclass
class ConditionResult:
    """Aggregated results for one experimental condition."""
    family: str
    depth: int
    cu: float
    snr: float
    n_trials: int
    accuracy: float
    accuracy_std: float
    mean_confidence: Optional[float]
    mean_calibration_error: Optional[float]
    per_step_error_rates: list[float]

    @classmethod
    def from_trials(cls, trials: list[TrialResult]) -> "ConditionResult":
        if not trials:
            raise ValueError("No trials to aggregate")

        t0 = trials[0]
        n = len(trials)
        acc = sum(1 for t in trials if t.correct) / n
        acc_std = (acc * (1 - acc) / n) ** 0.5

        confidences = [t.confidence for t in trials if t.confidence is not None]
        mean_conf = sum(confidences) / len(confidences) if confidences else None

        cal_errors = [t.calibration_error for t in trials if t.calibration_error is not None]
        mean_cal = sum(cal_errors) / len(cal_errors) if cal_errors else None

        # Per-step error rates
        max_steps = t0.depth
        step_errors = []
        for step in range(1, max_steps + 1):
            step_correct = sum(
                1 for t in trials
                if len(t.step_results) >= step and t.step_results[step - 1].found_in_response
            )
            step_errors.append(1.0 - step_correct / n)

        return cls(
            family=t0.family,
            depth=t0.depth,
            cu=t0.cu,
            snr=t0.snr,
            n_trials=n,
            accuracy=acc,
            accuracy_std=acc_std,
            mean_confidence=mean_conf,
            mean_calibration_error=mean_cal,
            per_step_error_rates=step_errors,
        )
