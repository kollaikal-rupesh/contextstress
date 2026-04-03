"""
ContextStress Experiment Runner — Token-Budget-Aware

Runs T2 (d=2) and T3 (d=4) tasks across 4 CU levels on GPT-4o,
with automatic token budget enforcement (240K daily limit).

Usage:
    export OPENAI_API_KEY=sk-...
    python experiments/run_experiment.py

    # Or with custom budget:
    python experiments/run_experiment.py --budget 200000 --tasks-per-condition 10
"""

import argparse
import csv
import json
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from contextstress.tasks.generator import generate_all_tasks
from contextstress.noise_corpus.generator import generate_noise_corpus
from contextstress.assembly import ContextAssembler
from contextstress.adapters import OpenAIAdapter, parse_answer_and_confidence, SYSTEM_PROMPT, USER_TEMPLATE
from contextstress.evaluate import Evaluator


# ── Token Budget Tracker ──────────────────────────────────────────────


@dataclass
class TokenBudget:
    """Tracks cumulative token usage against a daily budget."""
    daily_limit: int
    safety_margin: int
    tokens_used: int = 0
    requests_made: int = 0
    requests_skipped: int = 0
    _log: list = field(default_factory=list)

    @property
    def remaining(self) -> int:
        return max(0, self.daily_limit - self.safety_margin - self.tokens_used)

    @property
    def budget(self) -> int:
        return self.daily_limit - self.safety_margin

    def can_afford(self, estimated_tokens: int) -> bool:
        return self.tokens_used + estimated_tokens <= self.budget

    def record(self, tokens: int, condition: str):
        self.tokens_used += tokens
        self.requests_made += 1
        self._log.append({
            "request": self.requests_made,
            "tokens": tokens,
            "cumulative": self.tokens_used,
            "remaining": self.remaining,
            "condition": condition,
        })

    def skip(self):
        self.requests_skipped += 1

    def summary(self) -> str:
        pct = self.tokens_used / self.budget * 100 if self.budget > 0 else 0
        return (
            f"Token Budget Report:\n"
            f"  Daily limit:      {self.daily_limit:>10,}\n"
            f"  Safety margin:    {self.safety_margin:>10,}\n"
            f"  Effective budget: {self.budget:>10,}\n"
            f"  Tokens used:      {self.tokens_used:>10,} ({pct:.1f}%)\n"
            f"  Tokens remaining: {self.remaining:>10,}\n"
            f"  Requests made:    {self.requests_made:>10}\n"
            f"  Requests skipped: {self.requests_skipped:>10}\n"
        )

    def save_log(self, path: Path):
        with open(path, "w") as f:
            json.dump({
                "daily_limit": self.daily_limit,
                "safety_margin": self.safety_margin,
                "tokens_used": self.tokens_used,
                "requests_made": self.requests_made,
                "requests_skipped": self.requests_skipped,
                "log": self._log,
            }, f, indent=2)


# ── Token Estimator ───────────────────────────────────────────────────


def estimate_request_tokens(context_tokens: int) -> int:
    """Estimate total tokens for a request (input + output)."""
    system_tokens = len(SYSTEM_PROMPT.split()) * 2  # ~overestimate
    query_tokens = 50  # typical query length
    output_tokens = 256  # max_tokens setting
    return context_tokens + system_tokens + query_tokens + output_tokens


# ── Experiment Configuration ──────────────────────────────────────────


@dataclass
class ExperimentConfig:
    model_id: str = "gpt-4o"
    context_window: int = 128_000
    task_families: list = field(default_factory=lambda: ["T2", "T3"])
    cu_levels: list = field(default_factory=lambda: [0.05, 0.30, 0.60, 0.90])
    snr: float = 0.5  # Fixed for this experiment; extensible later
    tasks_per_condition: int = 20
    seed: int = 42
    daily_token_limit: int = 250_000
    safety_margin: int = 10_000

    @property
    def total_conditions(self) -> int:
        return len(self.task_families) * len(self.cu_levels)

    @property
    def total_requests(self) -> int:
        return self.total_conditions * self.tasks_per_condition

    def estimate_total_tokens(self) -> int:
        """Estimate total tokens for the full experiment."""
        total = 0
        for cu in self.cu_levels:
            ctx_tokens = int(cu * self.context_window)
            per_request = estimate_request_tokens(ctx_tokens)
            total += per_request * self.tasks_per_condition * len(self.task_families)
        return total


# ── CSV Logger ────────────────────────────────────────────────────────


class CSVLogger:
    """Logs individual trial results to CSV."""

    HEADERS = [
        "timestamp", "model", "task_family", "depth", "task_id",
        "cu_target", "cu_actual", "snr_target", "snr_actual",
        "context_tokens", "answer_expected", "answer_given",
        "correct", "confidence", "calibration_error",
        "tokens_used", "latency_ms",
    ]

    def __init__(self, path: Path):
        self.path = path
        self._file = open(path, "w", newline="")
        self._writer = csv.DictWriter(self._file, fieldnames=self.HEADERS)
        self._writer.writeheader()
        self._file.flush()

    def log(self, row: dict):
        self._writer.writerow(row)
        self._file.flush()

    def close(self):
        self._file.close()


# ── Main Runner ───────────────────────────────────────────────────────


def run_experiment(config: ExperimentConfig, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Pre-flight ────────────────────────────────────────────────
    budget = TokenBudget(
        daily_limit=config.daily_token_limit,
        safety_margin=config.safety_margin,
    )

    estimated_total = config.estimate_total_tokens()
    print("=" * 60)
    print("ContextStress Experiment Runner")
    print("=" * 60)
    print(f"Model:            {config.model_id}")
    print(f"Task families:    {', '.join(config.task_families)}")
    print(f"CU levels:        {', '.join(f'{cu:.0%}' for cu in config.cu_levels)}")
    print(f"SNR:              {config.snr:.0%}")
    print(f"Tasks/condition:  {config.tasks_per_condition}")
    print(f"Total conditions: {config.total_conditions}")
    print(f"Total requests:   {config.total_requests}")
    print(f"Est. total tokens: {estimated_total:,}")
    print(f"Token budget:     {budget.budget:,} (limit {budget.daily_limit:,} - margin {budget.safety_margin:,})")
    print()

    if estimated_total > budget.budget:
        print(f"WARNING: Estimated tokens ({estimated_total:,}) exceed budget ({budget.budget:,})!")
        print("The runner will stop when the budget is exhausted.")
        print("Conditions are ordered by CU (low→high) to maximize completed conditions.")
    else:
        print(f"Estimated tokens ({estimated_total:,}) within budget ({budget.budget:,}).")

    print()

    # ── Setup ─────────────────────────────────────────────────────
    print("Generating tasks...", end=" ", flush=True)
    all_tasks = generate_all_tasks(seed=config.seed)
    print(f"{len(all_tasks)} tasks")

    print("Generating noise corpus...", end=" ", flush=True)
    noise = generate_noise_corpus(seed=config.seed, num_passages=2000, num_clusters=20)
    print(f"{len(noise)} passages")

    adapter = OpenAIAdapter(model_id=config.model_id, max_context=config.context_window)
    assembler = ContextAssembler(noise, config.context_window)
    evaluator = Evaluator()
    csv_logger = CSVLogger(output_dir / "results.csv")

    # ── Depth mapping ─────────────────────────────────────────────
    depth_map = {"T1": 1, "T2": 2, "T3": 4, "T4": 3}

    # ── Run ───────────────────────────────────────────────────────
    condition_results = []
    budget_exhausted = False

    # Sort CU ascending so cheaper conditions run first
    for cu in sorted(config.cu_levels):
        if budget_exhausted:
            break

        for family in config.task_families:
            if budget_exhausted:
                break

            depth = depth_map[family]
            family_tasks = [t for t in all_tasks if t.family == family][:config.tasks_per_condition]
            condition_label = f"{family}/CU={cu:.0%}/SNR={config.snr:.0%}"

            # Pre-estimate cost for this condition
            est_ctx_tokens = int(cu * config.context_window)
            est_per_request = estimate_request_tokens(est_ctx_tokens)
            est_condition_total = est_per_request * len(family_tasks)

            print(f"\n{'─' * 60}")
            print(f"Condition: {condition_label}")
            print(f"  Est. tokens: {est_condition_total:,} | Budget remaining: {budget.remaining:,}")

            if not budget.can_afford(est_condition_total):
                # Try running as many as budget allows
                affordable = budget.remaining // est_per_request
                if affordable < 3:
                    print(f"  SKIPPED — insufficient budget for meaningful results")
                    budget.skip()
                    continue
                print(f"  Budget allows only {affordable}/{len(family_tasks)} tasks")
                family_tasks = family_tasks[:affordable]

            correct_count = 0
            condition_trials = []

            for i, task in enumerate(family_tasks):
                # Final budget check per request
                if not budget.can_afford(est_per_request):
                    print(f"  BUDGET EXHAUSTED at task {i+1}/{len(family_tasks)}")
                    budget_exhausted = True
                    break

                # Assemble context
                ctx = assembler.assemble(task, cu=cu, snr=config.snr, seed=config.seed)

                # Call model
                t0 = time.time()
                response = adapter.generate(task.query, ctx.text)
                latency = (time.time() - t0) * 1000

                # Evaluate
                trial = evaluator.evaluate_trial(task, response, cu, config.snr, config.seed)
                if trial.correct:
                    correct_count += 1

                # Record tokens
                budget.record(response.tokens_used, condition_label)

                # Log to CSV
                csv_logger.log({
                    "timestamp": datetime.now().isoformat(),
                    "model": config.model_id,
                    "task_family": family,
                    "depth": depth,
                    "task_id": task.id,
                    "cu_target": cu,
                    "cu_actual": f"{ctx.actual_cu:.4f}",
                    "snr_target": config.snr,
                    "snr_actual": f"{ctx.actual_snr:.4f}",
                    "context_tokens": ctx.total_tokens,
                    "answer_expected": task.answer,
                    "answer_given": response.answer[:200],
                    "correct": trial.correct,
                    "confidence": response.confidence,
                    "calibration_error": trial.calibration_error,
                    "tokens_used": response.tokens_used,
                    "latency_ms": f"{latency:.0f}",
                })

                condition_trials.append(trial)

                # Throttle to avoid rate limits (1s base + proportional to tokens)
                time.sleep(max(1.0, response.tokens_used / 50000))

                status = "✓" if trial.correct else "✗"
                print(
                    f"  [{i+1:>2}/{len(family_tasks)}] {status}  "
                    f"Exp: {task.answer[:15]:15s}  "
                    f"Got: {response.answer[:25]:25s}  "
                    f"Conf: {response.confidence}%  "
                    f"Tok: {response.tokens_used:,}  "
                    f"[{budget.tokens_used:,}/{budget.budget:,}]"
                )

            if condition_trials:
                n = len(condition_trials)
                acc = correct_count / n * 100
                confs = [t.confidence for t in condition_trials if t.confidence is not None]
                avg_conf = sum(confs) / len(confs) if confs else 0
                cal_errs = [t.calibration_error for t in condition_trials if t.calibration_error is not None]
                avg_cal = sum(cal_errs) / len(cal_errs) if cal_errs else 0

                condition_results.append({
                    "family": family,
                    "depth": depth,
                    "cu": cu,
                    "snr": config.snr,
                    "n": n,
                    "accuracy": acc,
                    "avg_confidence": avg_conf,
                    "avg_calibration_error": avg_cal,
                })
                print(f"  → Accuracy: {correct_count}/{n} = {acc:.0f}%  Avg confidence: {avg_conf:.0f}%  CCE: {avg_cal:.1f}%")

    csv_logger.close()

    # ── Summary ───────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)

    if condition_results:
        print(f"\n{'Family':>6} {'Depth':>5} {'CU':>6} {'SNR':>5} {'N':>4} {'Accuracy':>8} {'Confidence':>10} {'CCE':>6}")
        print("─" * 58)
        for r in condition_results:
            print(
                f"{r['family']:>6} {r['depth']:>5} {r['cu']:>5.0%} {r['snr']:>4.0%} "
                f"{r['n']:>4} {r['accuracy']:>7.0f}% {r['avg_confidence']:>9.0f}% {r['avg_calibration_error']:>5.1f}%"
            )

    print(f"\n{budget.summary()}")

    # Save summary
    summary_path = output_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump({
            "config": {
                "model": config.model_id,
                "task_families": config.task_families,
                "cu_levels": config.cu_levels,
                "snr": config.snr,
                "tasks_per_condition": config.tasks_per_condition,
            },
            "results": condition_results,
            "budget": {
                "daily_limit": budget.daily_limit,
                "tokens_used": budget.tokens_used,
                "requests_made": budget.requests_made,
                "requests_skipped": budget.requests_skipped,
            },
        }, f, indent=2)

    budget.save_log(output_dir / "token_log.json")

    print(f"\nFiles saved:")
    print(f"  {output_dir / 'results.csv'}")
    print(f"  {output_dir / 'summary.json'}")
    print(f"  {output_dir / 'token_log.json'}")

    return condition_results


# ── CLI ───────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="ContextStress Experiment Runner")
    parser.add_argument("--model", default="gpt-4o", help="Model ID (default: gpt-4o)")
    parser.add_argument("--budget", type=int, default=250_000, help="Daily token limit (default: 250000)")
    parser.add_argument("--margin", type=int, default=10_000, help="Safety margin tokens (default: 10000)")
    parser.add_argument("--tasks-per-condition", type=int, default=20, help="Tasks per condition (default: 20)")
    parser.add_argument("--cu-levels", nargs="+", type=float, default=[0.05, 0.30, 0.60, 0.90])
    parser.add_argument("--families", nargs="+", default=["T2", "T3"])
    parser.add_argument("--snr", type=float, default=0.5, help="Signal-to-noise ratio (default: 0.5)")
    parser.add_argument("--output-dir", default="results/", help="Output directory")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    config = ExperimentConfig(
        model_id=args.model,
        task_families=args.families,
        cu_levels=args.cu_levels,
        snr=args.snr,
        tasks_per_condition=args.tasks_per_condition,
        daily_token_limit=args.budget,
        safety_margin=args.margin,
        seed=args.seed,
    )

    output_dir = Path(args.output_dir) / f"{config.model_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_experiment(config, output_dir)


if __name__ == "__main__":
    main()
