"""
CLI entry point for ContextStress benchmark.

Usage:
    python -m contextstress --model openai --output-dir results/
    python -m contextstress --generate-tasks
    python -m contextstress --generate-noise
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np


# Experimental matrix
CU_LEVELS = [0.05, 0.15, 0.30, 0.50, 0.75, 0.95]
SNR_LEVELS = [1.0, 0.75, 0.50, 0.25, 0.10]
SEEDS = [42, 137, 256]


def generate_tasks(args):
    from contextstress.tasks.generator import generate_all_tasks
    output = Path(args.output_dir) / "tasks"
    tasks = generate_all_tasks(seed=42, output_dir=output)
    print(f"Generated {len(tasks)} tasks in {output}")


def generate_noise(args):
    from contextstress.noise_corpus.generator import generate_noise_corpus
    output = Path(args.output_dir) / "noise_corpus"
    corpus = generate_noise_corpus(seed=42, output_dir=output)
    print(f"Generated {len(corpus)} noise passages in {output}")


def run_benchmark(args):
    from contextstress.tasks.generator import generate_all_tasks
    from contextstress.noise_corpus.generator import generate_noise_corpus
    from contextstress.assembly import ContextAssembler
    from contextstress.evaluate import Evaluator, ConditionResult
    from contextstress.analysis import Analyzer, ThresholdResult

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize model adapter
    adapter = _get_adapter(args.model)
    print(f"Model: {adapter.name} (context window: {adapter.context_window:,} tokens)")

    # Generate data
    print("Generating tasks...")
    tasks = generate_all_tasks(seed=42)

    print("Generating noise corpus...")
    noise = generate_noise_corpus(seed=42)

    assembler = ContextAssembler(noise, adapter.context_window)
    evaluator = Evaluator()
    analyzer = Analyzer()

    # Run experimental matrix
    all_results = []
    total_conditions = len(CU_LEVELS) * len(SNR_LEVELS) * 4  # 4 task families
    condition_idx = 0

    for cu in CU_LEVELS:
        for snr in SNR_LEVELS:
            for family in ["T1", "T2", "T3", "T4"]:
                condition_idx += 1
                family_tasks = [t for t in tasks if t.family == family]

                print(
                    f"\r[{condition_idx}/{total_conditions}] "
                    f"CU={cu:.0%} SNR={snr:.0%} {family}",
                    end="", flush=True,
                )

                trial_results = []
                for task in family_tasks:
                    for seed in SEEDS:
                        # Assemble context
                        ctx = assembler.assemble(task, cu, snr, seed=seed)

                        # Generate response
                        response = adapter.generate(task.query, ctx.text)

                        # Evaluate
                        result = evaluator.evaluate_trial(
                            task, response, cu, snr, seed
                        )
                        trial_results.append(result)

                # Aggregate condition results
                condition = ConditionResult.from_trials(trial_results)
                all_results.append(condition)

    print(f"\nCompleted {len(all_results)} conditions.")

    # Save raw results
    results_path = output_dir / f"{adapter.name}_results.json"
    with open(results_path, "w") as f:
        json.dump(
            [
                {
                    "family": r.family,
                    "depth": r.depth,
                    "cu": r.cu,
                    "snr": r.snr,
                    "accuracy": r.accuracy,
                    "accuracy_std": r.accuracy_std,
                    "mean_confidence": r.mean_confidence,
                    "mean_calibration_error": r.mean_calibration_error,
                    "n_trials": r.n_trials,
                }
                for r in all_results
            ],
            f,
            indent=2,
        )
    print(f"Results saved to {results_path}")

    # Fit threshold law
    print("\nFitting Saturation Threshold Law...")
    thresholds = []
    for family in ["T1", "T2", "T3", "T4"]:
        depth = {"T1": 1, "T2": 2, "T3": 4, "T4": 3}[family]
        for snr in SNR_LEVELS:
            condition_data = [
                r for r in all_results
                if r.family == family and r.snr == snr
            ]
            if not condition_data:
                continue

            cu_arr = np.array([r.cu for r in condition_data])
            acc_arr = np.array([r.accuracy * 100 for r in condition_data])

            # Baseline = accuracy at lowest CU
            baseline = acc_arr[0] if len(acc_arr) > 0 else 100

            tau = analyzer.extract_threshold(cu_arr * 100, acc_arr, baseline)
            thresholds.append(ThresholdResult(
                family=family, depth=depth, snr=snr,
                tau=tau, baseline_accuracy=baseline,
            ))

    if len(thresholds) >= 4:
        law = analyzer.fit_threshold_law(thresholds)
        print(f"  τ_max = {law.tau_max:.1f}%")
        print(f"  γ₀    = {law.gamma_0:.3f}")
        print(f"  γ₁    = {law.gamma_1:.4f}")
        print(f"  R²    = {law.r_squared:.3f}")
        print(f"  MAE   = {law.mae:.1f}pp")

        law_path = output_dir / f"{adapter.name}_threshold_law.json"
        with open(law_path, "w") as f:
            json.dump({
                "tau_max": law.tau_max,
                "gamma_0": law.gamma_0,
                "gamma_1": law.gamma_1,
                "r_squared": law.r_squared,
                "mae": law.mae,
                "rmse": law.rmse,
            }, f, indent=2)
        print(f"  Saved to {law_path}")


def _get_adapter(model_name: str):
    from contextstress.adapters import OpenAIAdapter, AnthropicAdapter

    adapters = {
        "openai": lambda: OpenAIAdapter(),
        "gpt-4o": lambda: OpenAIAdapter(model_id="gpt-4o-2024-08-06"),
        "claude": lambda: AnthropicAdapter(),
        "claude-3.5-sonnet": lambda: AnthropicAdapter(),
    }

    if model_name not in adapters:
        available = ", ".join(adapters.keys())
        print(f"Unknown model '{model_name}'. Available: {available}")
        print("Or implement your own ModelAdapter subclass.")
        sys.exit(1)

    return adapters[model_name]()


def main():
    parser = argparse.ArgumentParser(
        prog="contextstress",
        description="ContextStress: Benchmarking Reasoning Collapse in Long-Context LLMs",
    )
    subparsers = parser.add_subparsers(dest="command")

    # Generate tasks
    gen_tasks = subparsers.add_parser("generate-tasks", help="Generate benchmark tasks")
    gen_tasks.add_argument("--output-dir", default="data/", help="Output directory")

    # Generate noise
    gen_noise = subparsers.add_parser("generate-noise", help="Generate noise corpus")
    gen_noise.add_argument("--output-dir", default="data/", help="Output directory")

    # Run benchmark
    run = subparsers.add_parser("run", help="Run the full benchmark")
    run.add_argument("--model", required=True, help="Model adapter name")
    run.add_argument("--output-dir", default="results/", help="Output directory")

    args = parser.parse_args()

    if args.command == "generate-tasks":
        generate_tasks(args)
    elif args.command == "generate-noise":
        generate_noise(args)
    elif args.command == "run":
        run_benchmark(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
