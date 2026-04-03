#!/usr/bin/env python3
"""
ContextStress — Post-hoc Analysis Script

Loads experiment results CSV files and produces all analysis needed for the paper:
  - Per-condition accuracy, confidence, calibration (ECE)
  - Tables by task family x CU level (per model, per SNR)
  - Curve fitting (linear, quadratic, exponential, sigmoidal) with R², AIC, BIC
  - Threshold extraction (τ at 80% of baseline)
  - Saturation Threshold Law fitting
  - Bootstrap 95% CIs for thresholds
  - Pairwise statistical tests between adjacent CU levels

Usage:
  python3 analyze_results.py --results-dir results/full_run/
  python3 analyze_results.py --results-dir results/full_run/ --output-dir results/analysis/
"""

import argparse
import csv
import glob
import json
import math
import os
import sys
import warnings
from collections import defaultdict
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import chi2


# ── Degradation Models (from contextstress.analysis) ────────────────────

def linear_model(cu, a, b):
    return a * cu + b

def quadratic_model(cu, a, b, c):
    return a * cu**2 + b * cu + c

def exponential_model(cu, a, k, c):
    return a * np.exp(-k * cu) + c

def sigmoid_model(cu, L, U, k, tau):
    return L + (U - L) / (1 + np.exp(k * (cu - tau)))

MODELS = {
    "linear":      (linear_model, 2, ["a", "b"]),
    "quadratic":   (quadratic_model, 3, ["a", "b", "c"]),
    "exponential": (exponential_model, 3, ["A", "k", "c"]),
    "sigmoidal":   (sigmoid_model, 4, ["L", "U", "k", "tau"]),
}


# ── Data Structures ─────────────────────────────────────────────────────

@dataclass
class Trial:
    model: str
    family: str
    depth: int
    cu_target: float
    snr_target: float
    correct: bool
    confidence: Optional[float]
    calibration_error: Optional[float]

@dataclass
class ConditionStats:
    model: str
    family: str
    depth: int
    cu: float
    snr: float
    n: int
    accuracy: float
    accuracy_std: float
    mean_confidence: Optional[float]
    mean_calibration_error: Optional[float]

@dataclass
class CurveFitResult:
    model_name: str
    params: dict
    r_squared: float
    aic: float
    bic: float

@dataclass
class ThresholdResult:
    model: str
    family: str
    depth: int
    snr: float
    tau: float
    tau_ci_lo: float
    tau_ci_hi: float
    baseline_accuracy: float


# ── CSV Loading ──────────────────────────────────────────────────────────

def find_csv_files(results_dir: str) -> List[str]:
    """Find all results.csv files under results_dir (recursive)."""
    patterns = [
        os.path.join(results_dir, "**", "results.csv"),
        os.path.join(results_dir, "*.csv"),
    ]
    files = set()
    for p in patterns:
        files.update(glob.glob(p, recursive=True))
    return sorted(files)


def load_trials(csv_paths: List[str]) -> List[Trial]:
    """Load trial rows from one or more CSV files."""
    trials = []
    for path in csv_paths:
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    correct_raw = row["correct"].strip()
                    correct = correct_raw in ("True", "true", "1")

                    conf = None
                    if row.get("confidence") and row["confidence"].strip():
                        try:
                            conf = float(row["confidence"])
                        except ValueError:
                            pass

                    cal_err = None
                    if row.get("calibration_error") and row["calibration_error"].strip():
                        try:
                            cal_err = float(row["calibration_error"])
                        except ValueError:
                            pass

                    trials.append(Trial(
                        model=row["model"].strip(),
                        family=row["task_family"].strip(),
                        depth=int(row["depth"]),
                        cu_target=float(row["cu_target"]),
                        snr_target=float(row["snr_target"]),
                        correct=correct,
                        confidence=conf,
                        calibration_error=cal_err,
                    ))
                except (KeyError, ValueError) as e:
                    print(f"  [warn] skipping malformed row in {path}: {e}", file=sys.stderr)
    return trials


# ── Condition Aggregation ────────────────────────────────────────────────

def aggregate_conditions(trials: List[Trial]) -> List[ConditionStats]:
    """Group trials by (model, family, depth, cu, snr) and compute stats."""
    groups = defaultdict(list)
    for t in trials:
        key = (t.model, t.family, t.depth, t.cu_target, t.snr_target)
        groups[key].append(t)

    results = []
    for (model, family, depth, cu, snr), group in sorted(groups.items()):
        n = len(group)
        n_correct = sum(1 for t in group if t.correct)
        acc = n_correct / n if n > 0 else 0.0
        acc_std = math.sqrt(acc * (1 - acc) / n) if n > 0 else 0.0

        confs = [t.confidence for t in group if t.confidence is not None]
        mean_conf = sum(confs) / len(confs) if confs else None

        cal_errs = [t.calibration_error for t in group if t.calibration_error is not None]
        mean_cal = sum(cal_errs) / len(cal_errs) if cal_errs else None

        results.append(ConditionStats(
            model=model, family=family, depth=depth,
            cu=cu, snr=snr, n=n,
            accuracy=acc, accuracy_std=acc_std,
            mean_confidence=mean_conf,
            mean_calibration_error=mean_cal,
        ))
    return results


# ── Table Generation ─────────────────────────────────────────────────────

def generate_accuracy_tables(conditions: List[ConditionStats]) -> List[dict]:
    """
    Table 1-style: Accuracy by task family x CU level.
    One table per (model, snr).
    Returns list of table dicts for JSON output.
    """
    # Discover unique values
    models = sorted(set(c.model for c in conditions))
    snrs = sorted(set(c.snr for c in conditions))
    families = sorted(set(c.family for c in conditions))
    cus = sorted(set(c.cu for c in conditions))

    lookup = {}
    for c in conditions:
        lookup[(c.model, c.snr, c.family, c.cu)] = c

    tables = []
    for model in models:
        for snr in snrs:
            table = {
                "title": f"Accuracy (%) — {model}, SNR={snr:.0%}",
                "model": model,
                "snr": snr,
                "cu_levels": [f"{cu:.0%}" for cu in cus],
                "rows": [],
            }

            header = f"| {'Family':<8} | {'d':>2} |"
            for cu in cus:
                header += f" {'CU=' + f'{cu:.0%}':>10} |"
            md_lines = [header]
            sep = "|" + "----------|" + "----|" + "------------|" * len(cus)
            md_lines.append(sep)

            for fam in families:
                depths_for_fam = sorted(set(
                    c.depth for c in conditions
                    if c.model == model and c.snr == snr and c.family == fam
                ))
                for depth in depths_for_fam:
                    row_data = {"family": fam, "depth": depth, "values": {}}
                    line = f"| {fam:<8} | {depth:>2} |"
                    for cu in cus:
                        c = lookup.get((model, snr, fam, cu))
                        if c and c.depth == depth:
                            val_str = f"{c.accuracy*100:5.1f}±{c.accuracy_std*100:.1f}"
                            row_data["values"][f"{cu:.0%}"] = {
                                "accuracy": round(c.accuracy * 100, 1),
                                "std": round(c.accuracy_std * 100, 1),
                                "n": c.n,
                            }
                        else:
                            val_str = "    —    "
                        line += f" {val_str:>10} |"
                    md_lines.append(line)
                    table["rows"].append(row_data)

            table["markdown"] = "\n".join(md_lines)
            tables.append(table)
    return tables


def generate_confidence_tables(conditions: List[ConditionStats]) -> List[dict]:
    """Confidence and calibration error table per model."""
    models = sorted(set(c.model for c in conditions))
    families = sorted(set(c.family for c in conditions))
    cus = sorted(set(c.cu for c in conditions))

    lookup = {}
    for c in conditions:
        lookup[(c.model, c.family, c.cu)] = c

    tables = []
    for model in models:
        md_lines = [f"### Confidence & Calibration — {model}"]
        header = f"| {'Family':<8} |"
        for cu in cus:
            header += f" {'CU=' + f'{cu:.0%}':>20} |"
        md_lines.append(header)
        sep = "|----------|" + "----------------------|" * len(cus)
        md_lines.append(sep)

        rows = []
        for fam in families:
            line = f"| {fam:<8} |"
            row_data = {"family": fam, "values": {}}
            for cu in cus:
                c = lookup.get((model, fam, cu))
                if c and c.mean_confidence is not None and c.mean_calibration_error is not None:
                    val_str = f"C={c.mean_confidence:.1f} E={c.mean_calibration_error:.1f}"
                    row_data["values"][f"{cu:.0%}"] = {
                        "mean_confidence": round(c.mean_confidence, 1),
                        "mean_calibration_error": round(c.mean_calibration_error, 1),
                    }
                else:
                    val_str = "—"
                line += f" {val_str:>20} |"
            md_lines.append(line)
            rows.append(row_data)

        tables.append({
            "title": f"Confidence & Calibration — {model}",
            "model": model,
            "markdown": "\n".join(md_lines),
            "rows": rows,
        })
    return tables


# ── Curve Fitting ────────────────────────────────────────────────────────

def fit_degradation_curves(
    cu_values: np.ndarray,
    accuracy_values: np.ndarray,
) -> Dict[str, CurveFitResult]:
    """Fit all 4 degradation models. cu_values in [0,1], accuracy in [0,100]."""
    n = len(cu_values)
    # Convert CU to percentage for fitting (matches analysis.py convention)
    cu_pct = cu_values * 100.0
    results = {}

    for name, (model_fn, n_params, param_names) in MODELS.items():
        try:
            p0, bounds = _initial_guess(name)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                popt, _ = curve_fit(model_fn, cu_pct, accuracy_values,
                                    p0=p0, bounds=bounds, maxfev=10000)

            predicted = model_fn(cu_pct, *popt)
            residuals = accuracy_values - predicted
            ss_res = np.sum(residuals**2)
            ss_tot = np.sum((accuracy_values - np.mean(accuracy_values))**2)
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

            mse = ss_res / n if n > 0 else 1e-10
            if mse <= 0:
                mse = 1e-10
            log_lik = -n / 2 * (np.log(2 * np.pi * mse) + 1)
            aic = 2 * n_params - 2 * log_lik
            bic = n_params * np.log(n) - 2 * log_lik

            results[name] = CurveFitResult(
                model_name=name,
                params=dict(zip(param_names, [float(v) for v in popt])),
                r_squared=float(r2),
                aic=float(aic),
                bic=float(bic),
            )
        except Exception:
            continue

    return results


def _initial_guess(name):
    if name == "linear":
        return [-0.5, 90.0], (-np.inf, np.inf)
    elif name == "quadratic":
        return [-0.01, 0.5, 90.0], (-np.inf, np.inf)
    elif name == "exponential":
        return [80.0, 0.02, 10.0], ([0, 0, 0], [200, 1, 100])
    elif name == "sigmoidal":
        return [10.0, 95.0, 0.05, 50.0], ([0, 0, 0.001, 1], [100, 100, 1.0, 99])
    return None, (-np.inf, np.inf)


# ── Threshold Extraction ────────────────────────────────────────────────

def extract_threshold(
    cu_values: np.ndarray,
    accuracy_values: np.ndarray,
    baseline_accuracy: float,
    fraction: float = 0.8,
) -> float:
    """
    Extract τ: CU (in %) at which accuracy drops to fraction × baseline.
    Returns CU as a percentage (0-100).
    """
    target = fraction * baseline_accuracy
    cu_pct = cu_values * 100.0

    # Try sigmoid fit
    try:
        p0 = [10.0, 95.0, 0.05, 50.0]
        bounds = ([0, 0, 0.001, 1], [100, 100, 1.0, 99])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            popt, _ = curve_fit(sigmoid_model, cu_pct, accuracy_values,
                                p0=p0, bounds=bounds, maxfev=10000)
        cu_fine = np.linspace(cu_pct.min(), cu_pct.max(), 1000)
        acc_fine = sigmoid_model(cu_fine, *popt)
        crossings = np.where(acc_fine <= target)[0]
        if len(crossings) > 0:
            return float(cu_fine[crossings[0]])
    except Exception:
        pass

    # Fallback: linear interpolation
    for i in range(len(accuracy_values) - 1):
        if accuracy_values[i] >= target >= accuracy_values[i + 1]:
            frac = (accuracy_values[i] - target) / (
                accuracy_values[i] - accuracy_values[i + 1]
            )
            return float(cu_pct[i] + frac * (cu_pct[i + 1] - cu_pct[i]))

    return float(cu_pct[-1])


def bootstrap_threshold(
    cu_values: np.ndarray,
    accuracy_by_trial: List[List[float]],
    baseline_accuracy: float,
    n_boot: int = 1000,
    fraction: float = 0.8,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """
    Bootstrap CI for threshold τ.
    accuracy_by_trial: list of lists, one per CU level, each a list of bool (correct/not).
    Returns (tau, ci_lo, ci_hi).
    """
    rng = np.random.RandomState(seed)
    taus = []
    for _ in range(n_boot):
        boot_accs = []
        for trial_list in accuracy_by_trial:
            if len(trial_list) == 0:
                boot_accs.append(0.0)
                continue
            sample = rng.choice(trial_list, size=len(trial_list), replace=True)
            boot_accs.append(float(np.mean(sample)) * 100.0)
        boot_accs = np.array(boot_accs)
        tau = extract_threshold(cu_values, boot_accs, baseline_accuracy, fraction)
        taus.append(tau)

    taus = np.array(taus)
    point = float(np.median(taus))
    ci_lo = float(np.percentile(taus, 2.5))
    ci_hi = float(np.percentile(taus, 97.5))
    return point, ci_lo, ci_hi


# ── Saturation Threshold Law ────────────────────────────────────────────

def fit_threshold_law(thresholds: List[ThresholdResult]) -> Optional[dict]:
    """
    Fit τ(SNR, d) = τ_max · SNR^(γ₀ + γ₁·d)
    via log-linear regression.
    """
    valid = [t for t in thresholds if t.tau > 0 and t.snr > 0]
    if len(valid) < 4:
        return None

    log_tau = np.array([np.log(t.tau) for t in valid])
    log_snr = np.array([np.log(t.snr) for t in valid])
    depths = np.array([float(t.depth) for t in valid])

    X = np.column_stack([
        np.ones(len(valid)),
        log_snr,
        depths * log_snr,
    ])

    try:
        coeffs, _, _, _ = np.linalg.lstsq(X, log_tau, rcond=None)
    except np.linalg.LinAlgError:
        return None

    c0, c1, c2 = coeffs
    tau_max = float(np.exp(c0))
    gamma_0 = float(c1)
    gamma_1 = float(c2)

    predicted = np.array([
        tau_max * t.snr ** (gamma_0 + gamma_1 * t.depth) for t in valid
    ])
    actual = np.array([t.tau for t in valid])

    ss_res = np.sum((actual - predicted)**2)
    ss_tot = np.sum((actual - np.mean(actual))**2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    mae = float(np.mean(np.abs(actual - predicted)))
    rmse = float(np.sqrt(np.mean((actual - predicted)**2)))

    return {
        "tau_max": tau_max,
        "gamma_0": gamma_0,
        "gamma_1": gamma_1,
        "r_squared": round(r2, 4),
        "mae": round(mae, 2),
        "rmse": round(rmse, 2),
        "n_thresholds": len(valid),
    }


# ── Statistical Tests ───────────────────────────────────────────────────

def proportion_test(n1_correct: int, n1_total: int,
                    n2_correct: int, n2_total: int) -> dict:
    """
    Two-proportion z-test between adjacent CU levels.
    Returns z-stat and two-sided p-value.
    """
    p1 = n1_correct / n1_total if n1_total > 0 else 0
    p2 = n2_correct / n2_total if n2_total > 0 else 0
    p_pool = (n1_correct + n2_correct) / (n1_total + n2_total) if (n1_total + n2_total) > 0 else 0

    se = math.sqrt(p_pool * (1 - p_pool) * (1/n1_total + 1/n2_total)) if p_pool > 0 and p_pool < 1 else 1e-10
    z = (p1 - p2) / se if se > 0 else 0
    # Two-sided p-value from normal approximation
    from scipy.stats import norm
    p_val = 2 * (1 - norm.cdf(abs(z)))

    return {
        "p1": round(p1, 4),
        "p2": round(p2, 4),
        "z": round(z, 4),
        "p_value": round(p_val, 6),
        "significant_005": bool(p_val < 0.05),
    }


def pairwise_cu_tests(trials: List[Trial]) -> List[dict]:
    """Pairwise tests between adjacent CU levels within each (model, family, snr)."""
    groups = defaultdict(lambda: defaultdict(list))
    for t in trials:
        key = (t.model, t.family, t.snr_target)
        groups[key][t.cu_target].append(t)

    results = []
    for (model, family, snr), cu_map in sorted(groups.items()):
        cu_levels = sorted(cu_map.keys())
        for i in range(len(cu_levels) - 1):
            cu_a, cu_b = cu_levels[i], cu_levels[i + 1]
            trials_a = cu_map[cu_a]
            trials_b = cu_map[cu_b]
            n1 = len(trials_a)
            n1_correct = sum(1 for t in trials_a if t.correct)
            n2 = len(trials_b)
            n2_correct = sum(1 for t in trials_b if t.correct)

            test = proportion_test(n1_correct, n1, n2_correct, n2)
            results.append({
                "model": model,
                "family": family,
                "snr": snr,
                "cu_a": cu_a,
                "cu_b": cu_b,
                **test,
            })
    return results


# ── Main Analysis Pipeline ──────────────────────────────────────────────

def run_analysis(results_dir: str, output_dir: str):
    # 1. Discover and load CSVs
    csv_files = find_csv_files(results_dir)
    if not csv_files:
        print(f"ERROR: No CSV files found under {results_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(csv_files)} CSV file(s):")
    for f in csv_files:
        print(f"  {f}")

    trials = load_trials(csv_files)
    print(f"Loaded {len(trials)} trials total.\n")

    if not trials:
        print("ERROR: No valid trial data.", file=sys.stderr)
        sys.exit(1)

    # 2. Aggregate per-condition stats
    conditions = aggregate_conditions(trials)
    print(f"Aggregated into {len(conditions)} conditions.\n")

    # Summary
    models = sorted(set(c.model for c in conditions))
    families = sorted(set(c.family for c in conditions))
    cus = sorted(set(c.cu for c in conditions))
    snrs = sorted(set(c.snr for c in conditions))
    print(f"Models:   {models}")
    print(f"Families: {families}")
    print(f"CU levels: {[f'{cu:.0%}' for cu in cus]}")
    print(f"SNR levels: {[f'{s:.0%}' for s in snrs]}")
    print()

    # 3. Generate accuracy tables
    print("=" * 72)
    print("TABLE 1: Accuracy by Task Family x CU Level")
    print("=" * 72)
    acc_tables = generate_accuracy_tables(conditions)
    for table in acc_tables:
        print(f"\n{table['title']}")
        print(table["markdown"])
    print()

    # 4. Confidence & calibration tables
    print("=" * 72)
    print("CONFIDENCE & CALIBRATION TABLES")
    print("=" * 72)
    conf_tables = generate_confidence_tables(conditions)
    for table in conf_tables:
        print(f"\n{table['title']}")
        print(table["markdown"])
    print()

    # 5. Curve fitting
    print("=" * 72)
    print("CURVE FITTING: 4 Degradation Models")
    print("=" * 72)
    curve_results = []
    for model in models:
        for fam in families:
            for snr in snrs:
                subset = [c for c in conditions
                          if c.model == model and c.family == fam and c.snr == snr]
                if len(subset) < 3:
                    continue
                subset.sort(key=lambda c: c.cu)
                cu_arr = np.array([c.cu for c in subset])
                acc_arr = np.array([c.accuracy * 100 for c in subset])

                fits = fit_degradation_curves(cu_arr, acc_arr)
                if not fits:
                    continue

                combo_label = f"{model} | {fam} | SNR={snr:.0%}"
                print(f"\n  {combo_label}")
                print(f"  {'Model':<14} {'R²':>8} {'AIC':>10} {'BIC':>10}")
                print(f"  {'-'*14} {'-'*8} {'-'*10} {'-'*10}")

                for fname in ["linear", "quadratic", "exponential", "sigmoidal"]:
                    if fname in fits:
                        fr = fits[fname]
                        print(f"  {fname:<14} {fr.r_squared:8.4f} {fr.aic:10.2f} {fr.bic:10.2f}")
                        curve_results.append({
                            "model": model, "family": fam, "snr": snr,
                            "fit_model": fname,
                            "r_squared": round(fr.r_squared, 4),
                            "aic": round(fr.aic, 2),
                            "bic": round(fr.bic, 2),
                            "params": fr.params,
                        })
    print()

    # 6. Threshold extraction with bootstrap CIs
    print("=" * 72)
    print("THRESHOLD EXTRACTION (tau at 80% of baseline)")
    print("=" * 72)

    # Build per-trial data for bootstrap
    trial_groups = defaultdict(lambda: defaultdict(list))
    for t in trials:
        key = (t.model, t.family, t.depth, t.snr_target)
        trial_groups[key][t.cu_target].append(1.0 if t.correct else 0.0)

    all_thresholds = []
    print(f"\n  {'Model':<16} {'Family':<6} {'d':>2} {'SNR':>6} | {'tau':>7} {'95% CI':>16} {'Baseline':>9}")
    print(f"  {'-'*16} {'-'*6} {'-'*2} {'-'*6} | {'-'*7} {'-'*16} {'-'*9}")

    for model in models:
        for fam in families:
            depths_for_fam = sorted(set(
                c.depth for c in conditions if c.family == fam and c.model == model
            ))
            for depth in depths_for_fam:
                for snr in snrs:
                    subset = [c for c in conditions
                              if c.model == model and c.family == fam
                              and c.depth == depth and c.snr == snr]
                    if len(subset) < 3:
                        continue
                    subset.sort(key=lambda c: c.cu)
                    cu_arr = np.array([c.cu for c in subset])
                    acc_arr = np.array([c.accuracy * 100 for c in subset])
                    baseline = acc_arr[0]  # lowest CU = baseline

                    # Point estimate
                    tau_point = extract_threshold(cu_arr, acc_arr, baseline)

                    # Bootstrap CI
                    key = (model, fam, depth, snr)
                    cu_sorted = sorted(trial_groups[key].keys())
                    accuracy_by_trial = [trial_groups[key][cu] for cu in cu_sorted]
                    cu_arr_boot = np.array(cu_sorted)

                    if len(cu_arr_boot) >= 3 and all(len(a) > 0 for a in accuracy_by_trial):
                        _, ci_lo, ci_hi = bootstrap_threshold(
                            cu_arr_boot, accuracy_by_trial, baseline,
                            n_boot=1000,
                        )
                    else:
                        ci_lo, ci_hi = tau_point, tau_point

                    tr = ThresholdResult(
                        model=model, family=fam, depth=depth, snr=snr,
                        tau=tau_point, tau_ci_lo=ci_lo, tau_ci_hi=ci_hi,
                        baseline_accuracy=baseline,
                    )
                    all_thresholds.append(tr)
                    print(f"  {model:<16} {fam:<6} {depth:>2} {snr:>5.0%} | "
                          f"{tau_point:7.1f}% [{ci_lo:6.1f}, {ci_hi:6.1f}] {baseline:8.1f}%")
    print()

    # 7. Saturation Threshold Law
    print("=" * 72)
    print("SATURATION THRESHOLD LAW FIT")
    print("=" * 72)
    stl = fit_threshold_law(all_thresholds)
    if stl:
        print(f"\n  tau(SNR, d) = {stl['tau_max']:.2f} * SNR^({stl['gamma_0']:.4f} + {stl['gamma_1']:.4f} * d)")
        print(f"  R² = {stl['r_squared']:.4f}")
        print(f"  MAE = {stl['mae']:.2f}%")
        print(f"  RMSE = {stl['rmse']:.2f}%")
        print(f"  Fitted on {stl['n_thresholds']} threshold estimates")
    else:
        print("\n  Insufficient data for Saturation Threshold Law fit (need >= 4 thresholds).")
        stl = {}
    print()

    # 8. Statistical tests
    print("=" * 72)
    print("PAIRWISE STATISTICAL TESTS (adjacent CU levels)")
    print("=" * 72)
    pairwise = pairwise_cu_tests(trials)
    if pairwise:
        print(f"\n  {'Model':<16} {'Fam':<6} {'SNR':>5} {'CU_a':>6} {'CU_b':>6} | "
              f"{'p1':>6} {'p2':>6} {'z':>7} {'p-val':>10} {'Sig?':>5}")
        print(f"  {'-'*16} {'-'*6} {'-'*5} {'-'*6} {'-'*6} | {'-'*6} {'-'*6} {'-'*7} {'-'*10} {'-'*5}")
        for r in pairwise:
            sig_marker = "*" if r["significant_005"] else ""
            print(f"  {r['model']:<16} {r['family']:<6} {r['snr']:>4.0%} "
                  f"{r['cu_a']:>5.0%} {r['cu_b']:>5.0%} | "
                  f"{r['p1']:6.3f} {r['p2']:6.3f} {r['z']:7.3f} {r['p_value']:10.6f} {sig_marker:>5}")
    else:
        print("\n  Insufficient data for pairwise tests.")
    print()

    # 9. Save outputs
    os.makedirs(output_dir, exist_ok=True)

    # JSON output
    output = {
        "summary": {
            "n_trials": len(trials),
            "n_conditions": len(conditions),
            "models": models,
            "families": families,
            "cu_levels": cus,
            "snr_levels": snrs,
        },
        "conditions": [
            {
                "model": c.model, "family": c.family, "depth": c.depth,
                "cu": c.cu, "snr": c.snr, "n": c.n,
                "accuracy": round(c.accuracy * 100, 2),
                "accuracy_std": round(c.accuracy_std * 100, 2),
                "mean_confidence": round(c.mean_confidence, 2) if c.mean_confidence is not None else None,
                "mean_calibration_error": round(c.mean_calibration_error, 2) if c.mean_calibration_error is not None else None,
            }
            for c in conditions
        ],
        "curve_fits": curve_results,
        "thresholds": [
            {
                "model": t.model, "family": t.family, "depth": t.depth,
                "snr": t.snr, "tau": round(t.tau, 2),
                "tau_ci_lo": round(t.tau_ci_lo, 2),
                "tau_ci_hi": round(t.tau_ci_hi, 2),
                "baseline_accuracy": round(t.baseline_accuracy, 2),
            }
            for t in all_thresholds
        ],
        "saturation_threshold_law": stl if stl else None,
        "pairwise_tests": pairwise,
    }

    json_path = os.path.join(output_dir, "analysis.json")
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Saved JSON: {json_path}")

    # Markdown output
    md_path = os.path.join(output_dir, "analysis.md")
    with open(md_path, "w") as f:
        f.write("# ContextStress Analysis Results\n\n")

        f.write("## Accuracy Tables\n\n")
        for table in acc_tables:
            f.write(f"### {table['title']}\n\n")
            f.write(table["markdown"] + "\n\n")

        f.write("## Confidence & Calibration\n\n")
        for table in conf_tables:
            f.write(table["markdown"] + "\n\n")

        f.write("## Curve Fitting\n\n")
        f.write("| Model | Family | SNR | Fit | R² | AIC | BIC |\n")
        f.write("|-------|--------|-----|-----|-----|-----|-----|\n")
        for cr in curve_results:
            f.write(f"| {cr['model']} | {cr['family']} | {cr['snr']:.0%} | "
                    f"{cr['fit_model']} | {cr['r_squared']:.4f} | "
                    f"{cr['aic']:.1f} | {cr['bic']:.1f} |\n")

        f.write("\n## Thresholds\n\n")
        f.write("| Model | Family | d | SNR | tau (%) | 95% CI | Baseline |\n")
        f.write("|-------|--------|---|-----|---------|--------|----------|\n")
        for t in all_thresholds:
            f.write(f"| {t.model} | {t.family} | {t.depth} | {t.snr:.0%} | "
                    f"{t.tau:.1f} | [{t.tau_ci_lo:.1f}, {t.tau_ci_hi:.1f}] | "
                    f"{t.baseline_accuracy:.1f} |\n")

        if stl:
            f.write("\n## Saturation Threshold Law\n\n")
            f.write(f"tau(SNR, d) = {stl['tau_max']:.2f} * SNR^({stl['gamma_0']:.4f} + {stl['gamma_1']:.4f} * d)\n\n")
            f.write(f"- R² = {stl['r_squared']:.4f}\n")
            f.write(f"- MAE = {stl['mae']:.2f}%\n")
            f.write(f"- RMSE = {stl['rmse']:.2f}%\n")

        f.write("\n## Pairwise Tests\n\n")
        f.write("| Model | Family | SNR | CU_a | CU_b | z | p-value | Sig? |\n")
        f.write("|-------|--------|-----|------|------|---|---------|------|\n")
        for r in pairwise:
            sig = "Yes" if r["significant_005"] else "No"
            f.write(f"| {r['model']} | {r['family']} | {r['snr']:.0%} | "
                    f"{r['cu_a']:.0%} | {r['cu_b']:.0%} | {r['z']:.3f} | "
                    f"{r['p_value']:.6f} | {sig} |\n")

    print(f"Saved Markdown: {md_path}")
    print("\nDone.")


# ── CLI ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="ContextStress — Analyze experiment results for the paper."
    )
    parser.add_argument(
        "--results-dir",
        required=True,
        help="Directory containing results CSV files (searched recursively).",
    )
    parser.add_argument(
        "--output-dir",
        default="results/analysis",
        help="Directory for output files (default: results/analysis/).",
    )
    args = parser.parse_args()
    run_analysis(args.results_dir, args.output_dir)


if __name__ == "__main__":
    main()
