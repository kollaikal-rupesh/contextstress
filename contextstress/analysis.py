"""
Analysis module for ContextStress.

Provides curve fitting (4 models), threshold extraction,
Saturation Threshold Law fitting, and CSI computation.
"""

import warnings
from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import pearsonr

from contextstress.evaluate import ConditionResult


# ── Degradation Curve Models ──────────────────────────────────────────


def linear_model(cu, a, b):
    return a * cu + b


def quadratic_model(cu, a, b, c):
    return a * cu**2 + b * cu + c


def exponential_model(cu, a, k, c):
    return a * np.exp(-k * cu) + c


def sigmoid_model(cu, L, U, k, tau):
    """Four-parameter logistic: A(CU) = L + (U-L) / (1 + exp(k*(CU - tau)))"""
    return L + (U - L) / (1 + np.exp(k * (cu - tau)))


MODELS = {
    "linear": (linear_model, 2),
    "quadratic": (quadratic_model, 3),
    "exponential": (exponential_model, 3),
    "sigmoidal": (sigmoid_model, 4),
}


@dataclass
class CurveFitResult:
    """Result of fitting a degradation curve."""
    model_name: str
    params: dict
    r_squared: float
    adj_r_squared: float
    aic: float
    bic: float
    residuals: np.ndarray


@dataclass
class ThresholdResult:
    """Extracted saturation threshold for one condition."""
    family: str
    depth: int
    snr: float
    tau: float  # CU at which accuracy = 80% of baseline
    baseline_accuracy: float


# ── Saturation Threshold Law ──────────────────────────────────────────


@dataclass
class SaturationThresholdLaw:
    """Fitted Saturation Threshold Law: τ(SNR, d) = τ_max · SNR^(γ₀ + γ₁·d)"""
    tau_max: float
    gamma_0: float
    gamma_1: float
    r_squared: float
    mae: float
    rmse: float

    def predict(self, snr: float, d: int) -> float:
        """Predict the saturation threshold for given SNR and depth."""
        return self.tau_max * snr ** (self.gamma_0 + self.gamma_1 * d)

    def csi(self, cu: float, snr: float, d: int) -> float:
        """Compute the Context Saturation Index."""
        tau = self.predict(snr, d)
        return cu / tau if tau > 0 else float("inf")

    def zone(self, cu: float, snr: float, d: int) -> str:
        """Classify the operating zone."""
        c = self.csi(cu, snr, d)
        if c < 0.7:
            return "Green"
        elif c < 1.0:
            return "Yellow"
        elif c < 1.5:
            return "Red"
        else:
            return "Critical"


class Analyzer:
    """Main analysis engine for ContextStress results."""

    def fit_degradation_curve(
        self,
        cu_values: np.ndarray,
        accuracy_values: np.ndarray,
    ) -> dict[str, CurveFitResult]:
        """Fit all four degradation models to accuracy-vs-CU data."""
        n = len(cu_values)
        results = {}

        for name, (model_fn, n_params) in MODELS.items():
            try:
                result = self._fit_single(
                    name, model_fn, n_params, cu_values, accuracy_values, n
                )
                results[name] = result
            except Exception:
                continue

        return results

    def _fit_single(
        self,
        name: str,
        model_fn,
        n_params: int,
        cu: np.ndarray,
        acc: np.ndarray,
        n: int,
    ) -> CurveFitResult:
        """Fit a single model."""
        # Initial parameter guesses
        bounds = (-np.inf, np.inf)
        p0 = None

        if name == "linear":
            p0 = [-0.5, 90.0]
        elif name == "quadratic":
            p0 = [-0.01, 0.5, 90.0]
        elif name == "exponential":
            p0 = [80.0, 0.02, 10.0]
            bounds = ([0, 0, 0], [200, 1, 100])
        elif name == "sigmoidal":
            p0 = [10.0, 95.0, 0.05, 50.0]
            bounds = ([0, 0, 0.001, 1], [100, 100, 1.0, 99])

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            popt, _ = curve_fit(model_fn, cu, acc, p0=p0, bounds=bounds, maxfev=10000)

        predicted = model_fn(cu, *popt)
        residuals = acc - predicted

        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((acc - np.mean(acc))**2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        adj_r2 = 1 - (1 - r2) * (n - 1) / (n - n_params - 1) if n > n_params + 1 else r2

        # AIC and BIC
        mse = ss_res / n
        log_likelihood = -n / 2 * (np.log(2 * np.pi * mse) + 1) if mse > 0 else 0
        aic = 2 * n_params - 2 * log_likelihood
        bic = n_params * np.log(n) - 2 * log_likelihood

        param_names = {
            "linear": ["a", "b"],
            "quadratic": ["a", "b", "c"],
            "exponential": ["A", "k", "c"],
            "sigmoidal": ["L", "U", "k", "tau"],
        }

        return CurveFitResult(
            model_name=name,
            params=dict(zip(param_names[name], popt)),
            r_squared=r2,
            adj_r_squared=adj_r2,
            aic=aic,
            bic=bic,
            residuals=residuals,
        )

    def extract_threshold(
        self,
        cu_values: np.ndarray,
        accuracy_values: np.ndarray,
        baseline_accuracy: float,
        threshold_fraction: float = 0.8,
    ) -> float:
        """
        Extract the saturation threshold τ: the CU at which accuracy
        drops to threshold_fraction × baseline.
        """
        target = threshold_fraction * baseline_accuracy

        # Fit sigmoid to get smooth curve
        try:
            fits = self.fit_degradation_curve(cu_values, accuracy_values)
            if "sigmoidal" in fits:
                sig = fits["sigmoidal"]
                # Find CU where sigmoid crosses target
                cu_fine = np.linspace(cu_values.min(), cu_values.max(), 1000)
                acc_fine = sigmoid_model(cu_fine, **{
                    k: v for k, v in sig.params.items()
                })
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
                return float(cu_values[i] + frac * (cu_values[i + 1] - cu_values[i]))

        return float(cu_values[-1])  # Never crosses threshold

    def fit_threshold_law(
        self,
        thresholds: list[ThresholdResult],
    ) -> SaturationThresholdLaw:
        """
        Fit the Saturation Threshold Law:
        τ(SNR, d) = τ_max · SNR^(γ₀ + γ₁·d)

        Takes log on both sides:
        log(τ) = log(τ_max) + (γ₀ + γ₁·d) · log(SNR)
        """
        # Filter valid thresholds
        valid = [t for t in thresholds if t.tau > 0 and t.snr > 0]
        if len(valid) < 4:
            raise ValueError(f"Need ≥4 thresholds, got {len(valid)}")

        log_tau = np.array([np.log(t.tau) for t in valid])
        log_snr = np.array([np.log(t.snr) for t in valid])
        depths = np.array([t.depth for t in valid])

        # Design matrix: log(τ) = c₀ + c₁·log(SNR) + c₂·d·log(SNR)
        X = np.column_stack([
            np.ones(len(valid)),
            log_snr,
            depths * log_snr,
        ])

        # Least squares fit
        coeffs, residuals, _, _ = np.linalg.lstsq(X, log_tau, rcond=None)
        c0, c1, c2 = coeffs

        tau_max = np.exp(c0)
        gamma_0 = c1
        gamma_1 = c2

        # Compute fit quality
        predicted = np.array([
            tau_max * t.snr ** (gamma_0 + gamma_1 * t.depth) for t in valid
        ])
        actual = np.array([t.tau for t in valid])

        errors = np.abs(predicted - actual)
        mae = float(np.mean(errors))
        rmse = float(np.sqrt(np.mean(errors**2)))

        ss_res = np.sum((actual - predicted)**2)
        ss_tot = np.sum((actual - np.mean(actual))**2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0

        return SaturationThresholdLaw(
            tau_max=float(tau_max),
            gamma_0=float(gamma_0),
            gamma_1=float(gamma_1),
            r_squared=float(r2),
            mae=mae,
            rmse=rmse,
        )
