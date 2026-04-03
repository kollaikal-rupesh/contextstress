"""
ContextStress: Benchmarking Reasoning Collapse in Long-Context Language Models.

A diagnostic benchmark for characterizing when and how LLM reasoning
collapses under increasing context load, with the Effective Context
Capacity (ECC) framework and Context Saturation Index (CSI).
"""

__version__ = "0.1.0"

from contextstress.assembly import ContextAssembler
from contextstress.evaluate import Evaluator
from contextstress.analysis import Analyzer, SaturationThresholdLaw
from contextstress.csi_guard import CSIGuard

__all__ = [
    "ContextAssembler",
    "Evaluator",
    "Analyzer",
    "SaturationThresholdLaw",
    "CSIGuard",
]
