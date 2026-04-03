"""
CSI-Guard: Adaptive Context Management Algorithm.

Uses the Saturation Threshold Law at runtime to prevent
operation in unreliable regimes.
"""

from dataclasses import dataclass
from typing import Optional

from contextstress.analysis import SaturationThresholdLaw


@dataclass
class Passage:
    """A candidate passage for context inclusion."""
    content: str
    relevance_score: float  # 0-1, from retriever
    token_count: int


@dataclass
class CSIGuardResult:
    """Result from CSI-Guard filtering."""
    selected_passages: list[Passage]
    rejected_passages: list[Passage]
    token_budget: int
    tokens_used: int
    csi_before: float
    csi_after: float
    zone_before: str
    zone_after: str
    decomposed: bool  # Whether query was decomposed


class CSIGuard:
    """
    CSI-Guard: Adaptive Context Management.

    Algorithm 1 from the paper. Uses model-specific Saturation
    Threshold Law parameters to cap context volume at runtime.
    """

    def __init__(
        self,
        law: SaturationThresholdLaw,
        context_window: int,
        target_zone: str = "Green",
    ):
        """
        Args:
            law: Fitted SaturationThresholdLaw with τ_max, γ₀, γ₁.
            context_window: Model's maximum context window in tokens.
            target_zone: "Green" (CSI < 0.7) or "Yellow" (CSI < 1.0).
        """
        self.law = law
        self.context_window = context_window
        self.target_zone = target_zone

    def filter_passages(
        self,
        query_tokens: int,
        system_prompt_tokens: int,
        passages: list[Passage],
        estimated_depth: int,
        estimated_snr: float,
    ) -> CSIGuardResult:
        """
        Apply CSI-Guard filtering (Algorithm 1, Steps 1-7).

        Args:
            query_tokens: Token count of the query.
            system_prompt_tokens: Token count of the system prompt.
            passages: Candidate passages sorted by relevance (descending).
            estimated_depth: Estimated reasoning depth d.
            estimated_snr: Estimated signal-to-noise ratio.

        Returns:
            CSIGuardResult with selected/rejected passages and CSI info.
        """
        # Step 4: Compute threshold
        tau = self.law.predict(estimated_snr, estimated_depth)

        # Step 5: Compute CSI target
        if self.target_zone == "Green":
            cu_max = 0.7 * tau / 100.0  # Convert from percentage
        else:  # Yellow
            cu_max = 1.0 * tau / 100.0

        # Step 6: Compute token budget
        budget = int(cu_max * self.context_window)

        # Compute CSI before filtering (all passages)
        all_tokens = sum(p.token_count for p in passages) + query_tokens + system_prompt_tokens
        cu_before = all_tokens / self.context_window
        csi_before = self.law.csi(cu_before * 100, estimated_snr, estimated_depth)
        zone_before = self.law.zone(cu_before * 100, estimated_snr, estimated_depth)

        # Step 7: Greedy passage selection
        sorted_passages = sorted(passages, key=lambda p: p.relevance_score, reverse=True)
        selected = []
        rejected = []
        tokens_used = query_tokens + system_prompt_tokens

        for passage in sorted_passages:
            if tokens_used + passage.token_count <= budget:
                selected.append(passage)
                tokens_used += passage.token_count
            else:
                rejected.append(passage)

        # Compute CSI after filtering
        cu_after = tokens_used / self.context_window
        csi_after = self.law.csi(cu_after * 100, estimated_snr, estimated_depth)
        zone_after = self.law.zone(cu_after * 100, estimated_snr, estimated_depth)

        return CSIGuardResult(
            selected_passages=selected,
            rejected_passages=rejected,
            token_budget=budget,
            tokens_used=tokens_used,
            csi_before=csi_before,
            csi_after=csi_after,
            zone_before=zone_before,
            zone_after=zone_after,
            decomposed=len(selected) == 0,
        )

    @staticmethod
    def estimate_depth(query: str) -> int:
        """
        Estimate reasoning depth from query text using syntactic heuristics.
        (Step 1 of Algorithm 1)
        """
        query_lower = query.lower()

        # Count complexity indicators
        relative_clauses = sum(1 for w in ["who", "which", "that", "whose", "where"]
                               if w in query_lower)
        entity_refs = len([w for w in query.split() if w[0].isupper()]) if query else 0
        comparisons = sum(1 for w in ["highest", "lowest", "most", "least", "compare",
                                       "between", "versus", "than"]
                          if w in query_lower)

        # Heuristic depth estimation
        complexity = relative_clauses + (entity_refs // 3) + comparisons

        if complexity <= 1:
            return 1
        elif complexity <= 2:
            return 2
        elif complexity <= 3:
            return 3
        else:
            return 4

    @staticmethod
    def estimate_snr(passages: list[Passage], threshold: float = 0.5) -> float:
        """
        Estimate SNR from retriever confidence scores.
        (Step 3 of Algorithm 1)
        """
        if not passages:
            return 0.5  # Conservative default

        signal_count = sum(1 for p in passages if p.relevance_score >= threshold)
        return signal_count / len(passages)
