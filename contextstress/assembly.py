"""
Context assembly module for ContextStress.

Assembles evaluation contexts by combining signal passages (from tasks)
with noise passages at specified CU and SNR levels.
"""

import random
from dataclasses import dataclass

from contextstress.tasks.generator import TaskInstance
from contextstress.noise_corpus.generator import NoisePassage


@dataclass
class AssembledContext:
    """A fully assembled evaluation context."""
    text: str
    total_tokens: int
    signal_tokens: int
    noise_tokens: int
    actual_cu: float
    actual_snr: float
    signal_positions: list[int]  # indices where signal passages were placed


class ContextAssembler:
    """Assembles contexts for ContextStress evaluation."""

    # Approximate tokens-per-word ratio
    TOKENS_PER_WORD = 1.33

    def __init__(
        self,
        noise_corpus: list[NoisePassage],
        context_window: int,
    ):
        self.noise_corpus = noise_corpus
        self.context_window = context_window
        self._noise_by_cluster: dict[int, list[NoisePassage]] = {}
        for p in noise_corpus:
            self._noise_by_cluster.setdefault(p.cluster_id, []).append(p)

    def _estimate_tokens(self, text: str) -> int:
        return int(len(text.split()) * self.TOKENS_PER_WORD)

    def assemble(
        self,
        task: TaskInstance,
        cu: float,
        snr: float,
        seed: int = 42,
        noise_type: str = "mixed",
    ) -> AssembledContext:
        """
        Assemble a context for evaluation.

        Args:
            task: The task instance containing signal passages.
            cu: Context utilization (0, 1]. Fraction of context window to fill.
            snr: Signal-to-noise ratio (0, 1]. Fraction of tokens that are signal.
            seed: Random seed for reproducible noise selection and placement.
            noise_type: "adjacent" (same cluster), "distant" (different), or "mixed".

        Returns:
            AssembledContext with the assembled text and metadata.
        """
        rng = random.Random(seed)
        target_tokens = int(cu * self.context_window)

        # Compute signal budget
        signal_passages = [sp["content"] for sp in task.signal_passages]
        signal_tokens = sum(self._estimate_tokens(p) for p in signal_passages)

        # Fill to target_tokens. Noise = total - signal.
        # At SNR=100%, still add noise to reach the CU target
        # (the SNR becomes approximate — actual SNR = signal/total).
        noise_tokens_target = max(0, target_tokens - signal_tokens)

        # Select noise passages
        noise_passages = self._select_noise(
            noise_tokens_target, rng, noise_type
        )
        noise_tokens = sum(self._estimate_tokens(p.content) for p in noise_passages)

        # Combine signal and noise into passage list
        all_passages = []

        # Add signal passages
        for sp in signal_passages:
            all_passages.append(("signal", sp))

        # Add noise passages
        for np_item in noise_passages:
            all_passages.append(("noise", np_item.content))

        # Shuffle to randomize signal positions (seeded)
        rng.shuffle(all_passages)

        # Record signal positions
        signal_positions = [
            i for i, (ptype, _) in enumerate(all_passages) if ptype == "signal"
        ]

        # Join with passage separators
        separator = "\n\n---\n\n"
        text = separator.join(content for _, content in all_passages)

        actual_total = self._estimate_tokens(text)

        return AssembledContext(
            text=text,
            total_tokens=actual_total,
            signal_tokens=signal_tokens,
            noise_tokens=noise_tokens,
            actual_cu=actual_total / self.context_window,
            actual_snr=signal_tokens / max(actual_total, 1),
            signal_positions=signal_positions,
        )

    def _select_noise(
        self,
        target_tokens: int,
        rng: random.Random,
        noise_type: str,
    ) -> list[NoisePassage]:
        """Select noise passages to fill the target token budget."""
        if target_tokens <= 0:
            return []

        if noise_type == "adjacent":
            # Use passages from a single cluster
            cluster_id = rng.randint(0, len(self._noise_by_cluster) - 1)
            candidates = list(self._noise_by_cluster.get(cluster_id, []))
        elif noise_type == "distant":
            # Use passages from multiple clusters
            candidates = list(self.noise_corpus)
        else:  # mixed
            candidates = list(self.noise_corpus)

        rng.shuffle(candidates)

        selected = []
        tokens_so_far = 0
        for passage in candidates:
            p_tokens = self._estimate_tokens(passage.content)
            if tokens_so_far + p_tokens > target_tokens:
                continue
            selected.append(passage)
            tokens_so_far += p_tokens
            if tokens_so_far >= target_tokens * 0.9:  # close enough
                break

        return selected
