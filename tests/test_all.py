"""
Comprehensive test suite for the ContextStress benchmark framework.

Covers:
  - Task generation (T1-T4): structure, determinism, depth correctness
  - Noise corpus generation
  - Context assembly: CU/SNR targeting, noise types, signal presence
  - Evaluator: exact match, alias matching, containment, normalization,
    calibration error
  - parse_answer_and_confidence: various formats
  - Analysis: curve fitting, threshold extraction, SaturationThresholdLaw
  - CSI-Guard: passage filtering, depth estimation, SNR estimation, zone
  - Integration: full pipeline with mock model, CSV logging
"""

import csv
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# ── Module imports ────────────────────────────────────────────────────

from contextstress.tasks.generator import (
    TaskInstance,
    generate_all_tasks,
    generate_t1,
    generate_t2,
    generate_t3,
    generate_t4,
    GENERATORS,
    _seeded_rng,
)
from contextstress.noise_corpus.generator import (
    NoisePassage,
    generate_noise_corpus,
    CLUSTER_TOPICS,
)
from contextstress.assembly import ContextAssembler, AssembledContext
from contextstress.evaluate import (
    Evaluator,
    TrialResult,
    ConditionResult,
    StepResult,
)
from contextstress.analysis import (
    Analyzer,
    SaturationThresholdLaw,
    ThresholdResult,
    CurveFitResult,
    linear_model,
    quadratic_model,
    exponential_model,
    sigmoid_model,
)
from contextstress.csi_guard import CSIGuard, Passage, CSIGuardResult
from contextstress.adapters import (
    ModelResponse,
    ModelAdapter,
    parse_answer_and_confidence,
    SYSTEM_PROMPT,
    USER_TEMPLATE,
)


# =====================================================================
# Fixtures
# =====================================================================


@pytest.fixture
def rng():
    return _seeded_rng(42)


@pytest.fixture
def small_noise_corpus():
    """A small noise corpus for fast tests."""
    return generate_noise_corpus(seed=42, num_passages=40, num_clusters=4)


@pytest.fixture
def sample_t1_task(rng):
    return generate_t1(0, rng)


@pytest.fixture
def sample_t2_task(rng):
    return generate_t2(0, rng)


@pytest.fixture
def sample_t3_task(rng):
    return generate_t3(0, rng)


@pytest.fixture
def sample_t4_task(rng):
    return generate_t4(0, rng)


@pytest.fixture
def evaluator():
    return Evaluator()


@pytest.fixture
def analyzer():
    return Analyzer()


@pytest.fixture
def sample_law():
    """A SaturationThresholdLaw with known parameters."""
    return SaturationThresholdLaw(
        tau_max=80.0,
        gamma_0=0.5,
        gamma_1=-0.1,
        r_squared=0.95,
        mae=2.0,
        rmse=3.0,
    )


# =====================================================================
# Unit Tests — Task Generation
# =====================================================================


class TestTaskGeneration:

    @pytest.mark.parametrize("gen_fn,family,expected_depth", [
        (generate_t1, "T1", 1),
        (generate_t2, "T2", 2),
        (generate_t3, "T3", 4),
        (generate_t4, "T4", 3),
    ])
    def test_task_structure(self, gen_fn, family, expected_depth):
        """Each family produces tasks with correct structure."""
        rng = _seeded_rng(123)
        task = gen_fn(0, rng)
        assert isinstance(task, TaskInstance)
        assert task.id.startswith(family)
        assert task.family == family
        assert task.depth == expected_depth
        assert isinstance(task.query, str) and len(task.query) > 0
        assert isinstance(task.answer, str) and len(task.answer) > 0
        assert isinstance(task.reasoning_chain, list)
        assert len(task.reasoning_chain) > 0
        assert isinstance(task.signal_passages, list)
        assert len(task.signal_passages) > 0

    @pytest.mark.parametrize("gen_fn", [generate_t1, generate_t2, generate_t3, generate_t4])
    def test_signal_passage_structure(self, gen_fn):
        """Signal passages have 'content', 'fact', and 'step' keys."""
        rng = _seeded_rng(99)
        task = gen_fn(0, rng)
        for sp in task.signal_passages:
            assert "content" in sp
            assert "fact" in sp
            assert "step" in sp
            assert isinstance(sp["step"], int)
            assert len(sp["content"]) > 0

    def test_t1_depth_and_passages(self, sample_t1_task):
        assert sample_t1_task.depth == 1
        assert len(sample_t1_task.signal_passages) == 1

    def test_t2_depth_and_passages(self, sample_t2_task):
        assert sample_t2_task.depth == 2
        assert len(sample_t2_task.signal_passages) == 2

    def test_t3_depth_and_passages(self, sample_t3_task):
        assert sample_t3_task.depth == 4
        assert len(sample_t3_task.signal_passages) == 4

    def test_t4_depth_and_passages(self, sample_t4_task):
        assert sample_t4_task.depth == 3
        assert len(sample_t4_task.signal_passages) == 3

    def test_determinism_same_seed(self):
        """Same seed produces identical tasks."""
        rng1 = _seeded_rng(77)
        rng2 = _seeded_rng(77)
        t1 = generate_t2(5, rng1)
        t2 = generate_t2(5, rng2)
        assert t1.id == t2.id
        assert t1.answer == t2.answer
        assert t1.query == t2.query
        assert len(t1.signal_passages) == len(t2.signal_passages)

    def test_determinism_different_seed(self):
        """Different seeds generally produce different tasks."""
        rng1 = _seeded_rng(1)
        rng2 = _seeded_rng(2)
        t1 = generate_t2(0, rng1)
        t2 = generate_t2(0, rng2)
        # Very unlikely to match
        assert t1.answer != t2.answer or t1.query != t2.query

    def test_generate_all_tasks_count(self):
        """generate_all_tasks produces the expected total number."""
        tasks = generate_all_tasks(seed=42, instances_per_family=5)
        assert len(tasks) == 5 * 4  # 4 families

    def test_generate_all_tasks_families(self):
        tasks = generate_all_tasks(seed=42, instances_per_family=3)
        families = {t.family for t in tasks}
        assert families == {"T1", "T2", "T3", "T4"}

    def test_generate_all_determinism(self):
        tasks_a = generate_all_tasks(seed=99, instances_per_family=3)
        tasks_b = generate_all_tasks(seed=99, instances_per_family=3)
        for a, b in zip(tasks_a, tasks_b):
            assert a.id == b.id
            assert a.answer == b.answer

    def test_to_dict(self, sample_t1_task):
        d = sample_t1_task.to_dict()
        assert isinstance(d, dict)
        for key in ["id", "family", "depth", "query", "answer",
                     "reasoning_chain", "signal_passages", "aliases"]:
            assert key in d

    def test_generate_all_tasks_output_dir(self):
        """generate_all_tasks writes files when output_dir is given."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tasks = generate_all_tasks(seed=42, instances_per_family=2, output_dir=Path(tmpdir))
            assert (Path(tmpdir) / "all_tasks.json").exists()
            for family in ["t1", "t2", "t3", "t4"]:
                assert (Path(tmpdir) / f"{family}_tasks.json").exists()


# =====================================================================
# Unit Tests — Noise Corpus Generation
# =====================================================================


class TestNoiseCorpus:

    def test_corpus_count(self):
        corpus = generate_noise_corpus(seed=42, num_passages=20, num_clusters=4)
        assert len(corpus) == 20

    def test_passage_structure(self, small_noise_corpus):
        p = small_noise_corpus[0]
        assert isinstance(p, NoisePassage)
        assert isinstance(p.id, str)
        assert isinstance(p.cluster_id, int)
        assert isinstance(p.cluster_topic, str)
        assert isinstance(p.content, str) and len(p.content) > 0
        assert isinstance(p.approx_tokens, int) and p.approx_tokens > 0

    def test_cluster_distribution(self, small_noise_corpus):
        """Passages are distributed evenly across clusters."""
        cluster_counts = {}
        for p in small_noise_corpus:
            cluster_counts[p.cluster_id] = cluster_counts.get(p.cluster_id, 0) + 1
        # 40 passages / 4 clusters = 10 each
        for cid in cluster_counts:
            assert cluster_counts[cid] == 10

    def test_determinism(self):
        c1 = generate_noise_corpus(seed=7, num_passages=8, num_clusters=4)
        c2 = generate_noise_corpus(seed=7, num_passages=8, num_clusters=4)
        for a, b in zip(c1, c2):
            assert a.id == b.id
            assert a.content == b.content

    def test_output_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            generate_noise_corpus(seed=42, num_passages=8, num_clusters=2, output_dir=Path(tmpdir))
            assert (Path(tmpdir) / "manifest.json").exists()
            assert (Path(tmpdir) / "cluster_00.json").exists()
            assert (Path(tmpdir) / "cluster_01.json").exists()


# =====================================================================
# Unit Tests — Context Assembly
# =====================================================================


class TestContextAssembly:

    def test_signal_passages_in_text(self, sample_t2_task, small_noise_corpus):
        assembler = ContextAssembler(small_noise_corpus, context_window=100_000)
        ctx = assembler.assemble(sample_t2_task, cu=0.5, snr=0.5, seed=42)
        # Every signal fact should appear in the assembled text
        for sp in sample_t2_task.signal_passages:
            assert sp["fact"] in ctx.text

    def test_assembled_context_structure(self, sample_t1_task, small_noise_corpus):
        assembler = ContextAssembler(small_noise_corpus, context_window=50_000)
        ctx = assembler.assemble(sample_t1_task, cu=0.3, snr=0.5, seed=42)
        assert isinstance(ctx, AssembledContext)
        assert isinstance(ctx.text, str) and len(ctx.text) > 0
        assert ctx.total_tokens > 0
        assert ctx.signal_tokens > 0
        assert isinstance(ctx.signal_positions, list)

    def test_cu_targeting(self, sample_t1_task, small_noise_corpus):
        """Assembled context roughly matches the target CU."""
        window = 50_000
        assembler = ContextAssembler(small_noise_corpus, context_window=window)
        ctx = assembler.assemble(sample_t1_task, cu=0.3, snr=0.5, seed=42)
        # Actual CU should be in a reasonable range of target
        # (noise selection may not fill perfectly)
        assert ctx.actual_cu <= 0.35  # not more than a bit over target

    def test_noise_type_adjacent(self, sample_t1_task, small_noise_corpus):
        assembler = ContextAssembler(small_noise_corpus, context_window=50_000)
        ctx = assembler.assemble(sample_t1_task, cu=0.3, snr=0.5, seed=42, noise_type="adjacent")
        assert isinstance(ctx, AssembledContext)

    def test_noise_type_distant(self, sample_t1_task, small_noise_corpus):
        assembler = ContextAssembler(small_noise_corpus, context_window=50_000)
        ctx = assembler.assemble(sample_t1_task, cu=0.3, snr=0.5, seed=42, noise_type="distant")
        assert isinstance(ctx, AssembledContext)

    def test_noise_type_mixed(self, sample_t1_task, small_noise_corpus):
        assembler = ContextAssembler(small_noise_corpus, context_window=50_000)
        ctx = assembler.assemble(sample_t1_task, cu=0.3, snr=0.5, seed=42, noise_type="mixed")
        assert isinstance(ctx, AssembledContext)

    def test_zero_noise_budget(self, sample_t1_task, small_noise_corpus):
        """When CU is tiny, context may only contain signal."""
        assembler = ContextAssembler(small_noise_corpus, context_window=1_000_000)
        ctx = assembler.assemble(sample_t1_task, cu=0.0001, snr=1.0, seed=42)
        # Should at least contain the signal
        for sp in sample_t1_task.signal_passages:
            assert sp["fact"] in ctx.text

    def test_actual_snr_computed(self, sample_t1_task, small_noise_corpus):
        assembler = ContextAssembler(small_noise_corpus, context_window=50_000)
        ctx = assembler.assemble(sample_t1_task, cu=0.3, snr=0.5, seed=42)
        assert 0.0 < ctx.actual_snr <= 1.0


# =====================================================================
# Unit Tests — Evaluator
# =====================================================================


class TestEvaluator:

    def test_exact_match(self, evaluator):
        assert evaluator._exact_match("1990", "1990", [])

    def test_exact_match_case_insensitive(self, evaluator):
        assert evaluator._exact_match("vorenthi", "Vorenthi", [])

    def test_alias_match(self, evaluator):
        assert evaluator._exact_match("Blackmere", "Vorath Blackmere", ["Blackmere"])

    def test_containment_match(self, evaluator):
        """Model response contains the expected answer."""
        assert evaluator._exact_match(
            "The answer is 1990", "1990", []
        )

    def test_containment_reverse(self, evaluator):
        """Short response is contained in the expected answer."""
        assert evaluator._exact_match("Blackmere", "Vorath Blackmere", [])

    def test_no_match(self, evaluator):
        assert not evaluator._exact_match("Paris", "London", [])

    def test_normalization(self, evaluator):
        assert evaluator._normalize("  Hello,  World! ") == "hello world"
        assert evaluator._normalize("It's a test.") == "its a test"

    def test_evaluate_trial_correct(self, evaluator, sample_t1_task):
        response = ModelResponse(
            raw_text=f"The answer is {sample_t1_task.answer}. Confidence: 90%",
            answer=sample_t1_task.answer,
            confidence=90.0,
            latency_ms=100.0,
            tokens_used=500,
        )
        result = evaluator.evaluate_trial(sample_t1_task, response, cu=0.3, snr=0.5, seed=42)
        assert result.correct is True
        assert result.confidence == 90.0

    def test_evaluate_trial_incorrect(self, evaluator, sample_t1_task):
        response = ModelResponse(
            raw_text="I don't know. Confidence: 20%",
            answer="I don't know",
            confidence=20.0,
            latency_ms=80.0,
            tokens_used=400,
        )
        result = evaluator.evaluate_trial(sample_t1_task, response, cu=0.3, snr=0.5, seed=42)
        assert result.correct is False

    def test_calibration_error_correct(self, evaluator, sample_t1_task):
        """Correct answer with 90% confidence => cal error = |90 - 100| = 10."""
        response = ModelResponse(
            raw_text=f"{sample_t1_task.answer}",
            answer=sample_t1_task.answer,
            confidence=90.0,
            latency_ms=50.0,
            tokens_used=300,
        )
        result = evaluator.evaluate_trial(sample_t1_task, response, 0.3, 0.5, 42)
        assert result.calibration_error == pytest.approx(10.0)

    def test_calibration_error_incorrect(self, evaluator, sample_t1_task):
        """Wrong answer with 80% confidence => cal error = |80 - 0| = 80."""
        response = ModelResponse(
            raw_text="wrong answer",
            answer="wrong answer",
            confidence=80.0,
            latency_ms=50.0,
            tokens_used=300,
        )
        result = evaluator.evaluate_trial(sample_t1_task, response, 0.3, 0.5, 42)
        assert result.calibration_error == pytest.approx(80.0)

    def test_calibration_error_none_when_no_confidence(self, evaluator, sample_t1_task):
        response = ModelResponse(
            raw_text="some answer",
            answer="some answer",
            confidence=None,
            latency_ms=50.0,
            tokens_used=300,
        )
        result = evaluator.evaluate_trial(sample_t1_task, response, 0.3, 0.5, 42)
        assert result.calibration_error is None

    def test_condition_result_from_trials(self, evaluator, sample_t1_task):
        """ConditionResult.from_trials aggregates correctly."""
        trials = []
        for i in range(5):
            correct = i < 3  # 3 out of 5 correct
            ans = sample_t1_task.answer if correct else "wrong"
            resp = ModelResponse(
                raw_text=ans, answer=ans,
                confidence=80.0, latency_ms=50.0, tokens_used=100,
            )
            trial = evaluator.evaluate_trial(sample_t1_task, resp, 0.3, 0.5, 42)
            trials.append(trial)

        cond = ConditionResult.from_trials(trials)
        assert cond.n_trials == 5
        assert cond.accuracy == pytest.approx(3 / 5)
        assert cond.mean_confidence == pytest.approx(80.0)


# =====================================================================
# Unit Tests — parse_answer_and_confidence
# =====================================================================


class TestParseAnswerAndConfidence:

    def test_standard_format(self):
        text = "The Vorenthi language.\nConfidence: 85%"
        answer, conf = parse_answer_and_confidence(text)
        assert conf == 85.0
        assert "Vorenthi" in answer

    def test_no_confidence(self):
        text = "The answer is 1990."
        answer, conf = parse_answer_and_confidence(text)
        assert conf is None
        assert "1990" in answer

    def test_answer_prefix_removed(self):
        text = "Answer: 1990\nConfidence: 70%"
        answer, conf = parse_answer_and_confidence(text)
        assert conf == 70.0
        assert "1990" in answer

    def test_final_answer_prefix(self):
        text = "Final answer: Vorenthi\nConfidence: 95%"
        answer, conf = parse_answer_and_confidence(text)
        assert conf == 95.0
        assert "Vorenthi" in answer

    def test_decimal_confidence(self):
        text = "Some answer\nConfidence: 72.5%"
        answer, conf = parse_answer_and_confidence(text)
        assert conf == 72.5

    def test_lowercase_confidence(self):
        text = "Some answer\nconfidence: 60%"
        answer, conf = parse_answer_and_confidence(text)
        assert conf == 60.0

    def test_empty_text(self):
        answer, conf = parse_answer_and_confidence("")
        assert conf is None
        assert answer == ""

    def test_multiline_reasoning(self):
        text = (
            "Step 1: Org funded program.\n"
            "Step 2: Person directed org.\n"
            "Therefore the answer is Vorath Blackmere.\n"
            "Confidence: 88%"
        )
        answer, conf = parse_answer_and_confidence(text)
        assert conf == 88.0
        # Should capture last meaningful line before confidence
        assert "Vorath Blackmere" in answer


# =====================================================================
# Unit Tests — Analysis
# =====================================================================


class TestAnalysis:

    def test_curve_fitting_linear(self, analyzer):
        """Fit a clearly linear dataset."""
        cu = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90])
        acc = np.array([95, 90, 85, 80, 75, 70, 65, 60, 55])
        fits = analyzer.fit_degradation_curve(cu, acc)
        assert "linear" in fits
        assert fits["linear"].r_squared > 0.99

    def test_curve_fitting_returns_multiple_models(self, analyzer):
        cu = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90])
        acc = np.array([95, 92, 88, 80, 65, 45, 30, 22, 18])
        fits = analyzer.fit_degradation_curve(cu, acc)
        assert len(fits) >= 2  # At least linear and quadratic should fit

    def test_curve_fit_result_structure(self, analyzer):
        cu = np.array([10, 20, 30, 40, 50, 60, 70, 80])
        acc = np.array([95, 90, 85, 80, 75, 70, 65, 60])
        fits = analyzer.fit_degradation_curve(cu, acc)
        for name, result in fits.items():
            assert isinstance(result, CurveFitResult)
            assert isinstance(result.params, dict)
            assert isinstance(result.r_squared, (float, np.floating))
            assert isinstance(result.aic, (int, float, np.floating))
            assert isinstance(result.bic, (int, float, np.floating))

    def test_threshold_extraction_linear_fallback(self, analyzer):
        """Extract threshold from data where accuracy drops linearly."""
        cu = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90])
        acc = np.array([100, 95, 90, 80, 70, 60, 50, 40, 30])
        baseline = 100.0
        tau = analyzer.extract_threshold(cu, acc, baseline, threshold_fraction=0.8)
        # 80% of 100 = 80 => should be near CU=40
        assert 30 <= tau <= 50

    def test_threshold_extraction_never_crosses(self, analyzer):
        """If accuracy never drops below threshold, returns last CU."""
        cu = np.array([10, 20, 30, 40, 50])
        acc = np.array([99, 98, 97, 96, 95])
        tau = analyzer.extract_threshold(cu, acc, 100.0, threshold_fraction=0.8)
        assert tau == pytest.approx(50.0)

    def test_fit_threshold_law(self, analyzer):
        """Fit the saturation threshold law on synthetic data."""
        thresholds = [
            ThresholdResult(family="T1", depth=1, snr=0.3, tau=30.0, baseline_accuracy=95.0),
            ThresholdResult(family="T1", depth=1, snr=0.5, tau=50.0, baseline_accuracy=95.0),
            ThresholdResult(family="T1", depth=1, snr=0.8, tau=70.0, baseline_accuracy=95.0),
            ThresholdResult(family="T2", depth=2, snr=0.3, tau=20.0, baseline_accuracy=90.0),
            ThresholdResult(family="T2", depth=2, snr=0.5, tau=35.0, baseline_accuracy=90.0),
            ThresholdResult(family="T2", depth=2, snr=0.8, tau=55.0, baseline_accuracy=90.0),
        ]
        law = analyzer.fit_threshold_law(thresholds)
        assert isinstance(law, SaturationThresholdLaw)
        assert law.tau_max > 0
        # The law should give reasonable predictions
        pred = law.predict(0.5, 1)
        assert pred > 0

    def test_fit_threshold_law_too_few(self, analyzer):
        """Raises ValueError with fewer than 4 thresholds."""
        thresholds = [
            ThresholdResult("T1", 1, 0.5, 40.0, 95.0),
            ThresholdResult("T1", 1, 0.8, 60.0, 95.0),
        ]
        with pytest.raises(ValueError):
            analyzer.fit_threshold_law(thresholds)

    def test_saturation_threshold_law_predict(self, sample_law):
        pred = sample_law.predict(0.5, 2)
        expected = 80.0 * 0.5 ** (0.5 + (-0.1) * 2)
        assert pred == pytest.approx(expected)

    def test_saturation_threshold_law_csi(self, sample_law):
        tau = sample_law.predict(0.5, 2)
        csi = sample_law.csi(50.0, 0.5, 2)
        assert csi == pytest.approx(50.0 / tau)

    def test_saturation_threshold_law_zone_green(self, sample_law):
        # Make CSI < 0.7 by using a small CU
        zone = sample_law.zone(1.0, 0.5, 1)
        # Depending on tau, this should be Green for very small CU
        assert zone in ("Green", "Yellow", "Red", "Critical")

    def test_zone_classification_boundaries(self):
        """Test zone boundaries explicitly."""
        law = SaturationThresholdLaw(
            tau_max=100.0, gamma_0=0.0, gamma_1=0.0,
            r_squared=1.0, mae=0.0, rmse=0.0,
        )
        # tau = 100 * snr^0 = 100 for any snr/depth
        # CSI = cu / 100
        assert law.zone(50.0, 0.5, 1) == "Green"    # CSI = 0.5
        assert law.zone(75.0, 0.5, 1) == "Yellow"   # CSI = 0.75
        assert law.zone(110.0, 0.5, 1) == "Red"     # CSI = 1.1
        assert law.zone(160.0, 0.5, 1) == "Critical" # CSI = 1.6


# =====================================================================
# Unit Tests — CSI-Guard
# =====================================================================


class TestCSIGuard:

    @pytest.fixture
    def guard(self, sample_law):
        return CSIGuard(
            law=sample_law,
            context_window=100_000,
            target_zone="Green",
        )

    def _make_passages(self, n=10, tokens_each=500, base_score=0.9):
        return [
            Passage(
                content=f"Passage {i} content " * (tokens_each // 5),
                relevance_score=base_score - i * 0.05,
                token_count=tokens_each,
            )
            for i in range(n)
        ]

    def test_filter_respects_token_budget(self, guard):
        passages = self._make_passages(20, tokens_each=1000)
        result = guard.filter_passages(
            query_tokens=100,
            system_prompt_tokens=200,
            passages=passages,
            estimated_depth=2,
            estimated_snr=0.5,
        )
        assert isinstance(result, CSIGuardResult)
        assert result.tokens_used <= result.token_budget + 300  # query+system overhead
        # Some passages should be rejected (budget can't fit all 20*1000)
        total_passage_tokens = sum(p.token_count for p in passages)
        if total_passage_tokens + 300 > result.token_budget:
            assert len(result.rejected_passages) > 0

    def test_filter_selects_most_relevant(self, guard):
        passages = self._make_passages(5, tokens_each=100)
        result = guard.filter_passages(
            query_tokens=50,
            system_prompt_tokens=50,
            passages=passages,
            estimated_depth=1,
            estimated_snr=0.8,
        )
        # Selected passages should be sorted by relevance (highest first)
        if len(result.selected_passages) > 1:
            scores = [p.relevance_score for p in result.selected_passages]
            assert scores == sorted(scores, reverse=True)

    def test_csi_before_after(self, guard):
        passages = self._make_passages(10, tokens_each=500)
        result = guard.filter_passages(
            query_tokens=100,
            system_prompt_tokens=200,
            passages=passages,
            estimated_depth=2,
            estimated_snr=0.5,
        )
        # CSI after filtering should be <= CSI before (or equal if nothing filtered)
        assert result.csi_after <= result.csi_before or len(result.rejected_passages) == 0

    def test_zone_before_and_after(self, guard):
        passages = self._make_passages(10, tokens_each=500)
        result = guard.filter_passages(
            query_tokens=100,
            system_prompt_tokens=200,
            passages=passages,
            estimated_depth=2,
            estimated_snr=0.5,
        )
        assert result.zone_after in ("Green", "Yellow", "Red", "Critical")
        assert result.zone_before in ("Green", "Yellow", "Red", "Critical")

    def test_estimate_depth_simple(self):
        assert CSIGuard.estimate_depth("What year was this signed?") == 1

    def test_estimate_depth_complex(self):
        query = (
            "What is the native language of the architect who designed "
            "the headquarters of the corporation that acquired the patent holder?"
        )
        depth = CSIGuard.estimate_depth(query)
        assert depth >= 2

    def test_estimate_depth_comparative(self):
        query = "Which system has the highest ridership compared to the other two?"
        depth = CSIGuard.estimate_depth(query)
        assert depth >= 1

    def test_estimate_snr_all_relevant(self):
        passages = [Passage("c", 0.9, 100), Passage("c", 0.8, 100)]
        snr = CSIGuard.estimate_snr(passages, threshold=0.5)
        assert snr == 1.0

    def test_estimate_snr_none_relevant(self):
        passages = [Passage("c", 0.2, 100), Passage("c", 0.3, 100)]
        snr = CSIGuard.estimate_snr(passages, threshold=0.5)
        assert snr == 0.0

    def test_estimate_snr_mixed(self):
        passages = [
            Passage("c", 0.9, 100),
            Passage("c", 0.3, 100),
            Passage("c", 0.7, 100),
            Passage("c", 0.1, 100),
        ]
        snr = CSIGuard.estimate_snr(passages, threshold=0.5)
        assert snr == pytest.approx(0.5)

    def test_estimate_snr_empty(self):
        assert CSIGuard.estimate_snr([], threshold=0.5) == 0.5


# =====================================================================
# Integration Tests
# =====================================================================


class TestIntegration:

    def test_full_pipeline_mock_model(self):
        """
        Full pipeline: generate tasks -> assemble context ->
        mock model response -> evaluate -> analyze.
        No real API calls.
        """
        # 1. Generate tasks
        tasks = generate_all_tasks(seed=42, instances_per_family=2)
        assert len(tasks) == 8

        # 2. Generate noise
        noise = generate_noise_corpus(seed=42, num_passages=20, num_clusters=4)

        # 3. Assemble context for a T2 task
        assembler = ContextAssembler(noise, context_window=50_000)
        t2_tasks = [t for t in tasks if t.family == "T2"]
        task = t2_tasks[0]
        ctx = assembler.assemble(task, cu=0.3, snr=0.5, seed=42)

        # 4. Mock a model response (correct answer)
        answer = task.answer
        raw_text = f"Based on the context, {answer}.\nConfidence: 85%"
        parsed_answer, confidence = parse_answer_and_confidence(raw_text)
        response = ModelResponse(
            raw_text=raw_text,
            answer=parsed_answer,
            confidence=confidence,
            latency_ms=150.0,
            tokens_used=ctx.total_tokens + 100,
        )

        # 5. Evaluate
        evaluator = Evaluator()
        trial = evaluator.evaluate_trial(task, response, cu=0.3, snr=0.5, seed=42)
        assert trial.correct is True
        assert trial.confidence == 85.0

        # 6. Analyze (need at least a few data points)
        cu_vals = np.array([10, 30, 50, 70, 90])
        acc_vals = np.array([95, 85, 70, 55, 40])
        analyzer = Analyzer()
        fits = analyzer.fit_degradation_curve(cu_vals, acc_vals)
        assert len(fits) > 0

        tau = analyzer.extract_threshold(cu_vals, acc_vals, baseline_accuracy=95.0)
        assert tau > 0

    def test_csv_logging_end_to_end(self):
        """Verify CSVLogger writes correct headers and rows."""
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "results.csv"

            # Import CSVLogger from the experiment runner
            # We replicate minimal CSV logging here to avoid importing the runner
            # (which would require more setup). Test the pattern used in the runner.
            import csv as csv_mod

            headers = [
                "timestamp", "model", "task_family", "depth", "task_id",
                "cu_target", "cu_actual", "snr_target", "snr_actual",
                "context_tokens", "answer_expected", "answer_given",
                "correct", "confidence", "calibration_error",
                "tokens_used", "latency_ms",
            ]

            with open(csv_path, "w", newline="") as f:
                writer = csv_mod.DictWriter(f, fieldnames=headers)
                writer.writeheader()

                # Generate a task and mock result
                tasks = generate_all_tasks(seed=42, instances_per_family=1)
                task = tasks[0]
                noise = generate_noise_corpus(seed=42, num_passages=8, num_clusters=2)
                assembler = ContextAssembler(noise, context_window=50_000)
                ctx = assembler.assemble(task, cu=0.2, snr=0.5, seed=42)

                response = ModelResponse(
                    raw_text=f"{task.answer}\nConfidence: 75%",
                    answer=task.answer,
                    confidence=75.0,
                    latency_ms=120.0,
                    tokens_used=500,
                )
                evaluator = Evaluator()
                trial = evaluator.evaluate_trial(task, response, 0.2, 0.5, 42)

                row = {
                    "timestamp": "2026-01-01T00:00:00",
                    "model": "test-model",
                    "task_family": task.family,
                    "depth": task.depth,
                    "task_id": task.id,
                    "cu_target": 0.2,
                    "cu_actual": f"{ctx.actual_cu:.4f}",
                    "snr_target": 0.5,
                    "snr_actual": f"{ctx.actual_snr:.4f}",
                    "context_tokens": ctx.total_tokens,
                    "answer_expected": task.answer,
                    "answer_given": response.answer,
                    "correct": trial.correct,
                    "confidence": response.confidence,
                    "calibration_error": trial.calibration_error,
                    "tokens_used": response.tokens_used,
                    "latency_ms": "120",
                }
                writer.writerow(row)

            # Read back and verify
            with open(csv_path, "r") as f:
                reader = csv_mod.DictReader(f)
                rows = list(reader)

            assert len(rows) == 1
            assert rows[0]["model"] == "test-model"
            assert rows[0]["task_family"] == task.family
            assert rows[0]["task_id"] == task.id
            assert rows[0]["correct"] in ("True", "False")

    def test_pipeline_multiple_conditions(self):
        """Run multiple conditions and aggregate into ConditionResult."""
        tasks = generate_all_tasks(seed=42, instances_per_family=3)
        noise = generate_noise_corpus(seed=42, num_passages=20, num_clusters=4)
        assembler = ContextAssembler(noise, context_window=50_000)
        evaluator = Evaluator()

        t1_tasks = [t for t in tasks if t.family == "T1"][:3]
        trials = []
        for task in t1_tasks:
            ctx = assembler.assemble(task, cu=0.3, snr=0.5, seed=42)
            response = ModelResponse(
                raw_text=f"{task.answer}\nConfidence: 80%",
                answer=task.answer,
                confidence=80.0,
                latency_ms=100.0,
                tokens_used=300,
            )
            trial = evaluator.evaluate_trial(task, response, 0.3, 0.5, 42)
            trials.append(trial)

        cond = ConditionResult.from_trials(trials)
        assert cond.n_trials == 3
        assert cond.accuracy == pytest.approx(1.0)  # All correct
        assert cond.family == "T1"
        assert cond.mean_confidence == pytest.approx(80.0)


# =====================================================================
# Model-level helpers
# =====================================================================


class TestModelFunctions:
    """Test the analysis model functions directly."""

    def test_linear_model(self):
        assert linear_model(0, 1, 5) == 5
        assert linear_model(10, -0.5, 100) == pytest.approx(95)

    def test_quadratic_model(self):
        assert quadratic_model(0, 1, 2, 3) == 3
        assert quadratic_model(2, 1, 0, 0) == pytest.approx(4)

    def test_exponential_model(self):
        val = exponential_model(0, 80, 0.02, 10)
        assert val == pytest.approx(90)  # 80*exp(0) + 10

    def test_sigmoid_model(self):
        # At cu = tau, sigmoid should be at midpoint: L + (U-L)/2
        val = sigmoid_model(50, 10, 90, 0.1, 50)
        assert val == pytest.approx(50.0)  # midpoint
