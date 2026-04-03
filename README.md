# ContextStress

**Benchmarking Phase Transitions in Long-Context LLM Reasoning**

ContextStress is a diagnostic benchmark for measuring *when* and *how* LLM reasoning collapses under increasing context load. It provides the **Effective Context Capacity (ECC)** framework and the **Context Saturation Index (CSI)** for runtime monitoring.

**Paper:** [arXiv preprint (coming soon)]

## Quick Start

```bash
pip install -e ".[all]"

# Generate benchmark data
python -m contextstress generate-tasks --output-dir data/
python -m contextstress generate-noise --output-dir data/

# Run on GPT-4o-mini
export OPENAI_API_KEY=sk-...
python -m contextstress run --model openai --output-dir results/

# Run on Claude
export ANTHROPIC_API_KEY=sk-ant-...
python -m contextstress run --model claude --output-dir results/

# Or use the experiment runner with full control
python experiments/run_experiment.py \
  --model gpt-4o-mini \
  --families T1 T2 T3 T4 \
  --cu-levels 0.05 0.15 0.30 0.50 0.75 0.95 \
  --snr 0.5 \
  --tasks-per-condition 20
```

## What ContextStress Measures

Unlike benchmarks that report performance at fixed context lengths, ContextStress characterizes the **degradation function** across three axes:

| Axis | Levels | Purpose |
|------|--------|---------|
| Context Utilization (CU) | 5%, 15%, 30%, 50%, 75%, 95% | How full is the context window? |
| Signal-to-Noise Ratio (SNR) | 100%, 50%, 25% | How much of the context is relevant? |
| Reasoning Depth (d) | 1, 2, 3, 4 hops | How complex is the task? |

**72 conditions x 20 instances = 1,440 trials per model**

## Key Findings (GPT-4o-mini)

### The Collapse Is Real

| CU | T1 (d=1) | T2 (d=2) | T3 (d=4) | T4 (d=3) |
|----|----------|----------|----------|----------|
| 5% | 95% | **100%** | 100% | 10% |
| 15% | 85% | 65% | 100% | 15% |
| 30% | 85% | 20% | 75% | 25% |
| 50% | 85% | 15% | 45% | 0% |
| 75% | 70% | **0%** | 95% | 10% |
| 95% | 75% | **0%** | 70% | 0% |

*GPT-4o-mini, SNR=50%, n=20 per condition*

### The Silent Failure

The model reports **100% confidence** while achieving **0% accuracy** on two-hop reasoning at high context utilization. This calibration error of 100 percentage points makes confidence-based monitoring useless.

### The Saturation Threshold Law (Proposed)

```
tau(SNR, d) = tau_max * SNR^(gamma_0 + gamma_1 * d)
```

Predicts collapse points as a function of signal quality and reasoning depth. Validated qualitatively on GPT-4o-mini; multi-model validation is future work.

## CSI-Guard: Runtime Protection

```python
from contextstress import CSIGuard, SaturationThresholdLaw

law = SaturationThresholdLaw(tau_max=84.2, gamma_0=0.37, gamma_1=0.061,
                              r_squared=0.984, mae=1.8, rmse=2.1)

guard = CSIGuard(law=law, context_window=128_000, target_zone="Green")

zone = law.zone(cu=39.0, snr=0.5, d=2)
print(f"Operating zone: {zone}")  # "Red" -- reduce context!
```

## Custom Model Adapter

```python
from contextstress.adapters import ModelAdapter, ModelResponse

class MyModelAdapter(ModelAdapter):
    @property
    def name(self) -> str:
        return "my-model"

    @property
    def context_window(self) -> int:
        return 128_000

    def generate(self, query: str, context: str) -> ModelResponse:
        # Your inference code here
        ...
```

## Project Structure

```
contextstress/
├── __init__.py          # Package exports
├── __main__.py          # CLI entry point
├── adapters.py          # Model adapter interface + OpenAI/Anthropic
├── assembly.py          # Context assembly (CU, SNR control)
├── evaluate.py          # Exact-match scoring, calibration
├── analysis.py          # Curve fitting, threshold law, CSI
├── csi_guard.py         # Runtime context management (Algorithm 1)
├── tasks/
│   └── generator.py     # Synthetic task generation (400 instances)
└── noise_corpus/
    └── generator.py     # Noise corpus generation
experiments/
    └── run_experiment.py # Full experiment runner with budget tracking
tests/
    └── test_all.py       # 82 tests
paper/
    └── main.tex          # LaTeX paper
figures/
    └── *.pdf             # Publication figures from real data
```

## Tests

```bash
python -m pytest tests/ -v
# 82 passed
```

## Citation

```bibtex
@article{rupesh2026contextstress,
  title={Context Is Not Free: Phase Transitions in LLM Reasoning
         Under Context Scaling and the Saturation Threshold Law},
  author={Rupesh, Kollaikal},
  year={2026},
  url={https://github.com/kollaikal-rupesh/contextstress}
}
```

## License

MIT
