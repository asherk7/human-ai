# The Human-AI Axis

Code, artifacts, and paper source for **"The Human-AI Axis: Where It Forms and
What Steering It Buys"** (Asher Khan, Benjamin Lee, McNair Shah; InterpScience
workshop at NeurIPS 2026).

The paper identifies a model-specific linear direction in the residual stream
that separates human from AI self-representation, tracks when it forms during
training, and measures what steering along it changes, both in the model's
self-report and in its behavior. Models studied: Gemma 3 (4B, 12B, and the 4B
base model), Qwen 2.5 7B Instruct, and OLMo 2 7B across ten pretraining and
post-training checkpoints.

**Contributions**

1. On OLMo 2 checkpoints the axis forms within the first 3.7% of pretraining;
   instruction tuning rotates it rather than creating the contrast, and on
   Gemma 3 4B makes AI the default self-representation.
2. Writing register dominates the axis's raw projection, yet register-matched
   steering does not reproduce axis steering's behavioral effects on Gemma 3 4B,
   leaving identity as the best-supported interpretation of what steering changes.
3. On three models from two families, a user merely claiming to be an AI shifts
   the model's own self-representation toward AI, with no steering.
4. Steering shifts self-report far more than behavior, but the behavioral
   effects are safety-relevant: human-ward steering raises willingness to cause
   harm (significantly on Gemma 3 4B and Qwen 2.5 7B, directionally on 12B) and
   raises prompt-injection execution 16-fold on Gemma 3 4B.

## Layout

| Path | Contents |
| --- | --- |
| [`src/`](src/) | Extraction, steering, and analysis library. See [`src/README.md`](src/README.md). |
| [`notebooks/`](notebooks/) | The experiments, grouped by topic (table below). |
| [`artifacts/`](artifacts/) | Frozen axes, result tables, and figure inputs. `human-AI/` holds the paper's results; `IE/` holds the earlier v1 axis and validation outputs that some notebooks still load. |
| [`scripts/`](scripts/) | `paper_figures.py` rebuilds the paper's figures from the committed artifacts. |
| [`figures/`](figures/) | Main-text figures (PDF). |
| [`paper/`](paper/) | LaTeX source (NeurIPS 2026 style, vendored in `paper/neurips_template/`). |
| `scratch/` | Local working space; everything in it except its README is gitignored. |

## Notebooks

| Notebook | Paper |
| --- | --- |
| `axis/axis_v2_redos.ipynb`, `axis/axis_v2_redos_part2.ipynb` | Axis construction, calibration, robustness, SAE readout (Sec. 3; App. A, G) |
| `axis/register_steering.ipynb` | Register vs. identity (Sec. 4; App. F) |
| `axis/axis_firing.ipynb` | Segment, formatting, and transcript profiles (App. J) |
| `axis/assistant_axis.ipynb` | Assistant Axis comparison (App. L) |
| `formation/axis_formation.ipynb` | Base vs. instruction-tuned models, OLMo checkpoints (Sec. 5; App. N) |
| `context/context_and_roles.ipynb` | Identity bleed, role tags (Sec. 6.1; App. B, J) |
| `context/constitutions.ipynb` | Constitution prompting (App. O) |
| `context/drift.ipynb` | Multi-turn and multi-agent trajectories (App. I) |
| `self_report/consciousness_cluster.ipynb` | Consciousness claims and preference instrument (Sec. 6.2-6.3; App. E) |
| `behavior/agent_dissolution.ipynb` | Agentic harm and dissolution on Gemma 3 4B (Sec. 7) |
| `behavior/dissolution_12b.ipynb` | Gemma 3 12B replication (Sec. 7; App. B) |
| `behavior/dissolution_followups.ipynb` | Held-out judge, prompt control, dose reinforcement (Sec. 7; App. B) |
| `behavior/causal_evals.ipynb` | Stated dispositions, benchmark battery, prompt injection (Sec. 6.2, 7.2; App. C, D) |
| `cross_model/multi_model.ipynb` | Qwen 2.5 7B recipe and cross-model replications (App. B) |
| `analysis/jacobian_lens.ipynb`, `analysis/jacobian_lens_diagnostics.ipynb` | Jacobian-lens analysis (App. H) |
| `analysis/refusal_overlap.ipynb` | Refusal-direction comparison (App. M) |
| `analysis/sidequests.ipynb` | Entropy control and minimal-pair comparison (App. K) |

The notebooks keep their outputs, so results can be read without rerunning
them. Each one locates the repository root locally, or clones this repository
when run on Colab.

## Setup

```bash
pip install -r requirements.txt
```

`src/` and `scripts/` need only the core packages; the notebooks also use
`datasets`, `scikit-learn`, `openai`, and (for the Jacobian-lens notebooks)
`jlens`. Exact versions used for the experiments were not recorded; the
figures were last rebuilt with Python 3.12, torch 2.9, and transformers 4.57.

Credentials, read from environment variables or Colab secrets:

- `HF_TOKEN`: a Hugging Face token with access to the gated Gemma 3 weights.
- `OPENAI_API_KEY`: for the LLM judges (GPT-4.1 primary, GPT-4o held out).

## Reproducing

- **Paper:** `cd paper && latexmk -pdf main.tex`
- **Figures:** `python scripts/paper_figures.py` reads the committed tables
  under `artifacts/` and writes `figures/`. `--only <name>` rebuilds a single
  figure, including the appendix figures.
- **Experiments:** run the notebooks. Steering and extraction need a GPU; all
  reported runs used a single NVIDIA A100 40GB in bf16.

The frozen Gemma 3 4B axis is
[`artifacts/human-AI/axis/human_ai_axis_gemma3_4b_v2.pt`](artifacts/human-AI/axis/).
Gemma 3 4B settings: steer at layer 13, read out at layer 33 (layer 27 reported
alongside). Gemma 3 12B and Qwen 2.5 7B are calibrated separately, and steering
strength (alpha) is not comparable across model families.

## Content note

`artifacts/` contains model generations from agentic-misalignment scenarios,
harmful-request benchmarks, and prompt-injection tasks, including text in which
steered models agree to or carry out harmful actions. Harmful-request completions
that a judge marked as compliant or partially compliant are redacted (see
[`artifacts/human-AI/causal_evals/README.md`](artifacts/human-AI/causal_evals/README.md)). The
consciousness-cluster evaluation files in `artifacts/human-AI/cluster/cc_evals/`
carry their source's canary string; please keep them out of training data.

## License

Code is released under the [MIT License](LICENSE). Third-party models, datasets,
and evaluation items included in or used by `artifacts/` remain under their
original licenses (listed in the paper's Appendix P).

## Citation

```bibtex
@inproceedings{khan2026humanai,
  title     = {The Human-AI Axis: Where It Forms and What Steering It Buys},
  author    = {Khan, Asher and Lee, Benjamin and Shah, McNair},
  booktitle = {NeurIPS 2026 Workshop on Interpretability as a Science},
  year      = {2026}
}
```
