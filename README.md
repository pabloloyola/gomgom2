# EIPG reviewer code

Minimal, paper-facing implementation of **Economically-Informed Persona Generators (EIPG)**.

This archive contains the code paths needed to inspect or reproduce the paper's three evaluation regimes, while omitting notebooks, manuscript files, generated outputs, internal notes, and superseded experiments.

## Contents

- `src/eipg/`: shared persona-generator, simulator, economic-model, calibration, and search code.
- `scripts/`: reviewer-relevant experiment drivers and freeze/finalization utilities.
- `configs/`: frozen/paper-aligned configurations for those drivers.
- `tests/`: compact tests for the core mechanisms.

## Install

Python 3.11 or 3.12:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
pytest -q
```

For local Hugging Face experiments:

```bash
pip install -e '.[local-llm]'
```

For GEPA:

```bash
pip install -e '.[local-llm,gepa]'
```

## 1. Controlled synthetic benchmark

This regime is fully self-contained.

Quick smoke check:

```bash
python scripts/run_controlled_synthetic.py \
  --config configs/controlled_synthetic_smoke.yaml \
  --stage all
```

Paper-aligned controlled pipeline:

```bash
python scripts/run_controlled_synthetic.py \
  --config configs/controlled_synthetic_paperlike.yaml \
  --stage all

python scripts/run_multi_intervention_identification.py \
  --config configs/multi_intervention_identification.yaml

python scripts/run_multi_intervention_eipg_comparison.py \
  --config configs/multi_intervention_eipg_comparison.yaml

python scripts/run_frozen_symbolic_final_test.py \
  --config configs/eipg_frozen_symbolic.yaml
```

The homogeneous-vs-latent-class diagnostic is in `scripts/run_inner_model_comparison.py`.

## 2. Natural-language / cross-model calibration

These experiments require the corresponding local Hugging Face model weights and sufficient accelerator memory. The model keys in the frozen protocol are `gemma3_4b`, `gemma3_12b`, `qwen35_9b`, and `phi4_14b`.

For each model, run the transfer gate and calibration:

```bash
python scripts/run_cross_model_anchored_calibration.py \
  --config configs/cross_model_anchored_core.yaml \
  --model-key <model-key> \
  --stage both
```

After all four calibration runs finish, freeze the cohort before revealing held-out contexts:

```bash
python scripts/finalize_cross_model_calibration.py \
  --config configs/cross_model_anchored_core.yaml
```

Then evaluate each frozen model:

```bash
python scripts/run_cross_model_anchored_heldout.py \
  --config configs/cross_model_anchored_core.yaml \
  --model-key <model-key>
```

The archive also contains the exact screen-confirm, random-subset, residual-prior LinUCB, history-aware LLM, and GEPA paths used in the finite-query comparisons. The adaptive and GEPA protocols deliberately require their corresponding `finalize_*.py` scripts before held-out evaluation; this preserves the sequential freeze discipline described in the paper.

## 3. Public Nutri2Cycle benchmark

Raw respondent-level data are not redistributed. The paper uses the public Nutri2Cycle discrete-choice data, Zenodo DOI **10.5281/zenodo.8338014**. After downloading the source CSV:

```bash
python scripts/prepare_public_food_benchmark.py \
  --config configs/public_food_benchmark.yaml \
  --csv /path/to/source.csv

python scripts/run_public_food_search.py \
  --config configs/public_food_benchmark.yaml \
  --csv /path/to/source.csv

python scripts/evaluate_public_food_final.py \
  --config configs/public_food_benchmark.yaml \
  --csv /path/to/source.csv \
  --selected-dir <search-output-dir> \
  --outdir <evaluation-output-dir>
```

`scripts/build_public_food_reference.py` fits the direct-human reference model. The frozen public YAML retains a few provenance-note paths from the full research repository; those are metadata only and are not execution dependencies for the included benchmark scripts.

## Paper-to-code map

| Paper evidence | Main code |
|---|---|
| Persona generator / symbolic simulator | `src/eipg/personas/`, `src/eipg/simulators/` |
| Inner MNL / latent-class diagnostic | `src/eipg/econ/`, `run_inner_model_comparison.py` |
| Behavioral moment calibration | `src/eipg/objectives/calibration.py` |
| Intervention identification | `run_multi_intervention_identification.py` |
| Frozen symbolic test | `run_frozen_symbolic_final_test.py` |
| Anchored text objective | `run_llm_prompt_refinement_calibration_v2.py` |
| Cross-model calibration | `run_cross_model_anchored_calibration.py` |
| Screen-confirm / LinUCB | `derive_cross_model_screen_confirm.py`, `derive_cross_model_residual_linucb.py` |
| GEPA stress test | `run_cross_model_gepa_calibration.py` |
| Public human-choice validation | `prepare_public_food_benchmark.py`, `run_public_food_search.py`, `evaluate_public_food_final.py` |

