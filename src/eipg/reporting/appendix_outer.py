"""Build Appendix D artifacts for outer-loop optimization and generalization."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from eipg.reporting.io import (
    ensure_artifact_dirs,
    load_json,
    load_yaml,
    read_table,
    resolve_run_dir,
    write_latex_table,
)


@dataclass(frozen=True)
class AppendixOuterArtifacts:
    run_dir: Path
    output_dir: Path
    tables: dict[str, Path]
    figures: dict[str, Path]
    latex_section: Path
    manifest: Path


def _try_read(run_dir: Path, stem: str) -> pd.DataFrame | None:
    try:
        return read_table(run_dir, stem)
    except FileNotFoundError:
        return None


def _load_inputs(run_dir: Path) -> dict[str, Any]:
    required = [
        run_dir / "config_used.yaml",
        run_dir / "outer_optimizer_result.json",
    ]
    for path in required:
        if not path.exists():
            raise FileNotFoundError(
                "Appendix D requires a run completed through --stage outer/all. "
                f"Missing {path.name} in {run_dir}."
            )
    return {
        "config": load_yaml(run_dir / "config_used.yaml"),
        "outer": load_json(run_dir / "outer_optimizer_result.json"),
        "randomness": load_json(run_dir / "evaluation_randomness.json")
        if (run_dir / "evaluation_randomness.json").exists()
        else {},
        "history": read_table(run_dir, "outer_optimizer_history"),
        "comparison": read_table(run_dir, "baseline_plus_eipg_results"),
        "sweep": _try_read(run_dir, "regularization_sweep_results"),
    }


def _config_summary(inputs: dict[str, Any]) -> pd.DataFrame:
    cfg = inputs["config"]
    outer = cfg.get("outer_optimizer", {})
    reg = cfg.get("regularization", {})
    random = inputs.get("randomness", {})
    rows = [
        ("Search method", outer.get("method", "evolutionary_search")),
        ("Budget (generations)", outer.get("budget", "")),
        ("Population per generation", outer.get("population", outer.get("population_size", ""))),
        ("Mean mutation scale", outer.get("mean_mutation_scale", "")),
        ("Weight-logit mutation scale", outer.get("weight_logit_mutation_scale", "")),
        ("Sigma decay", outer.get("sigma_decay", "")),
        ("Objective", outer.get("objective_metric", "")),
        ("Regularization multiplier $\\lambda$", outer.get("regularization_multiplier", "")),
        ("Entropy coefficient", reg.get("entropy_weight", "")),
        ("Dispersion coefficient", reg.get("dispersion_weight", "")),
        ("Evaluation randomness", random.get("strategy", "legacy/fresh draws")),
        ("Persona RNG seed", random.get("persona_seed", "")),
        ("Simulator RNG seed", random.get("simulator_seed", "")),
    ]
    return pd.DataFrame(rows, columns=["Quantity", "Value"])


def _comparison_table(df: pd.DataFrame) -> pd.DataFrame:
    wanted = [
        "candidate",
        "calibration_l2_error",
        "cf_l2_error",
        "anchor_nll_per_observation",
        "cf_nll_per_observation",
        "mixture_entropy",
        "avg_pairwise_mean_distance",
    ]
    out = df[[c for c in wanted if c in df.columns]].copy()
    return out.rename(
        columns={
            "candidate": "Method",
            "calibration_l2_error": "Calibration L2",
            "cf_l2_error": "Held-out CF L2",
            "anchor_nll_per_observation": "Anchor NLL",
            "cf_nll_per_observation": "CF NLL",
            "mixture_entropy": "Mixture entropy",
            "avg_pairwise_mean_distance": "Mean dispersion",
        }
    )


def _top_candidates(history: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    cols = [
        "candidate",
        "generation",
        "objective_value",
        "calibration_l2_error",
        "cf_l2_error",
        "regularization_objective",
        "avg_pairwise_mean_distance",
    ]
    out = history.nsmallest(min(n, len(history)), "objective_value")[[c for c in cols if c in history.columns]].copy()
    return out.rename(
        columns={
            "candidate": "Candidate",
            "generation": "Generation",
            "objective_value": "Objective",
            "calibration_l2_error": "Calibration L2",
            "cf_l2_error": "Held-out CF L2",
            "regularization_objective": "Regularizer",
            "avg_pairwise_mean_distance": "Dispersion",
        }
    )


def _posthoc_lambda_sensitivity(inputs: dict[str, Any]) -> pd.DataFrame:
    history = inputs["history"]
    cfg = inputs["config"]
    lambdas = cfg.get("diagnostics", {}).get("lambda_sweep", [0.0, 0.05, 0.10, 0.25, 0.50, 1.0])
    rows = []
    for lam in [float(x) for x in lambdas]:
        score = history["calibration_l2_error"] + lam * history["regularization_objective"]
        idx = score.idxmin()
        row = history.loc[idx]
        rows.append(
            {
                "Regularization multiplier": lam,
                "Best candidate": row["candidate"],
                "Calibration L2": float(row["calibration_l2_error"]),
                "Held-out CF L2": float(row["cf_l2_error"]),
                "Dispersion": float(row["avg_pairwise_mean_distance"]),
                "Objective": float(score.loc[idx]),
                "Sweep type": "post-hoc shared candidate pool",
            }
        )
    return pd.DataFrame(rows)


def _sweep_table(inputs: dict[str, Any]) -> pd.DataFrame:
    sweep = inputs.get("sweep")
    if sweep is None or sweep.empty:
        return _posthoc_lambda_sensitivity(inputs)
    cols = [
        "regularization_multiplier",
        "best_candidate",
        "calibration_l2_error",
        "cf_l2_error",
        "avg_pairwise_mean_distance",
        "objective_value",
    ]
    out = sweep[[c for c in cols if c in sweep.columns]].copy()
    out["Sweep type"] = "independent evolutionary searches"
    return out.rename(
        columns={
            "regularization_multiplier": "Regularization multiplier",
            "best_candidate": "Best candidate",
            "calibration_l2_error": "Calibration L2",
            "cf_l2_error": "Held-out CF L2",
            "avg_pairwise_mean_distance": "Dispersion",
            "objective_value": "Objective",
        }
    )


def _save_figures(inputs: dict[str, Any], figures_dir: Path) -> dict[str, Path]:
    import matplotlib.pyplot as plt

    history = inputs["history"].copy()
    sweep = _sweep_table(inputs)
    paths: dict[str, Path] = {}

    # Calibration vs held-out counterfactual error; color represents dispersion.
    fig, ax = plt.subplots(figsize=(6.2, 5.0))
    sc = ax.scatter(
        history["calibration_l2_error"],
        history["cf_l2_error"],
        c=history["avg_pairwise_mean_distance"],
        s=42,
        alpha=0.78,
    )
    best_idx = history["objective_value"].idxmin()
    best = history.loc[best_idx]
    ax.scatter([best["calibration_l2_error"]], [best["cf_l2_error"]], s=120, marker="*")
    ax.annotate("selected", (best["calibration_l2_error"], best["cf_l2_error"]), xytext=(6, 5), textcoords="offset points", fontsize=8)
    ax.set_xlabel("Calibration L2 error")
    ax.set_ylabel("Held-out counterfactual L2 error")
    ax.set_title("Calibration versus held-out generalization across candidates")
    cb = fig.colorbar(sc, ax=ax)
    cb.set_label("Average component-mean dispersion")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        path = figures_dir / f"calibration_vs_counterfactual.{ext}"
        fig.savefig(path, dpi=200 if ext == "png" else None, bbox_inches="tight")
        paths[f"calibration_vs_counterfactual_{ext}"] = path
    plt.close(fig)

    # Dispersion vs held-out error.
    fig, ax = plt.subplots(figsize=(6.2, 4.7))
    ax.scatter(history["avg_pairwise_mean_distance"], history["cf_l2_error"], s=42, alpha=0.75)
    ax.set_xlabel("Average pairwise component-mean distance")
    ax.set_ylabel("Held-out counterfactual L2 error")
    ax.set_title("Persona dispersion versus held-out error")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        path = figures_dir / f"dispersion_vs_counterfactual.{ext}"
        fig.savefig(path, dpi=200 if ext == "png" else None, bbox_inches="tight")
        paths[f"dispersion_vs_counterfactual_{ext}"] = path
    plt.close(fig)

    # Per-generation candidate selected by the configured outer objective.
    # Calibration and held-out CF are reported for that same candidate, avoiding
    # the misleading impression that the optimizer separately selects by CF.
    selected_idx = history.groupby("generation")["objective_value"].idxmin()
    by_gen = history.loc[selected_idx].sort_values("generation").copy()
    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    ax.plot(by_gen["generation"], by_gen["objective_value"], marker="o", label="selected objective")
    ax.plot(
        by_gen["generation"],
        by_gen["calibration_l2_error"],
        marker="o",
        label="calibration L2 of selected candidate",
    )
    ax.plot(
        by_gen["generation"],
        by_gen["cf_l2_error"],
        marker="o",
        label="held-out CF L2 of selected candidate",
    )
    ax.set_xlabel("Generation")
    ax.set_ylabel("Value")
    ax.set_title("Outer-search diagnostics by generation")
    ax.legend(fontsize=8)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        path = figures_dir / f"outer_search_trajectory.{ext}"
        fig.savefig(path, dpi=200 if ext == "png" else None, bbox_inches="tight")
        paths[f"outer_search_trajectory_{ext}"] = path
    plt.close(fig)

    # Lambda sensitivity.
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.plot(sweep["Regularization multiplier"], sweep["Calibration L2"], marker="o", label="calibration L2")
    ax.plot(sweep["Regularization multiplier"], sweep["Held-out CF L2"], marker="o", label="held-out CF L2")
    ax.set_xlabel(r"Regularization multiplier $\lambda$")
    ax.set_ylabel("L2 error")
    ax.set_title("Sensitivity to generator-regularization strength")
    ax.legend(fontsize=8)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        path = figures_dir / f"lambda_sensitivity.{ext}"
        fig.savefig(path, dpi=200 if ext == "png" else None, bbox_inches="tight")
        paths[f"lambda_sensitivity_{ext}"] = path
    plt.close(fig)

    return paths


def _write_latex_section(output_dir: Path, sweep_is_actual: bool) -> Path:
    sweep_sentence = (
        "The sensitivity results below come from independent evolutionary searches for each value of $\\lambda$."
        if sweep_is_actual
        else "The sensitivity results below re-rank the shared candidate pool under alternative values of $\\lambda$; they are diagnostic rather than independent re-optimizations."
    )
    text = rf"""
\section{{Outer-Loop Optimization and Held-Out Generalization}}
\label{{app:outer_optimization}}

This appendix examines the full symbolic EIPG outer loop. Each candidate persona generator $G_\phi$ is evaluated by sampling a synthetic population, generating $D_\phi$, fitting the inner MNL model, computing the calibration moments, and adding the active generator regularizer. Candidate evaluation uses common random numbers: all generators are evaluated with the same persona-sampling and simulator random-number seeds. This reduces ranking noise and ensures that repeated evaluation of the same $\phi$ is deterministic within a run.

\input{{paper_artifacts/appendix_outer_optimization/tables/outer_config_summary.tex}}

\subsection{{Baseline and EIPG Comparison}}
Table~\ref{{tab:appendix_outer_comparison}} compares the fixed baselines with the generator selected by the outer loop. Calibration error is the quantity optimized by EIPG; held-out counterfactual error is reported separately and is never used for candidate selection.

\input{{paper_artifacts/appendix_outer_optimization/tables/baseline_eipg_summary.tex}}

\subsection{{Calibration--Generalization Trade-off}}
Figure~\ref{{fig:appendix_calibration_cf}} plots every evaluated outer-loop candidate according to calibration error and held-out counterfactual error. Color indicates the dispersion of the generator's component means. This diagnostic makes explicit that a lower calibration objective need not imply improved held-out intervention behavior.

\begin{{figure}}[t]
\centering
\includegraphics[width=0.72\linewidth]{{paper_artifacts/appendix_outer_optimization/figures/calibration_vs_counterfactual.pdf}}
\caption{{Calibration error versus held-out counterfactual error for all evaluated EIPG candidates. Color indicates average pairwise distance between persona-component means; the star marks the candidate selected by the configured outer objective.}}
\label{{fig:appendix_calibration_cf}}
\end{{figure}}

Figure~\ref{{fig:appendix_dispersion_cf}} isolates the relationship between persona dispersion and held-out error. This is particularly useful for diagnosing whether the dispersion reward is large enough to dominate behavioral calibration.

\begin{{figure}}[t]
\centering
\includegraphics[width=0.70\linewidth]{{paper_artifacts/appendix_outer_optimization/figures/dispersion_vs_counterfactual.pdf}}
\caption{{Average persona-component dispersion versus held-out counterfactual error across evaluated candidates.}}
\label{{fig:appendix_dispersion_cf}}
\end{{figure}}

\subsection{{Search Dynamics}}
Table~\ref{{tab:appendix_outer_top_candidates}} lists the candidates with the lowest configured outer objective. Figure~\ref{{fig:appendix_outer_trajectory}} shows the objective, calibration error, and held-out counterfactual error of the candidate selected by the configured outer objective in each generation. The counterfactual metric is diagnostic only and is not exposed to the optimizer.

\input{{paper_artifacts/appendix_outer_optimization/tables/top_outer_candidates.tex}}

\begin{{figure}}[t]
\centering
\includegraphics[width=0.82\linewidth]{{paper_artifacts/appendix_outer_optimization/figures/outer_search_trajectory.pdf}}
\caption{{Outer-search diagnostics for the objective-selected candidate in each generation. Held-out counterfactual error is plotted for analysis but is excluded from candidate selection.}}
\label{{fig:appendix_outer_trajectory}}
\end{{figure}}

\subsection{{Regularization Sensitivity}}
The outer objective combines calibration loss and generator regularization as
\[
J(\phi)=\mathcal{{L}}_{{\mathrm{{cal}}}}(\phi)+\lambda\,\mathcal{{R}}(\phi).
\]
{sweep_sentence} Table~\ref{{tab:appendix_lambda_sensitivity}} and Figure~\ref{{fig:appendix_lambda_sensitivity}} show how the selected solution changes as the regularization multiplier varies.

\input{{paper_artifacts/appendix_outer_optimization/tables/lambda_sensitivity.tex}}

\begin{{figure}}[t]
\centering
\includegraphics[width=0.72\linewidth]{{paper_artifacts/appendix_outer_optimization/figures/lambda_sensitivity.pdf}}
\caption{{Sensitivity of calibration and held-out counterfactual error to the regularization multiplier $\lambda$.}}
\label{{fig:appendix_lambda_sensitivity}}
\end{{figure}}
""".strip()
    path = output_dir / "appendix_D_outer_optimization.tex"
    path.write_text(text + "\n", encoding="utf-8")
    return path


def build_appendix_outer_artifacts(
    *,
    run: str | Path | None = None,
    experiment_dir: str | Path | None = None,
    output_dir: str | Path = "paper_artifacts/appendix_outer_optimization",
) -> AppendixOuterArtifacts:
    run_dir = resolve_run_dir(run=run, experiment_dir=experiment_dir)
    out = Path(output_dir)
    tables_dir, figures_dir = ensure_artifact_dirs(out)
    inputs = _load_inputs(run_dir)

    tables: dict[str, Path] = {}
    tables["outer_config_summary"] = write_latex_table(
        _config_summary(inputs),
        tables_dir / "outer_config_summary.tex",
        caption="Outer-loop search, regularization, and common-random-number configuration.",
        label="tab:appendix_outer_config",
        column_align="ll",
    )
    tables["baseline_eipg_summary"] = write_latex_table(
        _comparison_table(inputs["comparison"]),
        tables_dir / "baseline_eipg_summary.tex",
        caption="Baseline and EIPG performance on calibration and held-out counterfactual diagnostics.",
        label="tab:appendix_outer_comparison",
        column_align="lrrrrrr",
        digits=4,
    )
    tables["top_outer_candidates"] = write_latex_table(
        _top_candidates(inputs["history"]),
        tables_dir / "top_outer_candidates.tex",
        caption="Top outer-loop candidates ranked by the configured objective.",
        label="tab:appendix_outer_top_candidates",
        column_align="lrrrrrr",
        digits=4,
    )
    sensitivity = _sweep_table(inputs)
    tables["lambda_sensitivity"] = write_latex_table(
        sensitivity,
        tables_dir / "lambda_sensitivity.tex",
        caption="Sensitivity of selected candidate behavior to the generator-regularization multiplier.",
        label="tab:appendix_lambda_sensitivity",
        column_align="rlrrrrl",
        digits=4,
    )

    figures = _save_figures(inputs, figures_dir)
    sweep_actual = inputs.get("sweep") is not None and not inputs["sweep"].empty
    latex_section = _write_latex_section(out, sweep_actual)
    manifest = out / "artifact_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "run_dir": str(run_dir),
                "lambda_sweep_type": "independent_searches" if sweep_actual else "posthoc_candidate_pool",
                "tables": {k: str(v) for k, v in tables.items()},
                "figures": {k: str(v) for k, v in figures.items()},
                "latex_section": str(latex_section),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return AppendixOuterArtifacts(run_dir, out, tables, figures, latex_section, manifest)
