"""Paper artifacts for Appendix E: economic-model capacity diagnostic."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import json

import numpy as np
import pandas as pd

from .io import ensure_artifact_dirs, load_json, read_table, resolve_run_dir, write_latex_table


@dataclass(frozen=True)
class AppendixModelCapacityArtifacts:
    run_dir: Path
    output_dir: Path
    tables: dict[str, Path]
    figures: dict[str, Path]
    latex_section: Path
    manifest: Path


def _summary_table(summary: dict[str, Any]) -> pd.DataFrame:
    rows = []
    ordered = [
        ("direct_oracle", "Direct oracle simulator"),
        ("oracle_segmented_mnl", "Oracle segmented MNL"),
        ("oracle_through_mnl", "Oracle homogeneous MNL"),
    ]
    for label, display in ordered:
        if label not in summary:
            continue
        values = summary[label]
        rows.append(
            {
                "Diagnostic path": display,
                "Calibration L2": float(values["calibration_l2_error"]),
                "Calibration RMSE": float(values["calibration_rmse"]),
                "Held-out CF L2": float(values["cf_l2_error"]),
                "Held-out CF RMSE": float(values["cf_rmse"]),
            }
        )
    return pd.DataFrame(rows)


def _intervention_table(comparison: pd.DataFrame) -> pd.DataFrame:
    rows = comparison[comparison["moment_type"] == "intervention_response"].copy()
    wanted = [
        "short_name",
        "target",
        "direct_oracle_model",
        "oracle_segmented_mnl_model",
        "oracle_through_mnl_model",
        "static_model",
        "eipg_model",
    ]
    rows = rows[[c for c in wanted if c in rows.columns]].copy()
    return rows.rename(
        columns={
            "short_name": "Moment",
            "target": "Target",
            "direct_oracle_model": "Direct oracle",
            "oracle_segmented_mnl_model": "Oracle segmented MNL",
            "oracle_through_mnl_model": "Oracle homogeneous MNL",
            "static_model": "Static homogeneous MNL",
            "eipg_model": "EIPG homogeneous MNL",
        }
    )


def _distortion_table(comparison: pd.DataFrame, n: int = 12) -> pd.DataFrame:
    df = comparison.copy()
    df["homogeneous_excess_over_direct"] = (
        df["oracle_through_mnl_abs_error"] - df["direct_oracle_abs_error"]
    )
    if "oracle_segmented_mnl_abs_error" in df.columns:
        df["segmented_excess_over_direct"] = (
            df["oracle_segmented_mnl_abs_error"] - df["direct_oracle_abs_error"]
        )
        df["homogeneous_excess_over_segmented"] = (
            df["oracle_through_mnl_abs_error"] - df["oracle_segmented_mnl_abs_error"]
        )
    else:
        df["segmented_excess_over_direct"] = np.nan
        df["homogeneous_excess_over_segmented"] = np.nan

    cols = [
        "block",
        "moment_type",
        "short_name",
        "target",
        "direct_oracle_abs_error",
        "oracle_segmented_mnl_abs_error",
        "oracle_through_mnl_abs_error",
        "segmented_excess_over_direct",
        "homogeneous_excess_over_segmented",
    ]
    cols = [c for c in cols if c in df.columns]
    sort_col = "homogeneous_excess_over_segmented" if "oracle_segmented_mnl_abs_error" in df.columns else "homogeneous_excess_over_direct"
    out = df.sort_values(sort_col, ascending=False).head(n)[cols].copy()
    return out.rename(
        columns={
            "block": "Block",
            "moment_type": "Moment type",
            "short_name": "Moment",
            "target": "Target",
            "direct_oracle_abs_error": "Direct abs. error",
            "oracle_segmented_mnl_abs_error": "Segmented-MNL abs. error",
            "oracle_through_mnl_abs_error": "Homogeneous-MNL abs. error",
            "segmented_excess_over_direct": "Segmented minus direct",
            "homogeneous_excess_over_segmented": "Homogeneous minus segmented",
        }
    )


def _segmented_coefficients_table(coefficients: pd.DataFrame) -> pd.DataFrame:
    wanted = [
        "segment_id",
        "feature",
        "beta",
        "segment_observation_weight",
        "n_observations",
        "train_nll_per_observation",
    ]
    out = coefficients[[c for c in wanted if c in coefficients.columns]].copy()
    return out.rename(
        columns={
            "segment_id": "Segment",
            "feature": "Feature",
            "beta": "Coefficient",
            "segment_observation_weight": "Training share",
            "n_observations": "N observations",
            "train_nll_per_observation": "Train NLL / obs.",
        }
    )


def _save_figures(
    comparison: pd.DataFrame,
    summary: dict[str, Any],
    figures_dir: Path,
) -> dict[str, Path]:
    import matplotlib.pyplot as plt

    paths: dict[str, Path] = {}

    # Moment-level error relative to direct oracle.
    fig, ax = plt.subplots(figsize=(6.4, 5.1))
    x = comparison["direct_oracle_abs_error"].to_numpy(float)
    all_values = [x]
    if "oracle_segmented_mnl_abs_error" in comparison.columns:
        y_seg = comparison["oracle_segmented_mnl_abs_error"].to_numpy(float)
        ax.scatter(x, y_seg, s=42, alpha=0.78, marker="o", label="oracle segmented MNL")
        all_values.append(y_seg)
    y_hom = comparison["oracle_through_mnl_abs_error"].to_numpy(float)
    ax.scatter(x, y_hom, s=42, alpha=0.78, marker="x", label="oracle homogeneous MNL")
    all_values.append(y_hom)
    lo = float(min(arr.min(initial=0.0) for arr in all_values))
    hi = float(max(arr.max(initial=0.0) for arr in all_values))
    pad = max((hi - lo) * 0.05, 1e-4)
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], linestyle="--")
    ax.set_xlabel("Direct-oracle absolute moment error")
    ax.set_ylabel("Economic-model absolute moment error")
    ax.set_title("Moment distortion under segmented and homogeneous MNLs")
    ax.legend(fontsize=8)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        path = figures_dir / f"direct_oracle_vs_mnl_moment_error.{ext}"
        fig.savefig(path, dpi=200 if ext == "png" else None, bbox_inches="tight")
        paths[f"direct_oracle_vs_mnl_moment_error_{ext}"] = path
    plt.close(fig)

    # Substitution response comparison.
    sub = comparison[comparison["moment_type"] == "intervention_response"].copy()
    share_mask = sub["short_name"].str.contains("delta_alt_|delta_intervened_alt_", regex=True)
    sub = sub[share_mask].copy()
    if not sub.empty:
        x_pos = np.arange(len(sub))
        fig, ax = plt.subplots(figsize=(7.7, 4.8))
        ax.plot(x_pos, sub["target"], marker="o", label="target")
        ax.plot(x_pos, sub["direct_oracle_model"], marker="o", label="direct oracle")
        if "oracle_segmented_mnl_model" in sub.columns:
            ax.plot(
                x_pos,
                sub["oracle_segmented_mnl_model"],
                marker="o",
                label="oracle segmented MNL",
            )
        ax.plot(
            x_pos,
            sub["oracle_through_mnl_model"],
            marker="o",
            label="oracle homogeneous MNL",
        )
        if "eipg_model" in sub.columns:
            ax.plot(x_pos, sub["eipg_model"], marker="o", label="EIPG homogeneous MNL")
        ax.axhline(0.0, linewidth=0.8)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(
            sub["short_name"].str.replace("delta_", "", regex=False),
            rotation=25,
            ha="right",
        )
        ax.set_ylabel("Change in choice share")
        ax.set_title("Substitution responses across economic-model capacity")
        ax.legend(fontsize=8)
        fig.tight_layout()
        for ext in ("pdf", "png"):
            path = figures_dir / f"substitution_response_capacity.{ext}"
            fig.savefig(path, dpi=200 if ext == "png" else None, bbox_inches="tight")
            paths[f"substitution_response_capacity_{ext}"] = path
        plt.close(fig)

    # Aggregate calibration / CF diagnostic.
    ordered = [
        ("direct_oracle", "Direct oracle"),
        ("oracle_segmented_mnl", "Segmented MNL"),
        ("oracle_through_mnl", "Homogeneous MNL"),
    ]
    ordered = [(k, label) for k, label in ordered if k in summary]
    labels = [label for _, label in ordered]
    calibration = [summary[k]["calibration_l2_error"] for k, _ in ordered]
    cf = [summary[k]["cf_l2_error"] for k, _ in ordered]
    x_pos = np.arange(len(ordered))
    width = 0.34
    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    ax.bar(x_pos - width / 2, calibration, width, label="calibration L2")
    ax.bar(x_pos + width / 2, cf, width, label="held-out CF L2")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels)
    ax.set_ylabel("L2 error")
    ax.set_title("Economic-model capacity: direct, segmented, homogeneous")
    ax.legend(fontsize=8)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        path = figures_dir / f"capacity_error_summary.{ext}"
        fig.savefig(path, dpi=200 if ext == "png" else None, bbox_inches="tight")
        paths[f"capacity_error_summary_{ext}"] = path
    plt.close(fig)

    return paths


def _write_latex_section(output_dir: Path) -> Path:
    text = r"""
\section{Economic-Model Capacity Diagnostic}
\label{app:economic_model_capacity}

The controlled benchmark makes it possible to separate persona-generator error from approximation error introduced by the inner economic model. We compare three oracle paths. The \emph{direct oracle} uses the ground-truth random-utility simulator probabilities for the exact sampled persona--context pairs in the target data and therefore provides a finite-sample reference without fitting an econometric model. The \emph{oracle segmented MNL} uses the known synthetic mixture-component labels to fit one MNL per latent segment and routes each target task through its ground-truth segment. Finally, the usual \emph{oracle homogeneous MNL} fits a single coefficient vector to choices produced by the ground-truth persona generator. The segmented model is intentionally an oracle diagnostic rather than a deployable estimator: its role is to test whether discrete preference heterogeneity accounts for structure lost by the homogeneous inner model.

\input{paper_artifacts/appendix_model_capacity/tables/capacity_summary.tex}

\subsection{Moment-Level Approximation Error}
Figure~\ref{fig:appendix_direct_oracle_mnl} compares absolute moment error under the direct simulator oracle with both the segmented and homogeneous MNL projections. The diagonal marks equality with the direct oracle. If the segmented-MNL points lie systematically closer to the diagonal than the homogeneous-MNL points, the benchmark provides evidence that explicitly representing preference heterogeneity preserves behavioral structure discarded by a single population-level coefficient vector.

\begin{figure}[t]
\centering
\includegraphics[width=0.72\linewidth]{paper_artifacts/appendix_model_capacity/figures/direct_oracle_vs_mnl_moment_error.pdf}
\caption{Absolute calibration-moment error under the direct ground-truth simulator, the oracle segmented MNL, and the oracle homogeneous MNL.}
\label{fig:appendix_direct_oracle_mnl}
\end{figure}

\input{paper_artifacts/appendix_model_capacity/tables/largest_inner_model_distortions.tex}

\subsection{Oracle Segment-Specific Coefficients}
Because segment labels are observed only in the controlled benchmark, the oracle segmented MNL serves as a capacity upper bound rather than a deployable inner model. Table~\ref{tab:appendix_segmented_mnl_coefficients} reports the coefficient vectors estimated separately within each ground-truth segment.

\input{paper_artifacts/appendix_model_capacity/tables/segmented_mnl_coefficients.tex}

\subsection{Substitution Responses}
The intervention-response comparison in Table~\ref{tab:appendix_model_capacity_substitution} and Figure~\ref{fig:appendix_substitution_capacity} focuses on how probability mass moves across alternatives after the controlled price intervention. This is the central diagnostic for whether discrete heterogeneity recovers substitution structure that is lost under the homogeneous MNL.

\input{paper_artifacts/appendix_model_capacity/tables/substitution_response_capacity.tex}

\begin{figure}[t]
\centering
\includegraphics[width=0.82\linewidth]{paper_artifacts/appendix_model_capacity/figures/substitution_response_capacity.pdf}
\caption{Intervention-induced choice-share changes under the empirical target, direct ground-truth simulator, oracle segmented MNL, oracle homogeneous MNL, and selected EIPG generator.}
\label{fig:appendix_substitution_capacity}
\end{figure}

Figure~\ref{fig:appendix_capacity_summary} summarizes aggregate calibration and held-out counterfactual discrepancies across the three oracle paths. These quantities are diagnostic rather than an exact additive error decomposition; differences indicate the scale of discrepancy associated with successively stronger compression of heterogeneous behavior under the finite benchmark sample.

\begin{figure}[t]
\centering
\includegraphics[width=0.68\linewidth]{paper_artifacts/appendix_model_capacity/figures/capacity_error_summary.pdf}
\caption{Aggregate calibration and held-out counterfactual error for the direct simulator oracle, oracle segmented MNL, and oracle homogeneous MNL.}
\label{fig:appendix_capacity_summary}
\end{figure}
""".strip() + "\n"
    path = output_dir / "appendix_E_model_capacity.tex"
    path.write_text(text, encoding="utf-8")
    return path


def build_appendix_model_capacity_artifacts(
    *,
    run: str | Path | None = None,
    experiment_dir: str | Path | None = None,
    output_dir: str | Path = "paper_artifacts/appendix_model_capacity",
) -> AppendixModelCapacityArtifacts:
    run_dir = resolve_run_dir(run=run, experiment_dir=experiment_dir)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    tables_dir, figures_dir = ensure_artifact_dirs(out)

    summary = load_json(run_dir / "economic_model_capacity_summary.json")
    comparison = read_table(run_dir, "economic_model_capacity_moment_comparison")

    tables: dict[str, Path] = {}
    tables["capacity_summary"] = write_latex_table(
        _summary_table(summary),
        tables_dir / "capacity_summary.tex",
        caption="Economic-model capacity diagnostic for the controlled benchmark.",
        label="tab:appendix_model_capacity_summary",
        column_align="lrrrr",
        digits=4,
    )
    intervention = _intervention_table(comparison)
    tables["substitution_response_capacity"] = write_latex_table(
        intervention,
        tables_dir / "substitution_response_capacity.tex",
        caption="Intervention-response moments across direct, segmented, homogeneous, and EIPG model paths.",
        label="tab:appendix_model_capacity_substitution",
        column_align="l" + "r" * (len(intervention.columns) - 1),
        digits=4,
    )
    distortions = _distortion_table(comparison)
    tables["largest_inner_model_distortions"] = write_latex_table(
        distortions,
        tables_dir / "largest_inner_model_distortions.tex",
        caption="Calibration moments with the largest additional absolute error under the homogeneous MNL relative to the oracle segmented MNL.",
        label="tab:appendix_model_capacity_distortions",
        digits=4,
    )

    coeff_path = run_dir / "oracle_segmented_mnl_coefficients.parquet"
    coeff_csv = run_dir / "oracle_segmented_mnl_coefficients.csv"
    if coeff_path.exists() or coeff_csv.exists():
        coeff = read_table(run_dir, "oracle_segmented_mnl_coefficients")
        coeff_table = _segmented_coefficients_table(coeff)
        tables["segmented_mnl_coefficients"] = write_latex_table(
            coeff_table,
            tables_dir / "segmented_mnl_coefficients.tex",
            caption="Oracle segment-specific MNL coefficients estimated using known ground-truth segment labels.",
            label="tab:appendix_segmented_mnl_coefficients",
            digits=4,
        )

    figures = _save_figures(comparison, summary, figures_dir)
    latex_section = _write_latex_section(out)

    manifest_payload = {
        "run_dir": str(run_dir),
        "output_dir": str(out),
        "tables": {k: str(v) for k, v in tables.items()},
        "figures": {k: str(v) for k, v in figures.items()},
        "latex_section": str(latex_section),
    }
    manifest = out / "artifact_manifest.json"
    manifest.write_text(json.dumps(manifest_payload, indent=2, sort_keys=True), encoding="utf-8")

    return AppendixModelCapacityArtifacts(
        run_dir=run_dir,
        output_dir=out,
        tables=tables,
        figures=figures,
        latex_section=latex_section,
        manifest=manifest,
    )
