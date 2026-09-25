"""Build Appendix C calibration-moment tables and figures.

The appendix begins from a run completed through ``--stage calibration``. It
visualizes how the initial fitted economic model turns into the outer-loop
calibration signal by comparing target moments ``M_tar`` with model-implied
moments ``M_phi``.
"""

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
class AppendixCalibrationArtifacts:
    run_dir: Path
    output_dir: Path
    tables: dict[str, Path]
    figures: dict[str, Path]
    latex_section: Path
    manifest: Path


def _load_required_inputs(run_dir: Path) -> dict[str, Any]:
    config_path = run_dir / "config_used.yaml"
    report_path = run_dir / "calibration_moments_initial.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Expected config_used.yaml in {run_dir}")
    if not report_path.exists():
        raise FileNotFoundError(
            "Appendix C requires a run completed through --stage calibration. "
            f"Missing {report_path.name} in {run_dir}."
        )

    try:
        comparison = read_table(run_dir, "calibration_moment_comparison")
    except FileNotFoundError:
        comparison = None

    return {
        "config": load_yaml(config_path),
        "metadata": load_json(run_dir / "run_metadata.json")
        if (run_dir / "run_metadata.json").exists()
        else {},
        "report": load_json(report_path),
        "table": read_table(run_dir, "calibration_moment_table_initial"),
        "comparison": comparison,
        "D_H": read_table(run_dir, "D_H"),
        "D_calib_int_truth": read_table(run_dir, "D_calib_int_truth"),
        "mnl_probabilities_D_H": read_table(run_dir, "mnl_probabilities_D_H"),
        "mnl_probabilities_D_calib_int_truth": read_table(
            run_dir, "mnl_probabilities_D_calib_int_truth"
        ),
    }


def _config_summary(inputs: dict[str, Any]) -> pd.DataFrame:
    cfg = inputs["config"]
    calibration = cfg.get("calibration", {})
    benchmark = cfg.get("benchmark", {})
    report_summary = inputs["report"].get("summary", {})
    rows = [
        ("Run name", cfg.get("run", {}).get("name", "")),
        ("Run seed", cfg.get("run", {}).get("seed", "")),
        ("Stage", inputs.get("metadata", {}).get("stage_resolved", "calibration")),
        ("Moment classes", ", ".join(calibration.get("moments", []))),
        ("Weighting", calibration.get("weights", "diagonal_uniform")),
        ("Item-share normalization", "conditional on alternative availability"),
        ("Drop redundant moments", calibration.get("drop_redundant_moments", True)),
        ("Intervention share response", calibration.get("intervention_share_response", "full_substitution")),
        ("Substitution reference alternative", calibration.get("substitution_reference_alternative_id", "automatic")),
        ("Intervened alternative", benchmark.get("intervened_alternative_id", "")),
        ("Price multiplier", benchmark.get("price_intervention_multiplier", "")),
        ("Number of moments", report_summary.get("n_moments", "")),
        ("Initial moment-vector L2 error", report_summary.get("l2_error", "")),
        ("Initial moment RMSE", report_summary.get("rmse", "")),
        ("Initial moment MAE", report_summary.get("mae", "")),
    ]
    return pd.DataFrame(rows, columns=["Quantity", "Value"])


def _moment_type_summary(table: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (block, moment_type), group in table.groupby(["block", "moment_type"], sort=True):
        diff = group["diff"].to_numpy(dtype=float)
        rows.append(
            {
                "Block": block,
                "Moment class": moment_type,
                "Count": len(group),
                "RMSE": float(np.sqrt(np.mean(diff * diff))),
                "MAE": float(np.mean(np.abs(diff))),
                "Max abs. error": float(np.max(np.abs(diff))),
                "Squared-error contribution": float(np.sum(diff * diff)),
            }
        )
    return pd.DataFrame(rows)


def _largest_discrepancies(table: pd.DataFrame, n: int = 12) -> pd.DataFrame:
    cols = ["block", "moment_type", "short_name", "target", "model", "diff", "abs_diff"]
    out = table.sort_values("abs_diff", ascending=False).head(n)[cols].copy()
    return out.rename(
        columns={
            "block": "Block",
            "moment_type": "Moment class",
            "short_name": "Moment",
            "target": "Target",
            "model": "Model",
            "diff": "Model - target",
            "abs_diff": "Absolute error",
        }
    )


def _intervention_response_table(table: pd.DataFrame) -> pd.DataFrame:
    out = table[table["moment_type"] == "intervention_response"].copy()
    cols = ["short_name", "target", "model", "diff"]
    out = out[cols]
    return out.rename(
        columns={
            "short_name": "Intervention-response moment",
            "target": "Target",
            "model": "Model",
            "diff": "Model - target",
        }
    )



def _moment_before_after_summary(comparison: pd.DataFrame) -> pd.DataFrame:
    """Summarize moment error before and after EIPG by behavioral class."""

    required = {"block", "moment_type", "static_diff", "eipg_diff"}
    missing = sorted(required - set(comparison.columns))
    if missing:
        raise ValueError(f"moment comparison missing required columns: {missing}")

    rows: list[dict[str, Any]] = []
    for (block, moment_type), group in comparison.groupby(["block", "moment_type"], sort=True):
        static = group["static_diff"].to_numpy(dtype=float)
        eipg = group["eipg_diff"].to_numpy(dtype=float)
        row: dict[str, Any] = {
            "Block": block,
            "Moment class": moment_type,
            "Count": int(len(group)),
            "Static RMSE": float(np.sqrt(np.mean(static * static))),
            "EIPG RMSE": float(np.sqrt(np.mean(eipg * eipg))),
            "Static MAE": float(np.mean(np.abs(static))),
            "EIPG MAE": float(np.mean(np.abs(eipg))),
        }
        row["RMSE reduction"] = row["Static RMSE"] - row["EIPG RMSE"]
        if "oracle_truth_diff" in group.columns:
            oracle = group["oracle_truth_diff"].to_numpy(dtype=float)
            row["Oracle RMSE"] = float(np.sqrt(np.mean(oracle * oracle)))
        rows.append(row)
    columns = [
        "Block", "Moment class", "Count", "Static RMSE", "EIPG RMSE",
        "RMSE reduction", "Static MAE", "EIPG MAE",
    ]
    if rows and "Oracle RMSE" in rows[0]:
        columns.append("Oracle RMSE")
    return pd.DataFrame(rows)[columns]


def _intervention_before_after_table(comparison: pd.DataFrame) -> pd.DataFrame:
    """Return the substitution/intervention moments for static, EIPG, and oracle."""

    out = comparison[comparison["moment_type"] == "intervention_response"].copy()
    columns = ["short_name", "target", "static_model", "eipg_model"]
    rename = {
        "short_name": "Intervention-response moment",
        "target": "Target",
        "static_model": "Static",
        "eipg_model": "EIPG",
    }
    if "oracle_truth_model" in out.columns:
        columns.append("oracle_truth_model")
        rename["oracle_truth_model"] = "Oracle"
    out = out[columns].rename(columns=rename)
    out["Static abs. error"] = (
        comparison.loc[out.index, "static_abs_error"].to_numpy(dtype=float)
    )
    out["EIPG abs. error"] = (
        comparison.loc[out.index, "eipg_abs_error"].to_numpy(dtype=float)
    )
    out["Error reduction"] = out["Static abs. error"] - out["EIPG abs. error"]
    return out.reset_index(drop=True)


def _largest_moment_changes(comparison: pd.DataFrame, n: int = 12) -> pd.DataFrame:
    """Moments with the largest absolute-error changes from static to EIPG."""

    out = comparison.copy()
    out["error_change"] = out["eipg_abs_error"] - out["static_abs_error"]
    out["magnitude"] = out["error_change"].abs()
    out = out.nlargest(min(n, len(out)), "magnitude")
    cols = [
        "block", "moment_type", "short_name", "target", "static_model", "eipg_model",
        "static_abs_error", "eipg_abs_error", "error_change",
    ]
    return out[cols].rename(columns={
        "block": "Block",
        "moment_type": "Moment class",
        "short_name": "Moment",
        "target": "Target",
        "static_model": "Static",
        "eipg_model": "EIPG",
        "static_abs_error": "Static abs. error",
        "eipg_abs_error": "EIPG abs. error",
        "error_change": "EIPG - static abs. error",
    })

def _save_figures(inputs: dict[str, Any], figures_dir: Path) -> dict[str, Path]:
    import matplotlib.pyplot as plt

    table = inputs["table"].copy()
    figure_paths: dict[str, Path] = {}

    # Target-vs-model moment scatter.
    fig, ax = plt.subplots(figsize=(5.6, 4.8))
    for moment_type, group in table.groupby("moment_type", sort=True):
        ax.scatter(group["target"], group["model"], s=42, alpha=0.8, label=str(moment_type))
    lo = float(min(table["target"].min(), table["model"].min()))
    hi = float(max(table["target"].max(), table["model"].max()))
    pad = max((hi - lo) * 0.06, 0.02)
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], linestyle="--", linewidth=1.0)
    top = table.nlargest(min(5, len(table)), "abs_diff")
    for _, row in top.iterrows():
        ax.annotate(
            str(row["short_name"]),
            (row["target"], row["model"]),
            xytext=(5, 4),
            textcoords="offset points",
            fontsize=7,
        )
    ax.set_title("Calibration moments: target vs. model-implied")
    ax.set_xlabel(r"Target moment $M_{\mathrm{tar}}$")
    ax.set_ylabel(r"Model-implied moment $M_{\phi_0}$")
    ax.legend(fontsize=7, loc="best")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        path = figures_dir / f"calibration_target_vs_model.{ext}"
        fig.savefig(path, dpi=200 if ext == "png" else None, bbox_inches="tight")
        figure_paths[f"calibration_target_vs_model_{ext}"] = path
    plt.close(fig)

    # RMSE by block and moment class.
    summary = _moment_type_summary(table)
    summary["Label"] = summary["Block"].astype(str) + " / " + summary["Moment class"].astype(str)
    fig, ax = plt.subplots(figsize=(7.3, 4.2))
    ax.bar(summary["Label"], summary["RMSE"])
    ax.set_title("Initial calibration error by moment class")
    ax.set_ylabel("RMSE")
    ax.tick_params(axis="x", labelrotation=35)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        path = figures_dir / f"calibration_error_by_type.{ext}"
        fig.savefig(path, dpi=200 if ext == "png" else None, bbox_inches="tight")
        figure_paths[f"calibration_error_by_type_{ext}"] = path
    plt.close(fig)

    # Intervention-response target/model comparison.
    intervention = table[table["moment_type"] == "intervention_response"].copy()
    if not intervention.empty:
        intervention = intervention.sort_values("short_name")
        x = np.arange(len(intervention))
        width = 0.38
        fig, ax = plt.subplots(figsize=(8.2, 4.3))
        ax.bar(x - width / 2, intervention["target"], width, label="target")
        ax.bar(x + width / 2, intervention["model"], width, label="model")
        ax.axhline(0, linewidth=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(intervention["short_name"], rotation=35, ha="right")
        ax.set_title("Calibration-intervention response moments")
        ax.set_ylabel("Moment value")
        ax.legend(fontsize=8)
        fig.tight_layout()
        for ext in ("pdf", "png"):
            path = figures_dir / f"intervention_response_comparison.{ext}"
            fig.savefig(path, dpi=200 if ext == "png" else None, bbox_inches="tight")
            figure_paths[f"intervention_response_comparison_{ext}"] = path
        plt.close(fig)

    # Conceptual flow using actual appendix objects.
    fig, ax = plt.subplots(figsize=(8.6, 3.8))
    ax.axis("off")
    boxes = [
        (0.10, 0.55, r"target behavior\n$D_H$, $D_{calib}$"),
        (0.34, 0.55, r"target moments\n$M_{tar}$"),
        (0.58, 0.55, r"MNL probabilities\n$p_{\beta^*(\phi)}(y\mid x)$"),
        (0.78, 0.55, r"model moments\n$M_{\phi}$"),
        (0.94, 0.55, r"discrepancy\n$M_{\phi}-M_{tar}$"),
    ]
    for x_pos, y_pos, text in boxes:
        ax.text(
            x_pos,
            y_pos,
            text,
            ha="center",
            va="center",
            fontsize=9.5,
            bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="0.35"),
            transform=ax.transAxes,
        )
    for start, end in [
        ((0.19, 0.55), (0.26, 0.55)),
        ((0.43, 0.55), (0.50, 0.55)),
        ((0.67, 0.55), (0.71, 0.55)),
        ((0.86, 0.55), (0.89, 0.55)),
    ]:
        ax.annotate(
            "",
            xy=end,
            xytext=start,
            xycoords="axes fraction",
            textcoords="axes fraction",
            arrowprops=dict(arrowstyle="->", lw=1.3),
        )
    ax.set_title("Construction of the calibration signal")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        path = figures_dir / f"calibration_moment_flow.{ext}"
        fig.savefig(path, dpi=200 if ext == "png" else None, bbox_inches="tight")
        figure_paths[f"calibration_moment_flow_{ext}"] = path
    plt.close(fig)

    comparison = inputs.get("comparison")
    if comparison is not None and not comparison.empty:
        # Moment-level before/after error: points below diagonal improved under EIPG.
        fig, ax = plt.subplots(figsize=(5.8, 5.0))
        ax.scatter(comparison["static_abs_error"], comparison["eipg_abs_error"], s=46, alpha=0.8)
        hi = float(max(comparison["static_abs_error"].max(), comparison["eipg_abs_error"].max()))
        pad = max(hi * 0.05, 0.002)
        ax.plot([0, hi + pad], [0, hi + pad], linestyle="--", linewidth=1.0)
        changed = comparison.assign(
            change=(comparison["eipg_abs_error"] - comparison["static_abs_error"]).abs()
        ).nlargest(min(7, len(comparison)), "change")
        for _, row in changed.iterrows():
            ax.annotate(
                str(row["short_name"]),
                (row["static_abs_error"], row["eipg_abs_error"]),
                xytext=(4, 4), textcoords="offset points", fontsize=7,
            )
        ax.set_xlabel("Static absolute moment error")
        ax.set_ylabel("EIPG absolute moment error")
        ax.set_title("Moment-level error before and after generator calibration")
        fig.tight_layout()
        for ext in ("pdf", "png"):
            path = figures_dir / f"moment_error_before_after.{ext}"
            fig.savefig(path, dpi=200 if ext == "png" else None, bbox_inches="tight")
            figure_paths[f"moment_error_before_after_{ext}"] = path
        plt.close(fig)

        intervention = comparison[comparison["moment_type"] == "intervention_response"].copy()
        if not intervention.empty:
            intervention = intervention.sort_values("short_name")
            x = np.arange(len(intervention))
            series = [("target", "Target"), ("static_model", "Static"), ("eipg_model", "EIPG")]
            if "oracle_truth_model" in intervention.columns:
                series.append(("oracle_truth_model", "Oracle"))
            width = 0.78 / len(series)
            fig, ax = plt.subplots(figsize=(9.0, 4.6))
            offsets = (np.arange(len(series)) - (len(series) - 1) / 2) * width
            for offset, (col, label) in zip(offsets, series):
                ax.bar(x + offset, intervention[col], width, label=label)
            ax.axhline(0, linewidth=0.8)
            ax.set_xticks(x)
            ax.set_xticklabels(intervention["short_name"], rotation=35, ha="right")
            ax.set_ylabel("Response moment")
            ax.set_title("Substitution-response moments: static versus calibrated generator")
            ax.legend(fontsize=8)
            fig.tight_layout()
            for ext in ("pdf", "png"):
                path = figures_dir / f"intervention_response_before_after.{ext}"
                fig.savefig(path, dpi=200 if ext == "png" else None, bbox_inches="tight")
                figure_paths[f"intervention_response_before_after_{ext}"] = path
            plt.close(fig)

    return figure_paths


def _write_latex_section(output_dir: Path, has_before_after: bool) -> Path:
    prefix = r"""
\section{Calibration Moments and Initial Discrepancy}
\label{app:calibration_moments}

This appendix details how the fitted inner-loop economic model is converted into the behavioral feedback used by EIPG. For a fixed generator $G_{\phi}$, the inner loop produces $\beta^*(\phi)$. We then compare behavioral moments implied by $p_{\beta^*(\phi)}(\cdot\mid x)$ with target moments derived from the anchor and calibration-intervention data.

\subsection{Moment Construction}

Let $M_{\mathrm{tar}}$ denote the target moment vector and $M_{\phi}$ the corresponding model-implied vector. The calibration loss used in the controlled benchmark is
\[
\mathcal{L}_{\mathrm{cal}}^{\mathrm{GMM}}(\phi)
=
(M_{\phi}-M_{\mathrm{tar}})^\top W(M_{\phi}-M_{\mathrm{tar}}),
\]
with diagonal uniform weighting in the experiments reported here. The moment vector contains item-level choice shares, group-level shares, average chosen attributes, and intervention-response summaries. Item-level shares are computed conditional on the corresponding alternative being available in the choice slate; an alternative is therefore not penalized for contexts in which it does not appear. Because uniform diagonal weighting would otherwise count some identical summaries multiple times, we use a reduced moment representation for exact redundancies. In the current benchmark, the first brand category is treated as the reference group, the binary mean-brand attribute is omitted when the corresponding brand share is present, and the post-intervention share of the intervened alternative is not repeated when that share already appears in the calibration-intervention item-share block.

\input{paper_artifacts/appendix_calibration_moments/tables/calibration_config_summary.tex}

Figure~\ref{fig:appendix_calibration_flow} summarizes the construction of the calibration signal. Target behavior produces $M_{\mathrm{tar}}$, while the fitted MNL probabilities produce $M_{\phi}$. Their discrepancy is the behavioral signal passed to the outer generator optimization.

\begin{figure}[t]
\centering
\includegraphics[width=0.93\linewidth]{paper_artifacts/appendix_calibration_moments/figures/calibration_moment_flow.pdf}
\caption{Construction of the calibration signal from target behavior and model-implied behavioral moments.}
\label{fig:appendix_calibration_flow}
\end{figure}

\subsection{Initial Moment Alignment}

Before outer-loop optimization, we evaluate the generator initialization $\phi_0$. Figure~\ref{fig:appendix_calibration_target_model} plots each target moment against its model-implied counterpart. Points on the diagonal are matched exactly; deviations from the diagonal identify behavioral summaries that the initial synthetic population fails to reproduce. Because the moment classes have different natural scales, Table~\ref{tab:appendix_calibration_by_type} and Figure~\ref{fig:appendix_calibration_error_type} additionally report error within each moment class.

\begin{figure}[t]
\centering
\includegraphics[width=0.68\linewidth]{paper_artifacts/appendix_calibration_moments/figures/calibration_target_vs_model.pdf}
\caption{Target versus model-implied calibration moments before outer-loop optimization. The most discrepant moments are annotated.}
\label{fig:appendix_calibration_target_model}
\end{figure}

\input{paper_artifacts/appendix_calibration_moments/tables/calibration_moment_summary.tex}

\begin{figure}[t]
\centering
\includegraphics[width=0.82\linewidth]{paper_artifacts/appendix_calibration_moments/figures/calibration_error_by_type.pdf}
\caption{Initial calibration RMSE by behavioral-moment class and calibration block.}
\label{fig:appendix_calibration_error_type}
\end{figure}

\subsection{Largest Initial Discrepancies}

Table~\ref{tab:appendix_calibration_largest} lists the moments with the largest absolute discrepancies before optimizing the persona generator. This diagnostic is useful for determining which aspects of behavior are not represented by the initial synthetic population and therefore exert the strongest pressure on the subsequent outer loop.

\input{paper_artifacts/appendix_calibration_moments/tables/largest_moment_discrepancies.tex}

\subsection{Calibration-Intervention Responses}

The controlled benchmark additionally contains intervention-response targets constructed from price-perturbed contexts. These moments measure whether the fitted economic model responds to the intervention in the same direction and magnitude as the target population. Table~\ref{tab:appendix_intervention_response} and Figure~\ref{fig:appendix_intervention_response} compare target and model-implied intervention responses. These calibration interventions are distinct from the held-out counterfactual contexts reserved for final evaluation.

\input{paper_artifacts/appendix_calibration_moments/tables/intervention_response_moments.tex}

\begin{figure}[t]
\centering
\includegraphics[width=0.88\linewidth]{paper_artifacts/appendix_calibration_moments/figures/intervention_response_comparison.pdf}
\caption{Target and model-implied response moments on calibration-intervention contexts.}
\label{fig:appendix_intervention_response}
\end{figure}
""".strip()

    before_after = r"""
\subsection{Moment-Level Effect of Generator Calibration}

For full outer-loop runs, we additionally compare each behavioral moment under the initial static generator, the final EIPG generator, and the controlled-benchmark oracle. Table~\ref{tab:appendix_moment_before_after} reports error by behavioral class, while Table~\ref{tab:appendix_intervention_before_after} isolates the intervention-response moments. Positive error reduction means that EIPG moved the fitted model closer to the calibration target than the static generator.

\input{paper_artifacts/appendix_calibration_moments/tables/moment_before_after_summary.tex}

\input{paper_artifacts/appendix_calibration_moments/tables/intervention_response_before_after.tex}

Figure~\ref{fig:appendix_moment_before_after} compares absolute error for every calibration moment before and after generator optimization. Points below the diagonal improved under EIPG; points above the diagonal became worse.

\begin{figure}[t]
\centering
\includegraphics[width=0.70\linewidth]{paper_artifacts/appendix_calibration_moments/figures/moment_error_before_after.pdf}
\caption{Absolute behavioral-moment error for the static generator versus the final EIPG generator. Points below the diagonal are improved by generator calibration.}
\label{fig:appendix_moment_before_after}
\end{figure}

Figure~\ref{fig:appendix_intervention_before_after} focuses on substitution and attribute responses to the calibration intervention. This makes it possible to determine whether a lower aggregate calibration norm corresponds to improved substitution structure or is driven primarily by easier moments.

\begin{figure}[t]
\centering
\includegraphics[width=0.93\linewidth]{paper_artifacts/appendix_calibration_moments/figures/intervention_response_before_after.pdf}
\caption{Target, static-generator, EIPG, and oracle intervention-response moments.}
\label{fig:appendix_intervention_before_after}
\end{figure}

Table~\ref{tab:appendix_largest_moment_changes} lists the moments whose absolute errors changed most strongly under generator calibration. Negative values in the final column indicate improvement, while positive values indicate that the selected EIPG generator moved farther from the corresponding target moment.

\input{paper_artifacts/appendix_calibration_moments/tables/largest_moment_changes.tex}
""".strip()

    suffix = r"""
\subsection{Role in the Outer Optimization}

The initial discrepancies define the signal supplied to the outer loop. Each candidate generator produces a new synthetic dataset $D_{\phi}$, a new inner-loop fit $\beta^*(\phi)$, and therefore a new moment vector $M_{\phi}$. The before--after diagnostics above compare the initial static generator with the final EIPG generator on exactly the same target moments and common-random-number evaluation protocol. Held-out counterfactual contexts $\mathcal{X}_{\mathrm{cf}}$ remain excluded from this objective.
""".strip()

    parts = [prefix]
    if has_before_after:
        parts.append(before_after)
    parts.append(suffix)
    text = "\n\n".join(parts)
    path = output_dir / "appendix_C_calibration_moments.tex"
    path.write_text(text + "\n", encoding="utf-8")
    return path

def build_appendix_calibration_artifacts(
    *,
    run: str | Path | None = None,
    experiment_dir: str | Path | None = None,
    output_dir: str | Path = "paper_artifacts/appendix_calibration_moments",
) -> AppendixCalibrationArtifacts:
    run_dir = resolve_run_dir(run=run, experiment_dir=experiment_dir)
    out = Path(output_dir)
    tables_dir, figures_dir = ensure_artifact_dirs(out)
    inputs = _load_required_inputs(run_dir)
    table = inputs["table"]

    tables: dict[str, Path] = {}
    tables["calibration_config_summary"] = write_latex_table(
        _config_summary(inputs),
        tables_dir / "calibration_config_summary.tex",
        caption="Configuration and initial summary for the calibration-moment objective.",
        label="tab:appendix_calibration_config",
        column_align="ll",
    )
    tables["calibration_moment_summary"] = write_latex_table(
        _moment_type_summary(table),
        tables_dir / "calibration_moment_summary.tex",
        caption="Initial calibration error summarized by block and behavioral-moment class.",
        label="tab:appendix_calibration_by_type",
        column_align="llrrrrr",
    )
    tables["largest_moment_discrepancies"] = write_latex_table(
        _largest_discrepancies(table),
        tables_dir / "largest_moment_discrepancies.tex",
        caption="Largest absolute behavioral-moment discrepancies before outer-loop optimization.",
        label="tab:appendix_calibration_largest",
        column_align="lllrrrr",
    )
    tables["intervention_response_moments"] = write_latex_table(
        _intervention_response_table(table),
        tables_dir / "intervention_response_moments.tex",
        caption="Target and model-implied calibration-intervention response moments.",
        label="tab:appendix_intervention_response",
        column_align="lrrr",
    )

    comparison = inputs.get("comparison")
    if comparison is not None and not comparison.empty:
        tables["moment_before_after_summary"] = write_latex_table(
            _moment_before_after_summary(comparison),
            tables_dir / "moment_before_after_summary.tex",
            caption="Calibration error before and after EIPG, summarized by behavioral-moment class.",
            label="tab:appendix_moment_before_after",
            column_align="llrrrrrrr",
            digits=4,
        )
        intervention_ba = _intervention_before_after_table(comparison)
        tables["intervention_response_before_after"] = write_latex_table(
            intervention_ba,
            tables_dir / "intervention_response_before_after.tex",
            caption="Intervention-response moments under the static generator, final EIPG generator, and controlled-benchmark oracle.",
            label="tab:appendix_intervention_before_after",
            column_align="l" + "r" * (len(intervention_ba.columns) - 1),
            digits=4,
        )
        largest_changes = _largest_moment_changes(comparison)
        tables["largest_moment_changes"] = write_latex_table(
            largest_changes,
            tables_dir / "largest_moment_changes.tex",
            caption="Behavioral moments with the largest changes in absolute error from the static generator to EIPG. Negative final-column values indicate improvement.",
            label="tab:appendix_largest_moment_changes",
            column_align="lll" + "r" * (len(largest_changes.columns) - 3),
            digits=4,
        )

    figures = _save_figures(inputs, figures_dir)
    latex_section = _write_latex_section(
        out, has_before_after=comparison is not None and not comparison.empty
    )

    manifest_payload = {
        "appendix": "C_calibration_moments",
        "run_dir": str(run_dir),
        "output_dir": str(out),
        "share_definition": "item shares conditional on alternative availability",
        "moment_before_after_available": bool(inputs.get("comparison") is not None),
        "tables": {k: str(v) for k, v in tables.items()},
        "figures": {k: str(v) for k, v in figures.items()},
        "latex_section": str(latex_section),
    }
    manifest = out / "artifact_manifest.json"
    manifest.write_text(json.dumps(manifest_payload, indent=2, sort_keys=True), encoding="utf-8")

    return AppendixCalibrationArtifacts(
        run_dir=run_dir,
        output_dir=out,
        tables=tables,
        figures=figures,
        latex_section=latex_section,
        manifest=manifest,
    )
