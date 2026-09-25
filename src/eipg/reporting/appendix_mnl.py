"""Build Appendix B inner-MNL-estimation tables and figures.

This appendix starts from a completed run folder that has been executed through
at least ``--stage mnl``.  It reads the fitted MNL object, evaluation summaries,
and long-format probability files, then writes paper-ready tables/figures plus a
LaTeX section that can be included in the manuscript.
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
class AppendixMNLArtifacts:
    """Paths produced by an Appendix B artifact build."""

    run_dir: Path
    output_dir: Path
    tables: dict[str, Path]
    figures: dict[str, Path]
    latex_section: Path
    manifest: Path


def _safe_get(config: dict[str, Any], *keys: str, default: Any = "") -> Any:
    cur: Any = config
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def _load_required_inputs(run_dir: Path) -> dict[str, Any]:
    config_path = run_dir / "config_used.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Expected config_used.yaml in {run_dir}")

    fit_path = run_dir / "mnl_fit_initial.json"
    eval_path = run_dir / "mnl_evaluation_initial.json"
    if not fit_path.exists() or not eval_path.exists():
        raise FileNotFoundError(
            "Appendix B requires a run completed through --stage mnl. "
            f"Missing {fit_path.name if not fit_path.exists() else eval_path.name} in {run_dir}."
        )

    metadata = load_json(run_dir / "run_metadata.json") if (run_dir / "run_metadata.json").exists() else None
    inputs: dict[str, Any] = {
        "config": load_yaml(config_path),
        "metadata": metadata,
        "mnl_fit": load_json(fit_path),
        "mnl_evaluation": load_json(eval_path),
        "D_phi_initial": read_table(run_dir, "D_phi_initial"),
        "D_H": read_table(run_dir, "D_H"),
        "D_calib_int_truth": read_table(run_dir, "D_calib_int_truth"),
        "D_cf_truth": read_table(run_dir, "D_cf_truth"),
        "mnl_probabilities_D_phi_initial": read_table(run_dir, "mnl_probabilities_D_phi_initial"),
        "mnl_probabilities_D_H": read_table(run_dir, "mnl_probabilities_D_H"),
        "mnl_probabilities_D_calib_int_truth": read_table(run_dir, "mnl_probabilities_D_calib_int_truth"),
        "mnl_probabilities_D_cf_truth": read_table(run_dir, "mnl_probabilities_D_cf_truth"),
    }
    return inputs


def _mnl_config_table(inputs: dict[str, Any]) -> pd.DataFrame:
    cfg = inputs["config"]
    fit = inputs["mnl_fit"]
    metadata = inputs.get("metadata") or {}
    rows = [
        ("Run name", _safe_get(cfg, "run", "name")),
        ("Run seed", _safe_get(cfg, "run", "seed")),
        ("Stage", metadata.get("stage_resolved", "mnl")),
        ("Training dataset", r"$D_{\phi_0}$"),
        ("Training observations", fit.get("n_observations", "")),
        ("Training rows", fit.get("n_rows", "")),
        ("Choice model", "Multinomial logit"),
        ("Feature vector", ", ".join(fit.get("features", []))),
        ("L2 penalty", fit.get("l2", "")),
        ("Optimizer success", fit.get("success", "")),
        ("Iterations", fit.get("n_iter", "")),
        ("Training NLL / obs.", fit.get("train_nll_per_observation", "")),
    ]
    return pd.DataFrame(rows, columns=["Quantity", "Value"])


def _coefficient_table(fit: dict[str, Any]) -> pd.DataFrame:
    beta_by_feature = fit.get("beta_by_feature", {})
    rows = []
    descriptions = {
        "price": "Price sensitivity",
        "quality": "Quality preference",
        "sustain": "Sustainability preference",
        "novelty": "Novelty preference",
        "brand": "Brand preference",
    }
    for feature in fit.get("features", list(beta_by_feature.keys())):
        value = float(beta_by_feature[feature])
        rows.append(
            {
                "Feature": feature,
                "Interpretation": descriptions.get(feature, "Utility feature"),
                r"$\hat\beta$": value,
                "Sign": "positive" if value > 0 else "negative" if value < 0 else "zero",
            }
        )
    return pd.DataFrame(rows)


def _evaluation_table(evaluation: dict[str, Any]) -> pd.DataFrame:
    datasets = evaluation.get("datasets", {})
    order = ["D_phi_initial", "D_H", "D_calib_int_truth", "D_cf_truth"]
    roles = {
        "D_phi_initial": "inner-loop training",
        "D_H": "anchor behavior",
        "D_calib_int_truth": "calibration intervention target",
        "D_cf_truth": "held-out counterfactual target",
    }
    rows = []
    for name in order:
        if name not in datasets:
            continue
        stats = datasets[name]
        rows.append(
            {
                "Dataset": name,
                "Role": roles.get(name, "evaluation"),
                "Observations": stats.get("n_observations", ""),
                "Rows": stats.get("n_rows", ""),
                "NLL / obs.": stats.get("nll_per_observation", ""),
                "Top-choice acc.": stats.get("top_choice_accuracy", ""),
            }
        )
    return pd.DataFrame(rows)


def _share_table(preds: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    roles = {
        "D_phi_initial": "training",
        "D_H": "anchor",
        "D_calib_int_truth": "calibration intervention",
        "D_cf_truth": "held-out counterfactual",
    }
    for name, df in preds.items():
        for alt in sorted(df["alternative_id"].dropna().unique().tolist()):
            g = df[df["alternative_id"] == alt]
            # Shares are conditional on availability: an alternative contributes
            # only in contexts in which it appears in the choice slate.
            n_available = max(int(g["observation_id"].nunique()), 1)
            empirical = float(g["chosen"].sum() / n_available)
            model = float(g["mnl_prob"].sum() / n_available)
            rows.append(
                {
                    "Dataset": name,
                    "Role": roles.get(name, "evaluation"),
                    "Alternative": int(alt),
                    "Available contexts": n_available,
                    "Empirical share": empirical,
                    "MNL-implied share": model,
                    "Difference": model - empirical,
                }
            )
    return pd.DataFrame(rows)


def _example_probability_table(df: pd.DataFrame) -> pd.DataFrame:
    obs_id = df["observation_id"].iloc[0]
    cols = [
        "observation_id",
        "alternative_id",
        "price",
        "quality",
        "sustain",
        "novelty",
        "brand",
        "chosen",
        "mnl_utility",
        "mnl_prob",
    ]
    out = df[df["observation_id"] == obs_id][cols].copy()
    return out.rename(
        columns={
            "observation_id": "Observation",
            "alternative_id": "Alt.",
            "price": "Price",
            "quality": "Quality",
            "sustain": "Sustain.",
            "novelty": "Novelty",
            "brand": "Brand",
            "chosen": "Chosen",
            "mnl_utility": r"$\hat v_{ij}$",
            "mnl_prob": r"$p_{\hat\beta}(j\mid x_i)$",
        }
    )


def _save_figures(inputs: dict[str, Any], figures_dir: Path, share_table: pd.DataFrame) -> dict[str, Path]:
    import matplotlib.pyplot as plt

    figure_paths: dict[str, Path] = {}
    fit = inputs["mnl_fit"]
    coef_df = _coefficient_table(fit)

    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    ax.bar(coef_df["Feature"], coef_df[r"$\hat\beta$"])
    ax.axhline(0, linewidth=0.8)
    ax.set_title("Fitted MNL coefficients")
    ax.set_xlabel("Feature")
    ax.set_ylabel(r"Coefficient $\hat\beta$")
    fig.tight_layout()
    pdf = figures_dir / "mnl_coefficients.pdf"
    png = figures_dir / "mnl_coefficients.png"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, dpi=200, bbox_inches="tight")
    plt.close(fig)
    figure_paths["mnl_coefficients_pdf"] = pdf
    figure_paths["mnl_coefficients_png"] = png

    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    anchor = share_table[share_table["Dataset"] == "D_H"]
    ax.scatter(anchor["Empirical share"], anchor["MNL-implied share"], s=50)
    for _, row in anchor.iterrows():
        ax.annotate(
            f"alt {int(row['Alternative'])}",
            (float(row["Empirical share"]), float(row["MNL-implied share"])),
            xytext=(5, 4),
            textcoords="offset points",
            fontsize=8,
        )
    min_v = float(min(anchor["Empirical share"].min(), anchor["MNL-implied share"].min()))
    max_v = float(max(anchor["Empirical share"].max(), anchor["MNL-implied share"].max()))
    pad = 0.03
    ax.plot([min_v - pad, max_v + pad], [min_v - pad, max_v + pad], linestyle="--", linewidth=1.0)
    ax.set_title("Anchor choice shares: empirical vs. MNL-implied")
    ax.set_xlabel("Empirical share")
    ax.set_ylabel("MNL-implied share")
    fig.tight_layout()
    pdf = figures_dir / "anchor_choice_share_fit.pdf"
    png = figures_dir / "anchor_choice_share_fit.png"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, dpi=200, bbox_inches="tight")
    plt.close(fig)
    figure_paths["anchor_choice_share_fit_pdf"] = pdf
    figure_paths["anchor_choice_share_fit_png"] = png

    d_phi = inputs["mnl_probabilities_D_phi_initial"].copy()
    # Bin fitted probabilities and compare to empirical chosen frequency.
    d_phi["prob_bin"] = pd.qcut(d_phi["mnl_prob"], q=min(6, d_phi["mnl_prob"].nunique()), duplicates="drop")
    calib = d_phi.groupby("prob_bin", observed=False).agg(
        mean_pred=("mnl_prob", "mean"), empirical_chosen=("chosen", "mean")
    ).reset_index()
    fig, ax = plt.subplots(figsize=(5.4, 4.2))
    ax.scatter(calib["mean_pred"], calib["empirical_chosen"], s=55)
    min_v = 0.0
    max_v = float(max(calib["mean_pred"].max(), calib["empirical_chosen"].max()))
    ax.plot([min_v, max_v], [min_v, max_v], linestyle="--", linewidth=1.0)
    ax.set_title("Training probability calibration")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Empirical chosen rate")
    fig.tight_layout()
    pdf = figures_dir / "training_probability_calibration.pdf"
    png = figures_dir / "training_probability_calibration.png"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, dpi=200, bbox_inches="tight")
    plt.close(fig)
    figure_paths["training_probability_calibration_pdf"] = pdf
    figure_paths["training_probability_calibration_png"] = png

    fig, ax = plt.subplots(figsize=(8.2, 3.8))
    ax.axis("off")
    boxes = [
        (0.12, 0.55, r"$D_{\phi_0}$\nsynthetic choices"),
        (0.38, 0.55, r"fit MNL\n$\hat\beta=\beta^*(\phi_0)$"),
        (0.64, 0.55, r"predict probabilities\n$p_{\hat\beta}(j\mid x)$"),
        (0.88, 0.55, r"evaluate on\n$D_H,D_{calib},D_{cf}$"),
    ]
    for x, y, text in boxes:
        ax.text(
            x,
            y,
            text,
            ha="center",
            va="center",
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="0.35"),
            transform=ax.transAxes,
        )
    for start, end in [((0.22, 0.55), (0.29, 0.55)), ((0.48, 0.55), (0.55, 0.55)), ((0.74, 0.55), (0.80, 0.55))]:
        ax.annotate("", xy=end, xytext=start, xycoords="axes fraction", textcoords="axes fraction", arrowprops=dict(arrowstyle="->", lw=1.4))
    ax.set_title("Inner-loop MNL estimation and evaluation")
    fig.tight_layout()
    pdf = figures_dir / "inner_mnl_flow.pdf"
    png = figures_dir / "inner_mnl_flow.png"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, dpi=200, bbox_inches="tight")
    plt.close(fig)
    figure_paths["inner_mnl_flow_pdf"] = pdf
    figure_paths["inner_mnl_flow_png"] = png

    return figure_paths


def _write_latex_section(output_dir: Path) -> Path:
    text = r"""
\section{Inner-Loop Economic Model Estimation}
\label{app:inner_mnl_estimation}

This appendix describes the inner-loop economic estimation stage used in the controlled synthetic benchmark. At this stage, the synthetic dataset $D_{\phi_0}$ has already been generated by sampling personas from the initial generator and simulating choices with the symbolic random-utility simulator. The role of the inner loop is to fit an econometric summary model to those simulated choices; it does not generate the choices itself.

\subsection{Estimation Problem}

For a fixed generator $G_{\phi_0}$, the dataset construction stage produces
\[
D_{\phi_0}=\{(x_i,z_i,y_i)\}_{i=1}^{n_S}.
\]
The inner loop fits a multinomial logit model to this dataset:
\[
\hat\beta=\beta^*(\phi_0)=\arg\min_{\beta}\mathcal{L}_{\mathrm{econ}}(\beta;D_{\phi_0}).
\]
For each alternative $j$ in context $x_i$, the fitted model assigns systematic utility
\[
\hat v_{ij}=\hat\beta^\top f(x_i,j),
\]
which induces probabilities
\[
 p_{\hat\beta}(y_i=j\mid x_i)
 =
 \frac{\exp(\hat v_{ij})}{\sum_{k\in\mathcal{C}(x_i)}\exp(\hat v_{ik})}.
\]
This fitted model is then evaluated on the initial synthetic data, the anchor data, the calibration-intervention data, and the held-out counterfactual data.

\input{paper_artifacts/appendix_inner_mnl/tables/mnl_config_summary.tex}

\subsection{Fitted Coefficients}

Table~\ref{tab:appendix_mnl_coefficients} reports the fitted coefficients. The sign and magnitude of these coefficients summarize the average choice behavior induced by the initial synthetic population and simulator. Figure~\ref{fig:appendix_mnl_coefficients} visualizes the same fitted coefficients.

\input{paper_artifacts/appendix_inner_mnl/tables/mnl_coefficients.tex}

\begin{figure}[t]
\centering
\includegraphics[width=0.78\linewidth]{paper_artifacts/appendix_inner_mnl/figures/mnl_coefficients.pdf}
\caption{Fitted MNL coefficients for the initial synthetic dataset $D_{\phi_0}$.}
\label{fig:appendix_mnl_coefficients}
\end{figure}

\subsection{Evaluation Across Dataset Splits}

After fitting on $D_{\phi_0}$, we evaluate the same fitted model on all relevant splits. Table~\ref{tab:appendix_mnl_evaluation} reports negative log-likelihood per observation and top-choice accuracy. These quantities are descriptive diagnostics for the inner model; the later calibration objective uses behavioral moments rather than likelihood alone.

\input{paper_artifacts/appendix_inner_mnl/tables/mnl_evaluation_summary.tex}

Figure~\ref{fig:appendix_anchor_choice_share_fit} compares empirical anchor choice shares with the shares implied by the fitted MNL model on the same anchor contexts. Each point corresponds to one alternative, and shares are computed conditional on that alternative being available in the choice set. Points near the diagonal indicate that the inner model reproduces aggregate anchor behavior for the corresponding alternative, while points above or below the diagonal indicate over- or under-prediction. This diagnostic evaluates anchor fit rather than held-out counterfactual validity.

\begin{figure}[t]
\centering
\includegraphics[width=0.65\linewidth]{paper_artifacts/appendix_inner_mnl/figures/anchor_choice_share_fit.pdf}
\caption{Empirical versus MNL-implied choice shares on the anchor dataset $D_H$. Each point corresponds to one alternative. Shares are computed conditional on alternative availability; labels identify the five alternatives in the controlled environment.}
\label{fig:appendix_anchor_choice_share_fit}
\end{figure}

\subsection{Example Probability Calculation}

Table~\ref{tab:appendix_mnl_example_probabilities} shows one concrete choice slate after MNL prediction. The table includes the observed attributes, the sampled chosen alternative, the fitted systematic utility $\hat v_{ij}$, and the model-implied probability $p_{\hat\beta}(j\mid x_i)$. This table makes explicit how the fitted economic model converts a context into a probability distribution over alternatives.

\input{paper_artifacts/appendix_inner_mnl/tables/example_mnl_probabilities.tex}

\subsection{Role in the Full EIPG Loop}

The fitted model $m_{\hat\beta}$ is the bridge between simulated choices and generator calibration. The next appendix uses the MNL-implied probabilities to construct $M_{\phi}$ and compares them against target moments $M_{\mathrm{tar}}$ from anchor and calibration-intervention behavior. Thus the inner-loop model does not by itself define the final objective; it supplies the economically interpretable behavior summaries used by the outer calibration loop.

\begin{figure}[t]
\centering
\includegraphics[width=0.88\linewidth]{paper_artifacts/appendix_inner_mnl/figures/inner_mnl_flow.pdf}
\caption{Inner-loop estimation stage. The MNL model is fitted to $D_{\phi_0}$ and then used to compute probabilities on anchor, calibration-intervention, and held-out counterfactual contexts.}
\label{fig:appendix_inner_mnl_flow}
\end{figure}
""".strip()
    path = output_dir / "appendix_B_inner_mnl_estimation.tex"
    path.write_text(text + "\n", encoding="utf-8")
    return path


def build_appendix_mnl_artifacts(
    *,
    run: str | Path | None = None,
    experiment_dir: str | Path | None = None,
    output_dir: str | Path = "paper_artifacts/appendix_inner_mnl",
) -> AppendixMNLArtifacts:
    """Build all Appendix B paper artifacts from a completed run."""

    run_dir = resolve_run_dir(run=run, experiment_dir=experiment_dir)
    out = Path(output_dir)
    tables_dir, figures_dir = ensure_artifact_dirs(out)
    inputs = _load_required_inputs(run_dir)

    preds = {
        "D_phi_initial": inputs["mnl_probabilities_D_phi_initial"],
        "D_H": inputs["mnl_probabilities_D_H"],
        "D_calib_int_truth": inputs["mnl_probabilities_D_calib_int_truth"],
        "D_cf_truth": inputs["mnl_probabilities_D_cf_truth"],
    }
    share_table = _share_table(preds)

    tables: dict[str, Path] = {}
    tables["mnl_config_summary"] = write_latex_table(
        _mnl_config_table(inputs),
        tables_dir / "mnl_config_summary.tex",
        caption="Configuration and fit summary for the inner-loop MNL model.",
        label="tab:appendix_mnl_config_summary",
        column_align="ll",
    )
    tables["mnl_coefficients"] = write_latex_table(
        _coefficient_table(inputs["mnl_fit"]),
        tables_dir / "mnl_coefficients.tex",
        caption="Fitted MNL coefficients from the initial synthetic dataset.",
        label="tab:appendix_mnl_coefficients",
        column_align="llrl",
    )
    tables["mnl_evaluation_summary"] = write_latex_table(
        _evaluation_table(inputs["mnl_evaluation"]),
        tables_dir / "mnl_evaluation_summary.tex",
        caption="Evaluation of the initial fitted MNL model across dataset splits.",
        label="tab:appendix_mnl_evaluation",
        column_align="llrrrr",
    )
    tables["choice_share_summary"] = write_latex_table(
        share_table,
        tables_dir / "choice_share_summary.tex",
        caption="Empirical and MNL-implied choice shares by dataset and alternative, conditional on alternative availability.",
        label="tab:appendix_mnl_choice_share_summary",
        column_align="llrrrrr",
    )
    tables["example_mnl_probabilities"] = write_latex_table(
        _example_probability_table(inputs["mnl_probabilities_D_H"]),
        tables_dir / "example_mnl_probabilities.tex",
        caption="Example MNL probability calculation for one anchor choice slate.",
        label="tab:appendix_mnl_example_probabilities",
        column_align="llrrrrrrrr",
    )

    figures = _save_figures(inputs, figures_dir, share_table)
    latex_section = _write_latex_section(out)

    manifest_payload = {
        "appendix": "B_inner_mnl_estimation",
        "run_dir": str(run_dir),
        "output_dir": str(out),
        "tables": {k: str(v) for k, v in tables.items()},
        "figures": {k: str(v) for k, v in figures.items()},
        "latex_section": str(latex_section),
    }
    manifest = out / "artifact_manifest.json"
    manifest.write_text(json.dumps(manifest_payload, indent=2, sort_keys=True), encoding="utf-8")

    return AppendixMNLArtifacts(
        run_dir=run_dir,
        output_dir=out,
        tables=tables,
        figures=figures,
        latex_section=latex_section,
        manifest=manifest,
    )
