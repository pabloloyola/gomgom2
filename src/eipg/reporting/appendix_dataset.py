"""Build Appendix A dataset-construction tables and figures."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

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
class AppendixArtifacts:
    """Paths produced by an appendix artifact build."""

    run_dir: Path
    output_dir: Path
    tables: dict[str, Path]
    figures: dict[str, Path]
    latex_section: Path
    manifest: Path


def _safe_unique(df: pd.DataFrame, column: str) -> int:
    if column not in df.columns:
        return 0
    return int(df[column].nunique())


def _choice_dataset_summary(name: str, df: pd.DataFrame, role: str, used_for_optimization: str) -> dict[str, Any]:
    chosen = df[df["chosen"] == 1] if "chosen" in df.columns else df.iloc[0:0]
    return {
        "Dataset": name,
        "Role": role,
        "Observations": _safe_unique(df, "observation_id"),
        "Rows": len(df),
        "Contexts": _safe_unique(df, "context_id"),
        "Personas": _safe_unique(df, "persona_id"),
        "Used for generator updates?": used_for_optimization,
        "Chosen rows": len(chosen),
    }


def _context_summary(name: str, df: pd.DataFrame, role: str, used_for_optimization: str) -> dict[str, Any]:
    return {
        "Context set": name,
        "Role": role,
        "Contexts": _safe_unique(df, "context_id"),
        "Rows": len(df),
        "Alternatives/context": int(round(len(df) / max(_safe_unique(df, "context_id"), 1))),
        "Used for generator updates?": used_for_optimization,
    }


def _run_config_table(config: dict[str, Any], run_metadata: dict[str, Any] | None) -> pd.DataFrame:
    run = config.get("run", {})
    benchmark = config.get("benchmark", {})
    persona = config.get("persona_generator", {})
    simulation = config.get("simulation", {})
    outer = config.get("outer_optimizer", {})
    rows = [
        ("Run name", run.get("name", "")),
        ("Run seed", run.get("seed", "")),
        ("Stage", (run_metadata or {}).get("stage_resolved", "datasets")),
        ("Mixture components", persona.get("k_components", "")),
        ("Latent features", ", ".join(persona.get("features", []))),
        ("Simulation contexts", benchmark.get("n_sim_contexts", "")),
        ("Anchor contexts", benchmark.get("n_anchor_contexts", "")),
        ("Calibration-intervention contexts", benchmark.get("n_calib_intervention_contexts", "")),
        ("Held-out counterfactual contexts", benchmark.get("n_cf_contexts", "")),
        ("Alternatives per context", benchmark.get("n_alternatives", "")),
        ("Sampled generator personas", simulation.get("n_personas", "")),
        ("Synthetic-human personas", simulation.get("n_human_personas", "")),
        ("Initial synthetic observations", simulation.get("n_obs", "")),
        ("Anchor observations", simulation.get("n_anchor_obs", "")),
        ("Calibration observations", simulation.get("n_calib_obs", "")),
        ("Held-out observations", simulation.get("n_cf_obs", "")),
        ("Outer budget", outer.get("budget", "not run in dataset stage")),
        ("Outer population", outer.get("population", "not run in dataset stage")),
    ]
    return pd.DataFrame(rows, columns=["Quantity", "Value"])


def _persona_segment_table(personas: pd.DataFrame, human_personas: pd.DataFrame) -> pd.DataFrame:
    feature_cols = [c for c in personas.columns if c.startswith("z_")]

    def summarize(df: pd.DataFrame, source: str) -> pd.DataFrame:
        g = df.groupby(["segment_id", "segment_label"], dropna=False)
        out = g.size().rename("Count").reset_index()
        out["Source"] = source
        total = max(len(df), 1)
        out["Share"] = out["Count"] / total
        for col in feature_cols:
            means = g[col].mean().reset_index(name=col)
            out = out.merge(means, on=["segment_id", "segment_label"], how="left")
        return out

    combined = pd.concat(
        [summarize(personas, "initial generator"), summarize(human_personas, "truth population")],
        ignore_index=True,
    )
    rename = {"segment_id": "Segment", "segment_label": "Label"}
    for col in feature_cols:
        rename[col] = col.replace("z_", "mean ")
    combined = combined.rename(columns=rename)
    columns = ["Source", "Segment", "Label", "Count", "Share"] + [rename[c] for c in feature_cols]
    return combined[columns]


def _example_context_table(contexts: pd.DataFrame) -> pd.DataFrame:
    context_id = str(contexts["context_id"].iloc[0])
    cols = ["context_id", "alternative_id", "product_id", "price", "quality", "sustain", "novelty", "brand"]
    out = contexts[contexts["context_id"] == context_id][cols].copy()
    return out.rename(
        columns={
            "context_id": "Context",
            "alternative_id": "Alt.",
            "product_id": "Product",
            "price": "Price",
            "quality": "Quality",
            "sustain": "Sustain.",
            "novelty": "Novelty",
            "brand": "Brand",
        }
    )


def _example_choice_table(d_phi: pd.DataFrame, d_h: pd.DataFrame, n: int = 6) -> pd.DataFrame:
    cols = [
        "dataset",
        "observation_id",
        "context_id",
        "persona_id",
        "persona_segment_label",
        "alternative_id",
        "product_id",
        "choice_prob",
    ]
    chosen = pd.concat(
        [d_phi[d_phi["chosen"] == 1].head(n // 2), d_h[d_h["chosen"] == 1].head(n - n // 2)],
        ignore_index=True,
    )
    out = chosen[cols].copy()
    return out.rename(
        columns={
            "dataset": "Dataset",
            "observation_id": "Obs.",
            "context_id": "Context",
            "persona_id": "Persona",
            "persona_segment_label": "Segment",
            "alternative_id": "Chosen alt.",
            "product_id": "Product",
            "choice_prob": "Sim. probability",
        }
    )


def _load_required_inputs(run_dir: Path) -> dict[str, Any]:
    config_path = run_dir / "config_used.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Expected config_used.yaml in {run_dir}")

    metadata = None
    if (run_dir / "run_metadata.json").exists():
        metadata = load_json(run_dir / "run_metadata.json")

    inputs: dict[str, Any] = {
        "config": load_yaml(config_path),
        "metadata": metadata,
        "personas": read_table(run_dir, "personas"),
        "human_personas": read_table(run_dir, "human_personas"),
        "X_sim": read_table(run_dir, "X_sim"),
        "X_H": read_table(run_dir, "X_H"),
        "X_calib_int": read_table(run_dir, "X_calib_int"),
        "X_cf_base": read_table(run_dir, "X_cf_base"),
        "X_cf": read_table(run_dir, "X_cf"),
        "D_phi_initial": read_table(run_dir, "D_phi_initial"),
        "D_H": read_table(run_dir, "D_H"),
        "D_calib_int_truth": read_table(run_dir, "D_calib_int_truth"),
        "D_cf_truth": read_table(run_dir, "D_cf_truth"),
    }
    if (run_dir / "choice_context_summary.json").exists():
        inputs["choice_context_summary"] = load_json(run_dir / "choice_context_summary.json")
    if (run_dir / "synthetic_choice_summary.json").exists():
        inputs["synthetic_choice_summary"] = load_json(run_dir / "synthetic_choice_summary.json")
    return inputs


def _save_figures(inputs: dict[str, Any], figures_dir: Path) -> dict[str, Path]:
    import matplotlib.pyplot as plt

    figure_paths: dict[str, Path] = {}

    personas = inputs["personas"]
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for label, group in personas.groupby("segment_label"):
        ax.scatter(group["z_price"], group["z_quality"], label=str(label), alpha=0.75, s=35)
    ax.set_title("Initial persona latents by segment")
    ax.set_xlabel("Price sensitivity latent")
    ax.set_ylabel("Quality preference latent")
    ax.axvline(0, linewidth=0.8)
    ax.axhline(0, linewidth=0.8)
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    pdf = figures_dir / "persona_latent_distribution.pdf"
    png = figures_dir / "persona_latent_distribution.png"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, dpi=200, bbox_inches="tight")
    plt.close(fig)
    figure_paths["persona_latent_distribution_pdf"] = pdf
    figure_paths["persona_latent_distribution_png"] = png

    # Dataset flow as a simple diagram.
    fig, ax = plt.subplots(figsize=(8.2, 4.2))
    ax.axis("off")
    boxes = [
        (0.08, 0.72, r"$G_{\phi_0}$\ninitial generator"),
        (0.08, 0.34, r"truth population\nsynthetic humans"),
        (0.36, 0.72, r"sampled personas\n$z \sim p_{\phi_0}(z)$"),
        (0.36, 0.34, r"context sets\n$X_{sim}, X_H, X_{calib\_int}, X_{cf}$"),
        (0.66, 0.54, r"symbolic simulator\n$\pi_\theta(y\mid x,z)$"),
        (0.86, 0.54, r"choice datasets\n$D_{\phi_0},D_H,D_{calib},D_{cf}$"),
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
    arrows = [((0.17, 0.72), (0.27, 0.72)), ((0.17, 0.34), (0.27, 0.34)),
              ((0.45, 0.72), (0.58, 0.59)), ((0.45, 0.34), (0.58, 0.49)),
              ((0.74, 0.54), (0.80, 0.54))]
    for start, end in arrows:
        ax.annotate("", xy=end, xytext=start, xycoords="axes fraction", textcoords="axes fraction",
                    arrowprops=dict(arrowstyle="->", lw=1.4))
    ax.set_title("Controlled synthetic dataset construction")
    fig.tight_layout()
    pdf = figures_dir / "dataset_construction_flow.pdf"
    png = figures_dir / "dataset_construction_flow.png"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, dpi=200, bbox_inches="tight")
    plt.close(fig)
    figure_paths["dataset_construction_flow_pdf"] = pdf
    figure_paths["dataset_construction_flow_png"] = png

    # Choice shares across generated datasets.
    share_rows = []
    for name in ["D_phi_initial", "D_H", "D_calib_int_truth", "D_cf_truth"]:
        df = inputs[name]
        chosen = df[df["chosen"] == 1]
        shares = chosen["alternative_id"].value_counts(normalize=True).sort_index()
        for alt, share in shares.items():
            share_rows.append({"dataset": name, "alternative_id": int(alt), "share": float(share)})
    shares_df = pd.DataFrame(share_rows)
    pivot = shares_df.pivot(index="alternative_id", columns="dataset", values="share").fillna(0.0)
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    pivot.plot(kind="bar", ax=ax)
    ax.set_title("Chosen alternative shares by dataset")
    ax.set_xlabel("Alternative id")
    ax.set_ylabel("Choice share")
    ax.legend(title="Dataset", fontsize=8)
    fig.tight_layout()
    pdf = figures_dir / "choice_share_diagnostics.pdf"
    png = figures_dir / "choice_share_diagnostics.png"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, dpi=200, bbox_inches="tight")
    plt.close(fig)
    figure_paths["choice_share_diagnostics_pdf"] = pdf
    figure_paths["choice_share_diagnostics_png"] = png

    return figure_paths


def _write_appendix_latex(output_dir: Path) -> Path:
    tex = r"""
\section{Controlled Synthetic Dataset Construction}
\label{app:controlled_dataset_construction}

This appendix describes the construction of the controlled synthetic datasets used in the symbolic EIPG experiments. The goal of this stage is to instantiate the objects introduced in the main text---persona latents, choice contexts, simulated choices, anchor data, calibration interventions, and held-out counterfactual data---before any econometric model is fit or any generator update is performed.

\paragraph{Run configuration.}
Table~\ref{tab:app_dataset_run_config} summarizes the configuration used for the run. The configuration fixes the random seed, context sizes, latent features, number of mixture components, number of sampled personas, and observation counts. Each run stores a copy of the configuration in the timestamped run directory, so the tables and figures in this appendix are traceable to the exact dataset-generation settings.

\input{paper_artifacts/appendix_dataset_construction/tables/run_config_summary.tex}

\paragraph{Generator and persona latents.}
The initial generator is a mixture distribution over economic preference vectors,
\[
p_{\phi_0}(z) = \sum_{k=1}^{K_z} w_k\mathcal{N}(z;\mu_k,\Sigma_k).
\]
Table~\ref{tab:app_dataset_persona_segments} summarizes the sampled segment composition and average latent coordinates for both the initial generator and the synthetic-human truth population. Figure~\ref{fig:app_persona_latent_distribution} visualizes the sampled initial persona latents along two economically interpretable dimensions.

\input{paper_artifacts/appendix_dataset_construction/tables/persona_segment_summary.tex}

\begin{figure}[t]
\centering
\includegraphics[width=0.75\linewidth]{paper_artifacts/appendix_dataset_construction/figures/persona_latent_distribution.pdf}
\caption{Initial persona latents sampled from $G_{\phi_0}$, projected onto price sensitivity and quality preference dimensions.}
\label{fig:app_persona_latent_distribution}
\end{figure}

\paragraph{Context sets.}
The benchmark constructs four context sets: simulation contexts $\mathcal{X}_{\mathrm{sim}}$, anchor contexts $\mathcal{X}_H$, calibration-intervention contexts $\mathcal{X}_{\mathrm{calib\_int}}$, and held-out counterfactual contexts $\mathcal{X}_{\mathrm{cf}}$. Table~\ref{tab:app_dataset_context_counts} reports the realized context counts and roles. Table~\ref{tab:app_dataset_example_context} shows one concrete choice context, making the abstract context $x$ visible as a slate of alternatives with observed attributes.

\input{paper_artifacts/appendix_dataset_construction/tables/context_set_counts.tex}
\input{paper_artifacts/appendix_dataset_construction/tables/example_choice_context.tex}

\paragraph{Choice generation.}
Given a persona latent $z$ and context $x$, the symbolic simulator assigns utility $U(j\mid x,z)=z^\top f(x,j)$ to each alternative and samples a choice from the induced logit probabilities. Figure~\ref{fig:app_dataset_flow} summarizes how the initial generator, truth population, context sets, and symbolic simulator produce the four choice datasets. Table~\ref{tab:app_dataset_choice_counts} reports the size and role of each generated choice dataset, while Table~\ref{tab:app_dataset_example_choices} shows example chosen records.

\begin{figure}[t]
\centering
\includegraphics[width=0.9\linewidth]{paper_artifacts/appendix_dataset_construction/figures/dataset_construction_flow.pdf}
\caption{Dataset-construction flow for the controlled synthetic benchmark. The initial generator produces $D_{\phi_0}$, while the synthetic-human truth population produces anchor, calibration-intervention, and held-out counterfactual datasets.}
\label{fig:app_dataset_flow}
\end{figure}

\input{paper_artifacts/appendix_dataset_construction/tables/choice_dataset_counts.tex}
\input{paper_artifacts/appendix_dataset_construction/tables/example_choice_records.tex}

\paragraph{Initial diagnostic.}
Figure~\ref{fig:app_choice_share_diagnostics} compares realized chosen-alternative shares across the initial synthetic dataset, anchor dataset, calibration-intervention truth dataset, and held-out counterfactual truth dataset. This diagnostic is not a calibration result: it is a pre-estimation check that the generated datasets contain meaningful variation across contexts and interventions.

\begin{figure}[t]
\centering
\includegraphics[width=0.85\linewidth]{paper_artifacts/appendix_dataset_construction/figures/choice_share_diagnostics.pdf}
\caption{Chosen-alternative shares across generated datasets before MNL fitting or generator optimization.}
\label{fig:app_choice_share_diagnostics}
\end{figure}
""".strip()
    path = output_dir / "appendix_A_dataset_construction.tex"
    path.write_text(tex + "\n", encoding="utf-8")
    return path


def build_appendix_dataset_artifacts(
    run: str | Path | None = None,
    output_dir: str | Path = "paper_artifacts/appendix_dataset_construction",
    experiment_dir: str | Path | None = None,
) -> AppendixArtifacts:
    """Build all Appendix A dataset-construction tables and figures."""

    run_dir = resolve_run_dir(run=run, experiment_dir=experiment_dir)
    out = Path(output_dir)
    tables_dir, figures_dir = ensure_artifact_dirs(out)
    inputs = _load_required_inputs(run_dir)

    tables: dict[str, Path] = {}

    run_config = _run_config_table(inputs["config"], inputs.get("metadata"))
    tables["run_config_summary"] = write_latex_table(
        run_config,
        tables_dir / "run_config_summary.tex",
        caption="Run configuration used for controlled synthetic dataset construction.",
        label="tab:app_dataset_run_config",
        column_align="ll",
    )

    context_rows = [
        _context_summary("X_sim", inputs["X_sim"], "Simulation contexts for candidate generators", "yes"),
        _context_summary("X_H", inputs["X_H"], "Anchor contexts defining D_H", "yes"),
        _context_summary("X_calib_int", inputs["X_calib_int"], "Calibration-intervention contexts", "yes"),
        _context_summary("X_cf_base", inputs["X_cf_base"], "Unperturbed base for held-out interventions", "no"),
        _context_summary("X_cf", inputs["X_cf"], "Held-out counterfactual contexts", "no"),
    ]
    tables["context_set_counts"] = write_latex_table(
        pd.DataFrame(context_rows),
        tables_dir / "context_set_counts.tex",
        caption="Context sets constructed for the controlled synthetic benchmark.",
        label="tab:app_dataset_context_counts",
        column_align="llrrrr",
    )

    choice_rows = [
        _choice_dataset_summary("D_phi_initial", inputs["D_phi_initial"], "Initial synthetic data induced by G_phi0", "candidate baseline only"),
        _choice_dataset_summary("D_H", inputs["D_H"], "Anchor behavior from truth population", "yes"),
        _choice_dataset_summary("D_calib_int_truth", inputs["D_calib_int_truth"], "Calibration-intervention targets", "yes"),
        _choice_dataset_summary("D_cf_truth", inputs["D_cf_truth"], "Held-out counterfactual truth", "no"),
    ]
    tables["choice_dataset_counts"] = write_latex_table(
        pd.DataFrame(choice_rows),
        tables_dir / "choice_dataset_counts.tex",
        caption="Generated choice datasets produced by the dataset-construction stage.",
        label="tab:app_dataset_choice_counts",
        column_align="llrrrrrr",
    )

    tables["persona_segment_summary"] = write_latex_table(
        _persona_segment_table(inputs["personas"], inputs["human_personas"]),
        tables_dir / "persona_segment_summary.tex",
        caption="Segment composition and average latent coordinates for initial and truth populations.",
        label="tab:app_dataset_persona_segments",
        column_align="lllrrrrrrr",
        digits=2,
    )

    tables["example_choice_context"] = write_latex_table(
        _example_context_table(inputs["X_H"]),
        tables_dir / "example_choice_context.tex",
        caption="Example anchor choice context represented as a slate of alternatives.",
        label="tab:app_dataset_example_context",
        column_align="llrrrrrr",
        digits=2,
    )

    tables["example_choice_records"] = write_latex_table(
        _example_choice_table(inputs["D_phi_initial"], inputs["D_H"]),
        tables_dir / "example_choice_records.tex",
        caption="Example chosen records from the initial synthetic and anchor datasets.",
        label="tab:app_dataset_example_choices",
        column_align="lllllllr",
        digits=3,
    )

    figures = _save_figures(inputs, figures_dir)
    latex_section = _write_appendix_latex(out)

    manifest = {
        "run_dir": str(run_dir),
        "output_dir": str(out),
        "tables": {k: str(v) for k, v in tables.items()},
        "figures": {k: str(v) for k, v in figures.items()},
        "latex_section": str(latex_section),
    }
    manifest_path = out / "artifact_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    return AppendixArtifacts(
        run_dir=run_dir,
        output_dir=out,
        tables=tables,
        figures=figures,
        latex_section=latex_section,
        manifest=manifest_path,
    )
