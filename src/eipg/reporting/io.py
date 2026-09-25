"""I/O helpers for building paper appendix artifacts.

These helpers intentionally live outside the core EIPG loop. They read completed run
folders and write tables/figures for the paper appendix.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


def resolve_run_dir(run: str | Path | None = None, experiment_dir: str | Path | None = None) -> Path:
    """Resolve a timestamped run directory.

    Parameters
    ----------
    run:
        Explicit run directory. If supplied, it is returned after validation.
    experiment_dir:
        Parent experiment directory containing ``LATEST_RUN.txt``. Used when
        ``run`` is omitted.
    """

    if run is not None:
        path = Path(run).expanduser().resolve()
    else:
        if experiment_dir is None:
            experiment_dir = Path("outputs/controlled_synthetic_paperlike")
        parent = Path(experiment_dir).expanduser().resolve()
        pointer = parent / "LATEST_RUN.txt"
        if not pointer.exists():
            raise FileNotFoundError(
                f"Could not find {pointer}. Run the pipeline first or pass --run explicitly."
            )
        path = Path(pointer.read_text(encoding="utf-8").strip()).expanduser().resolve()

    if not path.exists():
        raise FileNotFoundError(f"Run directory does not exist: {path}")
    return path


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def read_table(run_dir: Path, stem: str) -> pd.DataFrame:
    """Read ``stem.parquet`` or fallback ``stem.csv`` from a run directory."""

    parquet_path = run_dir / f"{stem}.parquet"
    csv_path = run_dir / f"{stem}.csv"
    if parquet_path.exists():
        return pd.read_parquet(parquet_path)
    if csv_path.exists():
        return pd.read_csv(csv_path)
    raise FileNotFoundError(f"Could not find {stem}.parquet or {stem}.csv in {run_dir}")


def ensure_artifact_dirs(output_dir: Path) -> tuple[Path, Path]:
    tables_dir = output_dir / "tables"
    figures_dir = output_dir / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    return tables_dir, figures_dir


def latex_escape(value: Any) -> str:
    """Escape a small value for use in a LaTeX table."""

    text = str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def format_cell(value: Any, digits: int = 3) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def write_latex_table(
    df: pd.DataFrame,
    path: Path,
    caption: str,
    label: str,
    column_align: str | None = None,
    digits: int = 3,
    small: bool = True,
) -> Path:
    """Write a simple booktabs LaTeX table.

    The output is a complete table environment so the paper can include it with
    ``\input{...}``.
    """

    if column_align is None:
        column_align = "l" * len(df.columns)

    lines: list[str] = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    if small:
        lines.append(r"\small")
    lines.append(rf"\begin{{tabular}}{{{column_align}}}")
    lines.append(r"\toprule")
    header = " & ".join(latex_escape(col) for col in df.columns) + r" \\"
    lines.append(header)
    lines.append(r"\midrule")
    for _, row in df.iterrows():
        cells = [latex_escape(format_cell(row[col], digits=digits)) for col in df.columns]
        lines.append(" & ".join(cells) + r" \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(rf"\caption{{{latex_escape(caption)}}}")
    lines.append(rf"\label{{{label}}}")
    lines.append(r"\end{table}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
