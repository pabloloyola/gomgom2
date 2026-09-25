"""Lightweight progress and ETA reporting for long symbolic experiments."""
from __future__ import annotations

from dataclasses import dataclass, field
from time import monotonic


@dataclass
class ProgressTracker:
    """Track completed work units and render a stable human-readable ETA."""

    total: int
    label: str = "progress"
    started_at: float = field(default_factory=monotonic)
    completed: int = 0

    def __post_init__(self) -> None:
        if self.total <= 0:
            raise ValueError("total must be positive")

    def update(self, n: int = 1) -> dict[str, float | int | str]:
        self.completed = min(self.total, self.completed + int(n))
        elapsed = max(monotonic() - self.started_at, 1.0e-9)
        rate = self.completed / elapsed
        remaining = max(self.total - self.completed, 0)
        eta_seconds = remaining / rate if rate > 0 else float("inf")
        return {
            "label": self.label,
            "completed": self.completed,
            "total": self.total,
            "fraction": self.completed / self.total,
            "elapsed_seconds": elapsed,
            "rate_per_second": rate,
            "eta_seconds": eta_seconds,
        }

    def format(self, *, prefix: str = "") -> str:
        elapsed = max(monotonic() - self.started_at, 1.0e-9)
        rate = self.completed / elapsed
        remaining = max(self.total - self.completed, 0)
        eta = remaining / rate if rate > 0 else float("inf")
        percent = 100.0 * self.completed / self.total
        return (
            f"{prefix}{self.completed}/{self.total} = {percent:.1f}% | "
            f"elapsed {_format_seconds(elapsed)} | ETA {_format_seconds(eta)}"
        )


def _format_seconds(seconds: float) -> str:
    if seconds == float("inf"):
        return "unknown"
    seconds = max(int(round(seconds)), 0)
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}h{minutes:02d}m"
    if minutes:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"
