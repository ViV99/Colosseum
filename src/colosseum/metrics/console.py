"""One progress line per agent, printed by the main process."""

from __future__ import annotations

import logging
import math

logger = logging.getLogger("colosseum.progress")


def _fmt(value: float | None, spec: str) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "-"
    return format(value, spec)


class ConsoleReporter:
    def __init__(self, total_timesteps: int, log: logging.Logger | None = None) -> None:
        self._total = max(1, int(total_timesteps))
        self._log = log or logger

    def format_line(self, agent_id: str, *, train_step: int, env_steps: int, fps: float,
                    loss: float | None, entropy: float | None, return_mean: float | None,
                    wr_vs_past: float | None, wr_arena: float | None,
                    bt: tuple[float, float] | None = None) -> str:
        """One progress line. ``bt``: the agent's live Bradley-Terry rating of its latest version bucket and the
        half width of its 95 % interval (Elo units, SP4)."""
        parts = [
            f"[{agent_id}] step {train_step}",
            f"{100.0 * min(1.0, env_steps / self._total):5.1f}% budget",
            f"{fps:,.0f} env-steps/s",
            f"loss {_fmt(loss, '.4f')}",
            f"entropy {_fmt(entropy, '.3f')}",
            f"return {_fmt(return_mean, '.3f')}",
        ]
        if wr_vs_past is not None:
            parts.append(f"wr_vs_past {wr_vs_past:.2f}")
        if wr_arena is not None:
            parts.append(f"wr_arena {wr_arena:.2f}")
        if bt is not None:
            parts.append(f"bt {round(bt[0])} ± {round(bt[1])}")      # round(): never "-0"
        return " | ".join(parts)

    def emit(self, lines: list[str]) -> None:
        for line in lines:
            self._log.info(line)
