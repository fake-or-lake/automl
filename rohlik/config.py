from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Config:
    data_dir: Path = Path("data")
    output_dir: Path = Path("submissions")
    horizon_days: int = 14
    random_state: int = 42

    eda_sample_rows: int = 300_000
