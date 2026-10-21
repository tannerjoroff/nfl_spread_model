"""Load config.yaml and resolve project paths."""

from pathlib import Path

import yaml


def repo_root() -> Path:
    """Return the repository root (the folder that contains config.yaml)."""
    return Path(__file__).resolve().parents[2]


def load_config(path: Path | None = None) -> dict:
    """Read config.yaml and check the values later phases rely on."""
    path = Path(path) if path is not None else repo_root() / "config.yaml"
    with path.open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    _validate(cfg)
    return cfg


def _validate(cfg: dict) -> None:
    data = cfg["data"]
    ev = cfg["evaluation"]

    if len(cfg["seeds"]) < 5:
        raise ValueError("config needs at least 5 seeds")
    if data["first_season"] > data["last_season"]:
        raise ValueError("data.first_season is after data.last_season")
    if not data["first_season"] <= ev["first_test_season"] <= ev["last_test_season"]:
        raise ValueError("test seasons must fall inside the data seasons")
    if ev["holdout_season"] <= ev["last_test_season"]:
        raise ValueError("holdout_season must come after the last test season")
    if ev["holdout_season"] > data["last_season"]:
        raise ValueError("holdout_season is not in the data snapshot")
