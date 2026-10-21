import copy

import pytest
import yaml

from nflmodel.config import load_config, repo_root


def test_repo_root_contains_config():
    assert (repo_root() / "config.yaml").is_file()


def test_config_loads(cfg):
    assert cfg["evaluation"]["holdout_season"] == 2025
    assert cfg["evaluation"]["holdout_locked"] is True
    assert {"schedules", "pbp", "team_stats"} <= set(cfg["data"]["tables"])


def _write(tmp_path, cfg):
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return path


@pytest.mark.parametrize(
    "section, key, value",
    [
        ("evaluation", "holdout_season", 2024),   # overlaps the test seasons
        ("evaluation", "holdout_season", 2030),   # not in the snapshot
        ("data", "first_season", 2016),           # starts after the first test season
    ],
)
def test_invalid_seasons_rejected(cfg, tmp_path, section, key, value):
    bad = copy.deepcopy(cfg)
    bad[section][key] = value
    with pytest.raises(ValueError):
        load_config(_write(tmp_path, bad))


def test_too_few_seeds_rejected(cfg, tmp_path):
    bad = copy.deepcopy(cfg)
    bad["seeds"] = [1, 2]
    with pytest.raises(ValueError):
        load_config(_write(tmp_path, bad))
