"""Snapshot and manifest logic, tested against a fake nflverse so no network is needed."""

import copy
import json

import polars as pl
import pytest

from nflmodel import data
from nflmodel.data import (
    SnapshotDriftError,
    build_snapshot,
    file_sha256,
    load_manifest,
    manifest_path,
    raw_path,
    read_raw,
    verify_snapshot,
    write_manifest,
)


class FakeUpstream:
    """Stands in for nflreadpy. Tests edit `changes` to simulate upstream updates."""

    def __init__(self):
        self.calls = []
        self.changes = {}   # (table, season) -> function(df) -> df

    def loader_for(self, table):
        def load(seasons, **kwargs):
            self.calls.append((table, seasons, kwargs))
            if seasons is True:
                df = pl.DataFrame({"season": [2020, 2021, 2022], "game_id": ["a", "b", "c"]})
                season = None
            else:
                df = pl.DataFrame({"season": [seasons] * 3, "epa": [0.1, -0.2, 0.3]})
                season = seasons
            change = self.changes.get((table, season))
            return change(df) if change else df
        return load


@pytest.fixture
def upstream(monkeypatch):
    fake = FakeUpstream()
    monkeypatch.setattr(data, "_loader", fake.loader_for)
    return fake


@pytest.fixture
def small_cfg(cfg, tmp_path):
    small = copy.deepcopy(cfg)
    small["data"]["dir"] = str(tmp_path)
    small["data"]["first_season"] = 2020
    small["data"]["last_season"] = 2021
    return small


def _key(table, season):
    return raw_path(table, season).as_posix()


def test_build_writes_files_and_manifest(small_cfg, upstream, tmp_path):
    manifest = build_snapshot(small_cfg)

    expected = {_key("schedules", None)} | {
        _key(t, s) for t in ("pbp", "team_stats") for s in (2020, 2021)
    }
    assert set(manifest["files"]) == expected
    for key, entry in manifest["files"].items():
        assert file_sha256(tmp_path / key) == entry["sha256"]
        assert entry["rows"] == 3 or key.startswith("raw/schedules")
    assert verify_snapshot(small_cfg) == []


def test_loader_kwargs_come_from_config(small_cfg, upstream):
    build_snapshot(small_cfg)
    team_calls = [kw for table, _, kw in upstream.calls if table == "team_stats"]
    assert team_calls and all(kw == {"summary_level": "week"} for kw in team_calls)


def test_schedules_drop_seasons_after_snapshot(small_cfg, upstream):
    build_snapshot(small_cfg)
    sched = read_raw("schedules", cfg=small_cfg)
    assert sched["season"].max() == 2021


def test_rerun_with_matching_files_does_not_download(small_cfg, upstream):
    build_snapshot(small_cfg)
    upstream.calls.clear()
    build_snapshot(small_cfg)
    assert upstream.calls == []


def test_missing_file_restored_when_upstream_unchanged(small_cfg, upstream, tmp_path):
    first = build_snapshot(small_cfg)
    key = _key("pbp", 2021)
    entry_before = dict(first["files"][key])
    (tmp_path / key).unlink()

    second = build_snapshot(small_cfg)
    assert (tmp_path / key).exists()
    assert second["files"][key] == entry_before   # same hash, original timestamp kept


def test_upstream_change_raises_and_keeps_old_data(small_cfg, upstream, tmp_path):
    build_snapshot(small_cfg)
    key = _key("pbp", 2021)
    old_bytes = (tmp_path / key).read_bytes()
    old_manifest = load_manifest(manifest_path(small_cfg))

    # Simulate the kind of change that broke the old notebook: a new column appears.
    upstream.changes[("pbp", 2021)] = lambda df: df.with_columns(game_id=pl.lit("x"))
    (tmp_path / key).write_bytes(b"damaged")   # force a re-download of this file

    with pytest.raises(SnapshotDriftError, match=r"pbp_2021.*added: game_id"):
        build_snapshot(small_cfg)
    assert (tmp_path / key).read_bytes() == b"damaged"           # new data not saved
    assert load_manifest(manifest_path(small_cfg)) == old_manifest
    assert old_bytes != b"damaged"


def test_value_change_reported_even_with_same_shape(small_cfg, upstream, tmp_path):
    build_snapshot(small_cfg)
    key = _key("pbp", 2020)
    upstream.changes[("pbp", 2020)] = lambda df: df.with_columns(pl.col("epa") * 2)
    (tmp_path / key).unlink()

    with pytest.raises(SnapshotDriftError, match="different values"):
        build_snapshot(small_cfg)


def test_refresh_accepts_upstream_change(small_cfg, upstream):
    build_snapshot(small_cfg)
    upstream.changes[("pbp", 2021)] = lambda df: pl.concat([df, df])

    manifest = build_snapshot(small_cfg, refresh={"pbp"})
    assert manifest["files"][_key("pbp", 2021)]["rows"] == 6
    assert verify_snapshot(small_cfg) == []


def test_refresh_of_unknown_table_rejected(small_cfg, upstream):
    with pytest.raises(ValueError):
        build_snapshot(small_cfg, refresh={"injuries"})


def test_verify_detects_tampering_and_missing_files(small_cfg, upstream, tmp_path):
    build_snapshot(small_cfg)
    (tmp_path / _key("pbp", 2020)).write_bytes(b"tampered")
    (tmp_path / _key("team_stats", 2021)).unlink()

    problems = verify_snapshot(small_cfg)
    assert any("pbp_2020" in p and "hash" in p for p in problems)
    assert any("team_stats_2021" in p and "missing" in p for p in problems)


def test_verify_flags_manifest_entries_config_no_longer_wants(small_cfg, upstream):
    build_snapshot(small_cfg)
    small_cfg["data"]["first_season"] = 2021
    assert any("pbp_2020" in p and "not requested" in p for p in verify_snapshot(small_cfg))


def test_manifest_is_sorted_utf8_json(tmp_path):
    manifest = {"schema_version": 1, "files": {"b": {"z": 1, "a": 2}, "a": {}}}
    path = tmp_path / "manifest.json"
    write_manifest(manifest, path)

    raw = path.read_bytes()
    assert b"\r\n" not in raw
    text = raw.decode("utf-8")
    assert json.loads(text) == manifest
    assert text.index('"a"') < text.index('"b"')
    assert load_manifest(path) == manifest


def test_read_raw_selects_seasons(small_cfg, upstream):
    build_snapshot(small_cfg)
    pbp = read_raw("pbp", seasons=[2021], cfg=small_cfg)
    assert set(pbp["season"]) == {2021}
    assert len(read_raw("pbp", cfg=small_cfg)) == 6


@pytest.mark.network
def test_real_schedules_download():
    df = data.download_table("schedules", None)
    assert {"game_id", "season", "spread_line", "home_moneyline"} <= set(df.columns)
