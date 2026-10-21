"""Download nflverse tables and keep a hashed snapshot of them.

nflverse updates past seasons in place (stat corrections, new columns), so a
second download is not guaranteed to match the first. Every raw file is
recorded in data/manifest.json with its row count, schema hash, and SHA-256.
A rebuild reuses local files that match the manifest, and stops with an error
if a fresh download differs from what the manifest recorded, unless that table
is explicitly refreshed.
"""

import hashlib
import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from typing import Callable

import pandas as pd
import polars as pl

from nflmodel.config import repo_root

log = logging.getLogger(__name__)

MANIFEST_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class TableSpec:
    loader: str          # name of the nflreadpy function
    per_season: bool     # one file per season, or one file for all seasons


TABLES: dict[str, TableSpec] = {
    "schedules": TableSpec("load_schedules", per_season=False),
    "pbp": TableSpec("load_pbp", per_season=True),
    "team_stats": TableSpec("load_team_stats", per_season=True),
}


class SnapshotDriftError(RuntimeError):
    """A fresh download does not match the manifest."""


# ---------------------------------------------------------------- paths

def data_dir(cfg: dict) -> Path:
    return repo_root() / cfg["data"]["dir"]


def manifest_path(cfg: dict) -> Path:
    return data_dir(cfg) / "manifest.json"


def raw_path(table: str, season: int | None) -> Path:
    """Path of a raw file relative to the data directory (also its manifest key)."""
    name = f"{table}_{season}.parquet" if season is not None else f"{table}.parquet"
    return Path("raw") / table / name


def expected_files(cfg: dict) -> list[tuple[str, int | None]]:
    """Every (table, season) file the config asks for. season is None for single-file tables."""
    seasons = range(cfg["data"]["first_season"], cfg["data"]["last_season"] + 1)
    files = []
    for table, opts in cfg["data"]["tables"].items():
        spec = _spec(table, opts)
        if spec.per_season:
            files.extend((table, s) for s in seasons)
        else:
            files.append((table, None))
    return files


def _spec(table: str, opts: dict) -> TableSpec:
    if table not in TABLES:
        raise ValueError(f"unknown table {table!r}; known tables: {sorted(TABLES)}")
    spec = TABLES[table]
    if opts.get("per_season", spec.per_season) != spec.per_season:
        raise ValueError(f"config per_season for {table!r} does not match TABLES")
    return spec


def _loader_kwargs(opts: dict) -> dict:
    return {k: v for k, v in opts.items() if k != "per_season"}


# ---------------------------------------------------------------- download

def _loader(table: str) -> Callable[..., pl.DataFrame]:
    import nflreadpy
    from nflreadpy.config import update_config

    # Bypass nflreadpy's own cache so every download reflects upstream right now.
    update_config(cache_mode="off", timeout=120)
    return getattr(nflreadpy, TABLES[table].loader)


def download_table(table: str, season: int | None, **kwargs) -> pl.DataFrame:
    """Download one season of a table, or all seasons when season is None."""
    loader = _loader(table)
    return loader(seasons=season if season is not None else True, **kwargs)


# ---------------------------------------------------------------- manifest

def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _columns_sha256(df: pl.DataFrame) -> str:
    schema = "\n".join(f"{name}:{dtype}" for name, dtype in df.schema.items())
    return hashlib.sha256(schema.encode("utf-8")).hexdigest()


def manifest_entry(path: Path, table: str, season: int | None, df: pl.DataFrame) -> dict:
    return {
        "table": table,
        "season": season,
        "loader": TABLES[table].loader,
        "downloaded_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "rows": df.height,
        "n_columns": df.width,
        "columns": df.columns,
        "columns_sha256": _columns_sha256(df),
        "sha256": file_sha256(path),
        "bytes": path.stat().st_size,
        "nflreadpy_version": version("nflreadpy"),
        "polars_version": version("polars"),
    }


def load_manifest(path: Path) -> dict:
    if not path.exists():
        return {"schema_version": MANIFEST_SCHEMA_VERSION, "files": {}}
    return json.loads(path.read_text(encoding="utf-8"))


def write_manifest(manifest: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    path.write_text(text, encoding="utf-8", newline="\n")


def _describe_drift(key: str, old: dict, new: dict) -> str:
    parts = []
    if old["rows"] != new["rows"]:
        parts.append(f"rows {old['rows']} -> {new['rows']}")
    if old["columns_sha256"] != new["columns_sha256"]:
        before, after = set(old["columns"]), set(new["columns"])
        detail = f"columns {old['n_columns']} -> {new['n_columns']}"
        if after - before:
            detail += f", added: {', '.join(sorted(after - before))}"
        if before - after:
            detail += f", removed: {', '.join(sorted(before - after))}"
        if before == after:
            detail += ", same names with different types"
        parts.append(detail)
    if not parts:
        parts.append("same shape and columns, different values")
    return f"{key} changed upstream: " + "; ".join(parts)


# ---------------------------------------------------------------- snapshot

def build_snapshot(cfg: dict, refresh: set[str] | None = None) -> dict:
    """Download whatever is missing and check it against the manifest.

    A local file whose hash matches the manifest is reused without a download.
    Tables named in `refresh` are downloaded again and their manifest entries
    replaced. For any other file, a download that differs from the manifest is
    not saved and raises SnapshotDriftError after all files are checked.
    """
    refresh = set(refresh or ())
    unknown = refresh - set(cfg["data"]["tables"])
    if unknown:
        raise ValueError(f"cannot refresh tables not in config: {sorted(unknown)}")

    root = data_dir(cfg)
    mpath = manifest_path(cfg)
    manifest = load_manifest(mpath)
    files = manifest["files"]
    drift = []

    for table, season in expected_files(cfg):
        key = raw_path(table, season).as_posix()
        path = root / key
        old = files.get(key)

        if table not in refresh and old and path.exists() and file_sha256(path) == old["sha256"]:
            log.info("ok        %s", key)
            continue

        log.info("download  %s", key)
        df = download_table(table, season, **_loader_kwargs(cfg["data"]["tables"][table]))
        if season is None and "season" in df.columns:
            # Keep later, still-changing seasons out of the frozen snapshot.
            df = df.filter(pl.col("season") <= cfg["data"]["last_season"])

        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        df.write_parquet(tmp, compression="zstd")
        new = manifest_entry(tmp, table, season, df)

        if old is None or table in refresh:
            tmp.replace(path)
            files[key] = new
            write_manifest(manifest, mpath)
        elif new["sha256"] == old["sha256"]:
            tmp.replace(path)  # local copy was missing or damaged; upstream unchanged
        else:
            drift.append(_describe_drift(key, old, new))
            tmp.unlink()

    write_manifest(manifest, mpath)
    if drift:
        raise SnapshotDriftError("\n".join(drift))
    return manifest


def verify_snapshot(cfg: dict) -> list[str]:
    """Rehash local files against the manifest without downloading. Empty list means OK."""
    root = data_dir(cfg)
    files = load_manifest(manifest_path(cfg))["files"]
    expected = {raw_path(t, s).as_posix() for t, s in expected_files(cfg)}
    problems = []

    for key in sorted(expected):
        path = root / key
        if key not in files:
            problems.append(f"{key}: not in manifest")
        elif not path.exists():
            problems.append(f"{key}: missing on disk")
        elif file_sha256(path) != files[key]["sha256"]:
            problems.append(f"{key}: hash does not match manifest")
    for key in sorted(set(files) - expected):
        problems.append(f"{key}: in manifest but not requested by config")
    return problems


def read_raw(table: str, seasons: list[int] | None = None, cfg: dict | None = None) -> pd.DataFrame:
    """Read raw snapshot files as pandas. seasons=None reads every snapshot season."""
    if cfg is None:
        from nflmodel.config import load_config
        cfg = load_config()
    root = data_dir(cfg)
    spec = _spec(table, cfg["data"]["tables"][table])

    if spec.per_season:
        if seasons is None:
            seasons = list(range(cfg["data"]["first_season"], cfg["data"]["last_season"] + 1))
        df = pl.concat(
            [pl.read_parquet(root / raw_path(table, s)) for s in seasons],
            how="diagonal_relaxed",
        )
    else:
        df = pl.read_parquet(root / raw_path(table, None))
        if seasons is not None:
            df = df.filter(pl.col("season").is_in(seasons))
    return df.to_pandas()
