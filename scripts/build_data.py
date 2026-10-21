"""Build or verify the raw nflverse snapshot.

    python -m scripts.build_data                    download missing files, check against manifest
    python -m scripts.build_data --verify           rehash local files only, no network
    python -m scripts.build_data --refresh pbp      accept a new upstream version of a table
"""

import argparse
import logging
import sys
from pathlib import Path

from nflmodel.config import load_config
from nflmodel.data import SnapshotDriftError, build_snapshot, verify_snapshot


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=None, help="path to config.yaml")
    parser.add_argument("--verify", action="store_true", help="rehash local files against the manifest")
    parser.add_argument("--refresh", action="append", default=[], metavar="TABLE",
                        help="re-download TABLE and replace its manifest entries (repeatable)")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    cfg = load_config(args.config)

    if not args.verify:
        try:
            build_snapshot(cfg, refresh=set(args.refresh))
        except SnapshotDriftError as e:
            print(f"\nUpstream data changed since the snapshot:\n{e}", file=sys.stderr)
            print("Local files were left as they were. To accept the new data, rerun with "
                  "--refresh TABLE and log the change in docs/RESEARCH.md.", file=sys.stderr)
            return 1

    problems = verify_snapshot(cfg)
    if problems:
        print("\n".join(problems), file=sys.stderr)
        return 1
    print("Snapshot OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
