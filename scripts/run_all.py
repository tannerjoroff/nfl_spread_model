"""Rebuild everything from raw data to results. Each phase adds its step here."""

from scripts import baselines, build_data


def main() -> int:
    steps = [
        ("build_data", lambda: build_data.main([])),
        ("baselines", lambda: baselines.main([])),
    ]
    for name, step in steps:
        print(f"== {name}")
        code = step()
        if code != 0:
            print(f"{name} failed with exit code {code}")
            return code
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
