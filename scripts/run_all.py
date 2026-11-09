"""Rebuild everything from raw data to results. Each phase adds its step here."""

from scripts import baselines, build_data, build_features, train_margin


def main() -> int:
    steps = [
        ("build_data", lambda: build_data.main([])),
        ("build_features", lambda: build_features.main([])),
        ("baselines", lambda: baselines.main([])),
        ("train_margin", lambda: train_margin.main([])),
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
