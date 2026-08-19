"""Print the number of examples per emotion label in the processed JSONL dataset."""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.go_emotions import DEFAULT_OUTPUT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Count examples per emotion label in a Go Emotions JSONL file."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(DEFAULT_OUTPUT),
        help="Path to the processed JSONL file.",
    )
    return parser.parse_args()


def count_labels(path: Path) -> Counter:
    counts: Counter = Counter()
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            counts[json.loads(line)["label"]] += 1
    return counts


def main() -> None:
    args = parse_args()
    if not args.input.is_file():
        raise FileNotFoundError(f"Dataset not found: {args.input}")

    counts = count_labels(args.input)
    total = sum(counts.values())

    print(f"Dataset: {args.input}")
    print(f"Total examples: {total}")
    print(f"Labels ({len(counts)}):")
    for label, count in sorted(counts.items(), key=lambda item: (-item[1], item[0])):
        print(f"  {label}: {count}")


if __name__ == "__main__":
    main()
