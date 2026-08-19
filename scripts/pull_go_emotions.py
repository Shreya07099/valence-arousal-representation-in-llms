"""Pull single-label examples from the Go Emotions dataset into one JSONL file."""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.go_emotions import DEFAULT_OUTPUT, pull_single_label_examples


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract Go Emotions examples with exactly one label (all splits combined)."
    )
    parser.add_argument(
        "--max-per-class",
        type=int,
        default=None,
        help="Maximum number of examples to keep per emotion label.",
    )
    parser.add_argument(
        "--labels",
        nargs="+",
        default=None,
        help="Optional subset of emotion labels to include.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(DEFAULT_OUTPUT),
        help="Output JSONL file path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    examples = pull_single_label_examples(
        max_per_class=args.max_per_class,
        labels=args.labels,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    counts = Counter(ex["label"] for ex in examples)
    print(f"Saved {len(examples)} examples to {args.output}")
    print(f"Labels ({len(counts)}): {dict(sorted(counts.items()))}")


if __name__ == "__main__":
    main()
