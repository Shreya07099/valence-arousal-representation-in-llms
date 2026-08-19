"""Load and filter the GoEmotions dataset to single-label examples."""

from collections import defaultdict
from typing import Dict, List, Optional

from datasets import load_dataset

GO_EMOTIONS_DATASET = "google-research-datasets/go_emotions"
GO_EMOTIONS_CONFIG = "simplified"

DEFAULT_OUTPUT = "data/processed/go_emotions_single_label.jsonl"


def _load_all_splits():
    return load_dataset(GO_EMOTIONS_DATASET, GO_EMOTIONS_CONFIG)


def pull_single_label_examples(
    max_per_class: Optional[int] = None,
    labels: Optional[List[str]] = None,
) -> List[dict]:
    """
    Return single-label examples from all GoEmotions splits combined.

    Each item is a dict with keys: id, text, label.
    """
    ds_dict = _load_all_splits()
    label_names = ds_dict["train"].features["labels"].feature.names
    allowed_labels = set(labels) if labels is not None else None
    counts: Dict[str, int] = defaultdict(int)
    examples: List[dict] = []

    for split_ds in ds_dict.values():
        for ex in split_ds:
            if len(ex["labels"]) != 1:
                continue

            label = label_names[ex["labels"][0]]
            if allowed_labels is not None and label not in allowed_labels:
                continue
            if max_per_class is not None and counts[label] >= max_per_class:
                continue

            examples.append({"id": ex["id"], "text": ex["text"], "label": label})
            counts[label] += 1

    return examples


def get_single_label_examples(
    max_per_class: Optional[int] = None,
    labels: Optional[List[str]] = None,
) -> Dict[str, List[str]]:
    """Group single-label examples by emotion name (label -> list of texts)."""
    buckets: Dict[str, List[str]] = defaultdict(list)

    for ex in pull_single_label_examples(
        max_per_class=max_per_class,
        labels=labels,
    ):
        buckets[ex["label"]].append(ex["text"])

    return dict(buckets)
