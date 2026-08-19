# Valence and Arousal Representation in LLMs

This repository studies how large language models internally represent **valence** (pleasant vs. unpleasant) and **arousal** (calm vs. activated). The current code prepares a single-label emotion dataset from [GoEmotions](https://huggingface.co/datasets/google-research-datasets/go_emotions) and queries a small Qwen model for **model-native valence–arousal coordinates** of each emotion label.

The prompting setup follows the three templates in Appendix A.1 of Sun et al. (2026). Coordinates are averaged across templates and clamped to `[-1, +1]`.

## Repository layout

```
.
├── config.py                 # 28 emotion labels (27 GoEmotions + neutral)
├── requirements.txt
├── data/
│   ├── go_emotions.py        # load and filter GoEmotions
│   └── processed/            # merged single-label JSONL
├── scripts/
│   ├── pull_go_emotions.py   # write the JSONL dataset
│   ├── count_emotion_labels.py
│   └── query_coordinates.py  # query Qwen for VA coordinates
├── outputs/                  # generated coordinate JSON (not committed)
└── steering/                 # reserved for later steering experiments
```

## Setup

Python 3.10+ is recommended.

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

GPU is optional for dataset extraction. Querying Qwen needs a GPU (Colab T4 is enough for `Qwen/Qwen3-1.7B`).

## 1. Build the single-label dataset

GoEmotions is multi-label. We keep only comments with **exactly one** emotion, and merge the original train / validation / test splits into one file.

```bash
python scripts/pull_go_emotions.py
```

Output: `data/processed/go_emotions_single_label.jsonl` (~45k examples).

Optional flags:

```bash
python scripts/pull_go_emotions.py --max-per-class 100
python scripts/pull_go_emotions.py --labels anger joy sadness
python scripts/pull_go_emotions.py --output path/to/file.jsonl
```

Count examples per label:

```bash
python scripts/count_emotion_labels.py
```

## 2. Query model-native coordinates

This step loads Qwen, asks for valence and arousal of each of the 28 labels using three prompt templates, then writes averaged coordinates.

**Local GPU**

```bash
python scripts/query_coordinates.py
```

**Google Colab** (GPU runtime):

```python
!pip install -q "transformers>=4.51" accelerate torch
!python scripts/query_coordinates.py --output /content/drive/MyDrive/qwen_native_coordinates.json
```

Useful flags:

```bash
python scripts/query_coordinates.py --model Qwen/Qwen3-1.7B
python scripts/query_coordinates.py --output outputs/qwen_native_coordinates.json
```

Default output: `outputs/qwen_native_coordinates.json`

```json
{
  "anger": { "valence": -0.72, "arousal": 0.81 },
  "joy": { "valence": 0.85, "arousal": 0.64 },
  "neutral": { "valence": 0.00, "arousal": 0.00 }
}
```

If a template fails to parse as JSON, that template is skipped. If all three fail for a label, the script records `(0.0, 0.0)` and prints a warning.

## Dataset notes

- Source: `google-research-datasets/go_emotions`, config `simplified`
- Labels: 27 GoEmotions emotions plus `neutral` (see `config.py`)
- Each JSONL line: `{"id": "...", "text": "...", "label": "..."}`
- First download of GoEmotions and Qwen requires internet access (and enough disk for the Hugging Face cache)

## Status

Implemented:

- single-label GoEmotions extraction
- per-label counts
- Qwen valence–arousal querying

Planned:

- representation analysis
- activation steering along valence and arousal (`steering/`)

## References

- Demszky, D., et al. (2020). *GoEmotions: A Dataset of Fine-Grained Emotions.*
- Sun et al. (2026). Templates in Appendix A.1 for rating emotion labels on valence and arousal.
