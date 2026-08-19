# Valence and Arousal Representation in LLMs

This repository studies how large language models internally represent **valence** (pleasant vs. unpleasant) and **arousal** (calm vs. activated). The current code prepares a single-label emotion dataset from [GoEmotions](https://huggingface.co/datasets/google-research-datasets/go_emotions) and queries a small Qwen model for **model-native valence–arousal coordinates** of each emotion label.

The prompting setup follows the three templates in Appendix A.1 of Sun et al. (2026). Coordinates are averaged across templates and clamped to `[-1, +1]`.

**GPU work is done in Google Colab.** Loading Qwen and scoring valence/arousal is not practical on a local CPU. Use the notebook in [`notebooks/gpu_emotion_extraction_and_valence.ipynb`](notebooks/gpu_emotion_extraction_and_valence.ipynb) (or the [live Colab copy](https://colab.research.google.com/drive/1TEOHKThV3pn3VaTYYUxC3al-CW18DEkf)).

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1TEOHKThV3pn3VaTYYUxC3al-CW18DEkf)

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
├── notebooks/
│   └── gpu_emotion_extraction_and_valence.ipynb
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

Local CPU is enough for dataset scripts that only read or write JSONL. **Emotion extraction with Hugging Face `datasets` plus Qwen valence ratings need a GPU.** Those steps live in the Colab notebook so they do not have to run on a laptop CPU.

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

## 2. GPU notebook: emotion extraction and valence ratings

Run this on **Google Colab with a GPU** (Runtime → Change runtime type → GPU). A local CPU will be too slow or run out of memory for Qwen.

Notebook in this repo: [`notebooks/gpu_emotion_extraction_and_valence.ipynb`](notebooks/gpu_emotion_extraction_and_valence.ipynb)

Live Colab: [https://colab.research.google.com/drive/1TEOHKThV3pn3VaTYYUxC3al-CW18DEkf](https://colab.research.google.com/drive/1TEOHKThV3pn3VaTYYUxC3al-CW18DEkf)

The notebook:

1. Checks that CUDA is available
2. Extracts single-label GoEmotions examples into one JSONL file
3. Queries `Qwen/Qwen3-1.7B` for valence and arousal of all 28 labels
4. Saves `outputs/qwen_native_coordinates.json` (and optionally copies it to Google Drive)

If you already have a local GPU, the same commands work outside Colab:

```bash
python scripts/query_coordinates.py
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
- Qwen valence–arousal querying (GPU / Colab notebook)

Planned:

- representation analysis
- activation steering along valence and arousal (`steering/`)

## References

- Demszky, D., et al. (2020). *GoEmotions: A Dataset of Fine-Grained Emotions.*
- Sun et al. (2026). Templates in Appendix A.1 for rating emotion labels on valence and arousal.
