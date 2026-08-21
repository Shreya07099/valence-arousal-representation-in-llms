# Valence and Arousal Representation in LLMs

This repository studies how large language models internally represent **valence** (pleasant vs. unpleasant) and **arousal** (calm vs. activated). It builds a single-label emotion dataset from [GoEmotions](https://huggingface.co/datasets/google-research-datasets/go_emotions), queries a small Qwen model for **model-native valence–arousal coordinates** of each emotion label, then extracts and validates linear valence/arousal directions inside the model's hidden states.

The prompting setup follows the three templates in Appendix A.1 of Sun et al. (2026). Coordinates are averaged across templates and clamped to `[-1, +1]`.

**GPU work is done in Google Colab.** Loading Qwen and scoring valence/arousal is not practical on a local CPU. Use the notebook in [`notebooks/gpu_emotion_extraction_and_valence.ipynb`](notebooks/gpu_emotion_extraction_and_valence.ipynb) (or the [live Colab copy](https://colab.research.google.com/drive/1TEOHKThV3pn3VaTYYUxC3al-CW18DEkf)).

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1TEOHKThV3pn3VaTYYUxC3al-CW18DEkf)

## Repository layout

```
.
├── config.py                    # 28 emotion labels (27 GoEmotions + neutral)
├── requirements.txt
├── data/
│   ├── go_emotions.py            # load and filter GoEmotions
│   ├── processed/                # merged single-label JSONL
│   ├── qwen-native-coordinates.json   # model-native valence/arousal targets, 27 emotions
│   ├── emotion_avg_vectors.pt         # per-emotion average hidden states, 29 layers
│   ├── emotion_contrast_vectors.pt    # v_e = mean_e - mean_neutral, 29 layers
│   ├── emotion_centered_vectors.pt    # mean-centered contrast vectors + mu_per_layer
│   ├── emotion_pca_vectors.pt         # per-layer top-k PCA basis + scores
│   ├── emotion_va_ridge_betas.pt      # ridge-regression coefficients (PC space) for V/A
│   ├── emotion_va_directions.pt       # raw reconstructed 2048-dim V/A directions
│   └── emotion_va_directions_orth...  # Gram-Schmidt orthogonalized, unit-norm V/A axes
├── scripts/
│   ├── pull_go_emotions.py            # write the JSONL dataset
│   ├── count_emotion_labels.py
│   ├── query_coordinates.py           # query Qwen for VA coordinates
│   ├── contrast_emotion_vectors.py    # build emotion_contrast_vectors.pt
│   ├── mean_center_emotion_vect...py  # build emotion_centered_vectors.pt
│   ├── sanity_check_mean_centring...  # verify neutral==0, centering correctness
│   ├── PCA_...py                      # build emotion_pca_vectors.pt (per-layer SVD)
│   ├── Ridge_regression.py            # fit ridge V/A models in PC space
│   ├── reconstruction_vectors.py      # reconstruct raw 2048-dim V/A directions
│   ├── Gram_schmit.py                 # orthogonalize Arousal against Valence, normalize
│   └── plotting.py                    # circumplex validation scatter plot
├── notebooks/
│   └── gpu_emotion_extraction_and_valence.ipynb
├── outputs/                     # generated coordinate JSON + validation plots (not committed)
└── steering/                    # reserved for activation-steering experiments
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

Local CPU is enough for dataset scripts that only read or write JSONL, and for all of the representation-analysis pipeline below (PCA, ridge regression, Gram-Schmidt). **Emotion extraction with Hugging Face `datasets` plus Qwen valence ratings need a GPU.** Those steps live in the Colab notebook so they do not have to run on a laptop CPU.

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

## 3. Representation-analysis pipeline

Once `emotion_avg_vectors.pt` (per-emotion average hidden states from Qwen, 29 layers) and `qwen-native-coordinates.json` (27-emotion valence/arousal targets) exist, the following scripts extract and validate linear valence/arousal directions in the model's residual stream. All steps run on CPU and operate independently per layer (0–28).

**Step 1 — Contrast vectors** (`contrast_emotion_vectors.py`)
Builds `data/emotion_contrast_vectors.pt`: `v_e = mean_e - mean_neutral` per layer, per emotion (28 keys including an all-zero `neutral`, `[2048]` each).

**Step 2 — Mean-centering** (`mean_center_emotion_vect...py`)
Builds `data/emotion_centered_vectors.pt`: subtracts the per-layer centroid across the 27 GoEmotions labels, giving `V_centered_per_layer` (`[27, 2048]`), `centered_per_layer` (dict form), and `mu_per_layer` (the removed centroid, so `V_raw = V_centered + mu`). `sanity_check_mean_centring...py` confirms `neutral` is present-but-zero in the source and correctly excluded from the centering stack.

**Step 3 — PCA basis extraction** (`PCA_...py`)
Builds `data/emotion_pca_vectors.pt`: per-layer SVD on the centered `[27, 2048]` matrix, keeping the top `k=10` components. Outputs `components_per_layer` (`[10, 2048]`, the PC directions), `scores_per_layer` (`[27, 10]`, PC coordinates), and `explained_variance_ratio_per_layer`. PC1 was confirmed to align with valence (separating negative emotions like fear/disgust from positive ones like joy/gratitude), consistent with hedonic tone being the dominant axis of variance.

**Step 4 — Ridge regression in PC space** (`Ridge_regression.py`)
Builds `data/emotion_va_ridge_betas.pt`: fits `Ridge(alpha=1.0, fit_intercept=False)` per layer, regressing the `[27, 10]` PC scores onto mean-centered valence and arousal targets separately, giving `beta_V_per_layer` / `beta_A_per_layer` (`[10]` each). In-sample R² plateaus around 0.85–0.88 for valence and 0.60–0.67 for arousal from layer 3 onward, with arousal showing a mild late-layer decay.

**Step 5 — Reconstruction to activation space** (`reconstruction_vectors.py`)
Builds `data/emotion_va_directions.pt`: projects the PC-space coefficients back to the native 2048-dim space via `w_raw = beta @ Uk`, giving raw (un-normalized) `w_V_raw_per_layer` / `w_A_raw_per_layer`.

**Step 6 — Gram-Schmidt orthogonalization** (`Gram_schmit.py`)
Builds `data/emotion_va_directions_orthogonal.pt`: normalizes `w_V_raw` to unit length, then projects `w_A_raw` onto the orthogonal complement of `w_V` and normalizes the result, so the final `w_V_per_layer` / `w_A_per_layer` are unit-norm and mutually orthogonal at every layer (`cos_sim ≈ 0`, confirmed to ~1e-7 float noise). Before orthogonalization, valence and arousal directions were found to be substantially correlated in raw activation space (`cos_sim` ranging ~0.78–0.98 across layers), consistent with real-world co-occurrence of high-arousal negative states.

**Step 7 — Circumplex validation** (`plotting.py`)
Projects the 27 mean-centered emotion vectors onto the final orthonormal `(w_V, w_A)` plane per layer and produces a 2D scatter plot (`outputs/circumplex_validation_layer{N}.png`), checking for Russell's circumplex structure — high-arousal negative emotions (anger, fear) opposite low-arousal positive emotions (relief, calm), and positive valence (joy) opposite negative valence (sadness/grief).

## Dataset notes

- Source: `google-research-datasets/go_emotions`, config `simplified`
- Labels: 27 GoEmotions emotions plus `neutral` (see `config.py`)
- Each JSONL line: `{"id": "...", "text": "...", "label": "..."}`
- First download of GoEmotions and Qwen requires internet access (and enough disk for the Hugging Face cache)
- Model used throughout: `Qwen/Qwen3-1.7B` (hidden size 2048, 29 layers including embeddings)

## Status

Implemented:

- single-label GoEmotions extraction
- per-label counts
- Qwen valence–arousal querying (GPU / Colab notebook)
- contrast vectors, mean-centering, and sanity checks
- per-layer PCA basis extraction (top-10 components)
- ridge regression from PC scores to valence/arousal targets
- reconstruction of raw valence/arousal directions in 2048-dim activation space
- Gram-Schmidt orthogonalization of the valence/arousal axes
- circumplex (geometric) validation plot

In progress / planned:

- lexical validation checks (Phase 4, part 2)
- cross-layer comparison of circumplex quality
- activation steering along valence and arousal (`steering/`)

## References

- Demszky, D., et al. (2020). *GoEmotions: A Dataset of Fine-Grained Emotions.*
- Sun et al. (2026). Templates in Appendix A.1 for rating emotion labels on valence and arousal.