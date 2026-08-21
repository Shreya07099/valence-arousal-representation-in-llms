"""Build contrastive emotion vectors by subtracting the neutral mean (eq. 1).

Input ``emotion_avg_vectors.pt`` already stores, for each layer ell and emotion e,

    avg_e^(ell) = (1 / |D_e|) * sum_{x in D_e} h^(ell)(x)

so the contrastive vector is

    v_e^(ell) = avg_e^(ell) - avg_neutral^(ell)

The saved file keeps the same layout: a list over layers, each a dict of
emotion name -> vector. The ``neutral`` entry is the zero vector.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import NEUTRAL_LABEL

DEFAULT_INPUT = ROOT / "data" / "emotion_avg_vectors.pt"
DEFAULT_OUTPUT = ROOT / "data" / "emotion_contrast_vectors.pt"


def contrast_with_neutral(
    avg_per_layer: list[dict[str, torch.Tensor]],
    *,
    neutral_label: str = NEUTRAL_LABEL,
) -> list[dict[str, torch.Tensor]]:
    """Subtract the per-layer neutral mean from every emotion mean."""
    contrast_per_layer: list[dict[str, torch.Tensor]] = []

    for layer_idx, emotion_avgs in enumerate(avg_per_layer):
        if neutral_label not in emotion_avgs:
            raise KeyError(
                f"Layer {layer_idx} is missing {neutral_label!r}; "
                f"keys={sorted(emotion_avgs)}"
            )
        neutral_mean = emotion_avgs[neutral_label]
        contrast_per_layer.append(
            {emotion: (avg - neutral_mean) for emotion, avg in emotion_avgs.items()}
        )

    return contrast_per_layer


def load_avg_vectors(path: Path) -> dict:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or "avg_per_layer" not in payload:
        raise ValueError(
            f"{path} must be a dict with key 'avg_per_layer' "
            "(list of per-layer emotion-name -> tensor maps)."
        )
    return payload


def save_contrast_vectors(
    contrast_per_layer: list[dict[str, torch.Tensor]],
    output_path: Path,
    *,
    model: str | None,
    source: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "contrast_per_layer": contrast_per_layer,
            "model": model,
            "source": str(source),
            "neutral_label": NEUTRAL_LABEL,
            "formula": "v_e = mean_e - mean_neutral",
        },
        output_path,
    )


def _layer_norm_summary(
    contrast_per_layer: list[dict[str, torch.Tensor]], layer_idx: int
) -> str:
    vectors = contrast_per_layer[layer_idx]
    parts = []
    for emotion in ("anger", "joy", "sadness", NEUTRAL_LABEL):
        if emotion in vectors:
            parts.append(f"{emotion}={vectors[emotion].float().norm().item():.4f}")
    return ", ".join(parts)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Subtract the per-layer neutral mean from stored emotion averages "
            "to obtain contrastive emotion vectors."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"Path to emotion_avg_vectors.pt (default: {DEFAULT_INPUT}).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output .pt path (default: {DEFAULT_OUTPUT}).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = load_avg_vectors(args.input)
    avg_per_layer = payload["avg_per_layer"]

    n_layers = len(avg_per_layer)
    n_emotions = len(avg_per_layer[0]) if avg_per_layer else 0
    hidden = next(iter(avg_per_layer[0].values())).numel() if avg_per_layer else 0
    print(
        f"Loaded {args.input}: {n_layers} layers, {n_emotions} emotions, "
        f"hidden={hidden}"
    )

    contrast_per_layer = contrast_with_neutral(avg_per_layer)
    save_contrast_vectors(
        contrast_per_layer,
        args.output,
        model=payload.get("model"),
        source=args.input,
    )

    mid = n_layers // 2
    print(f"Layer 0 L2 norms: { _layer_norm_summary(contrast_per_layer, 0)}")
    if n_layers > 1:
        print(f"Layer {mid} L2 norms: {_layer_norm_summary(contrast_per_layer, mid)}")
        print(
            f"Layer {n_layers - 1} L2 norms: "
            f"{_layer_norm_summary(contrast_per_layer, n_layers - 1)}"
        )
    print(f"Saved contrastive vectors to {args.output}")
    print(
        "Output dict keys: contrast_per_layer, model, source, "
        "neutral_label, formula"
    )
    print(
        "Access example: payload['contrast_per_layer'][layer][emotion]  "
        "-> tensor of shape [hidden]"
    )


if __name__ == "__main__":
    main()
