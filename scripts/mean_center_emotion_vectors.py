"""Mean-center contrastive emotion vectors across the 27 non-neutral labels.

For a layer, stack v_e into V in a fixed emotion order, then

    mu = mean_e(V)
    V_centered = V - mu

Neutral is excluded. Applied independently at every layer.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import GO_EMOTIONS_LABELS, NEUTRAL_LABEL

DEFAULT_INPUT = ROOT / "data" / "emotion_contrast_vectors.pt"
DEFAULT_OUTPUT = ROOT / "data" / "emotion_centered_vectors.pt"
DEFAULT_CHECK_LAYER = 14


def emotion_order(layer_dict: dict[str, torch.Tensor]) -> list[str]:
    """27 GoEmotions labels, skipping any missing from this layer."""
    missing = [e for e in GO_EMOTIONS_LABELS if e not in layer_dict]
    if missing:
        raise KeyError(f"Layer is missing emotions: {missing}")
    return list(GO_EMOTIONS_LABELS)


def stack_layer(
    layer_dict: dict[str, torch.Tensor], emotions: list[str]
) -> torch.Tensor:
    """Stack emotion vectors into [n_emotions, hidden]."""
    return torch.stack([layer_dict[e] for e in emotions], dim=0)


def mean_center(V: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (V - mu, mu) with mu the mean over emotions (dim=0)."""
    mu = V.mean(dim=0)
    return V - mu, mu


def mean_center_all_layers(
    contrast_per_layer: list[dict[str, torch.Tensor]],
) -> tuple[list[str], list[torch.Tensor], list[torch.Tensor], list[dict[str, torch.Tensor]]]:
    emotions = emotion_order(contrast_per_layer[0])
    V_centered_per_layer: list[torch.Tensor] = []
    mu_per_layer: list[torch.Tensor] = []
    centered_per_layer: list[dict[str, torch.Tensor]] = []

    for layer_dict in contrast_per_layer:
        V = stack_layer(layer_dict, emotions)
        V_centered, mu = mean_center(V)
        V_centered_per_layer.append(V_centered)
        mu_per_layer.append(mu)
        centered_per_layer.append(
            {emotion: V_centered[i] for i, emotion in enumerate(emotions)}
        )

    return emotions, V_centered_per_layer, mu_per_layer, centered_per_layer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Mean-center contrastive emotion vectors across 27 labels."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"Path to emotion_contrast_vectors.pt (default: {DEFAULT_INPUT}).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output .pt path (default: {DEFAULT_OUTPUT}).",
    )
    parser.add_argument(
        "--check-layer",
        type=int,
        default=DEFAULT_CHECK_LAYER,
        help=f"Layer index to print a sanity check for (default: {DEFAULT_CHECK_LAYER}).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = torch.load(args.input, map_location="cpu", weights_only=False)
    if "contrast_per_layer" not in payload:
        raise ValueError(f"{args.input} must contain key 'contrast_per_layer'.")

    contrast_per_layer = payload["contrast_per_layer"]
    n_layers = len(contrast_per_layer)
    if not (0 <= args.check_layer < n_layers):
        raise ValueError(
            f"--check-layer {args.check_layer} is out of range for {n_layers} layers."
        )

    emotions, V_centered_per_layer, mu_per_layer, centered_per_layer = (
        mean_center_all_layers(contrast_per_layer)
    )

    layer = args.check_layer
    V = stack_layer(contrast_per_layer[layer], emotions)
    V_centered = V_centered_per_layer[layer]
    mu = mu_per_layer[layer]
    residual = V_centered.mean(dim=0).abs().max().item()

    print(f"Loaded {args.input}")
    print(f"len(emotions) = {len(emotions)}")  # expect 27
    print(f"V.shape = {tuple(V.shape)}")  # expect [27, 2048]
    print(f"mu.shape = {tuple(mu.shape)}")  # expect [2048]
    print(f"V_centered.shape = {tuple(V_centered.shape)}")  # expect [27, 2048]
    print(f"layer {layer} max |mean| after centering = {residual:.3e}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "centered_per_layer": centered_per_layer,
            "V_centered_per_layer": V_centered_per_layer,
            "mu_per_layer": mu_per_layer,
            "emotions": emotions,
            "neutral_excluded": NEUTRAL_LABEL,
            "model": payload.get("model"),
            "source": str(args.input),
            "formula": "V_centered = V - mean_e(V)",
        },
        args.output,
    )
    print(f"Saved mean-centered vectors to {args.output}")
    print(
        "Access example: payload['centered_per_layer'][layer][emotion]  "
        "or payload['V_centered_per_layer'][layer]  # [27, hidden]"
    )


if __name__ == "__main__":
    main()
