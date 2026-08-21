"""Gram-Schmidt orthogonalization of the Valence/Arousal directions.

Real-world emotions correlate (e.g. panic = high arousal + very negative
valence), so the raw reconstructed axes w_V_raw / w_A_raw are not
perpendicular -- steering along one leaks into the other. This script
makes Valence the primary axis and orthogonalizes Arousal against it,
so the two axes become independently controllable.

For each layer:
    1. w_V = w_V_raw / ||w_V_raw||                          -- normalize valence
    2. w_A_proj = w_A_raw - (w_A_raw . w_V) * w_V            -- strip valence leakage from arousal
    3. w_A = w_A_proj / ||w_A_proj||                         -- normalize arousal

After this, dot(w_V, w_A) == 0 by construction at every layer.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1] if (Path(__file__).resolve().parent.name == "scripts") else Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_INPUT = ROOT / "data" / "emotion_va_directions.pt"
DEFAULT_OUTPUT = ROOT / "data" / "emotion_va_directions_orthogonal.pt"
DEFAULT_CHECK_LAYER = 14


def gram_schmidt_layer(w_V_raw: torch.Tensor, w_A_raw: torch.Tensor):
    """Orthogonalize arousal against valence for one layer.

    Returns (w_V, w_A, leakage), all [2048] except leakage (scalar):
        w_V     -- unit valence axis (primary, unchanged direction)
        w_A     -- unit arousal axis, orthogonal to w_V
        leakage -- w_A_raw . w_V before projection (how much valence was in raw arousal)
    """
    w_V = w_V_raw / w_V_raw.norm()

    leakage = torch.dot(w_A_raw, w_V)
    w_A_proj = w_A_raw - leakage * w_V
    w_A = w_A_proj / w_A_proj.norm()

    return w_V, w_A, leakage.item()


def gram_schmidt_all_layers(
    w_V_raw_per_layer: list[torch.Tensor],
    w_A_raw_per_layer: list[torch.Tensor],
):
    w_V_per_layer = []
    w_A_per_layer = []
    leakage_per_layer = []

    for w_V_raw, w_A_raw in zip(w_V_raw_per_layer, w_A_raw_per_layer):
        w_V, w_A, leakage = gram_schmidt_layer(w_V_raw, w_A_raw)
        w_V_per_layer.append(w_V)
        w_A_per_layer.append(w_A)
        leakage_per_layer.append(leakage)

    return w_V_per_layer, w_A_per_layer, leakage_per_layer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Gram-Schmidt orthogonalize Arousal against Valence, per layer."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT,
                         help=f"Path to emotion_va_directions.pt (default: {DEFAULT_INPUT}).")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
                         help=f"Output .pt path (default: {DEFAULT_OUTPUT}).")
    parser.add_argument("--check-layer", type=int, default=DEFAULT_CHECK_LAYER,
                         help=f"Layer index to print a sanity check for (default: {DEFAULT_CHECK_LAYER}).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = torch.load(args.input, map_location="cpu", weights_only=False)

    required = ("w_V_raw_per_layer", "w_A_raw_per_layer")
    missing = [key for key in required if key not in payload]
    if missing:
        raise ValueError(
            f"{args.input} is missing keys: {missing}. "
            "Did you run reconstruct_va_directions.py?"
        )

    w_V_raw_per_layer = payload["w_V_raw_per_layer"]
    w_A_raw_per_layer = payload["w_A_raw_per_layer"]
    n_layers = len(w_V_raw_per_layer)

    if not (0 <= args.check_layer < n_layers):
        raise ValueError(f"--check-layer {args.check_layer} out of range for {n_layers} layers.")

    # cosine similarity BEFORE orthogonalization, for comparison
    cos_sim_before_per_layer = [
        torch.dot(
            w_V_raw_per_layer[i] / w_V_raw_per_layer[i].norm(),
            w_A_raw_per_layer[i] / w_A_raw_per_layer[i].norm(),
        ).item()
        for i in range(n_layers)
    ]

    w_V_per_layer, w_A_per_layer, leakage_per_layer = gram_schmidt_all_layers(
        w_V_raw_per_layer, w_A_raw_per_layer
    )

    # cosine similarity AFTER -- should be ~0 (up to float error) at every layer
    cos_sim_after_per_layer = [
        torch.dot(w_V_per_layer[i], w_A_per_layer[i]).item() for i in range(n_layers)
    ]

    layer = args.check_layer
    print(f"Loaded {args.input}")
    print(f"w_V[{layer}] norm = {w_V_per_layer[layer].norm().item():.6f}  (should be 1.0)")
    print(f"w_A[{layer}] norm = {w_A_per_layer[layer].norm().item():.6f}  (should be 1.0)")
    print(f"layer {layer}: cos_sim before = {cos_sim_before_per_layer[layer]:.4f}, "
          f"after = {cos_sim_after_per_layer[layer]:.2e}  (should be ~0)")

    print()
    print("cos_sim(w_V, w_A) before / after orthogonalization, all layers:")
    for i in range(n_layers):
        print(f"  layer {i:2d}: before={cos_sim_before_per_layer[i]:+.4f}  after={cos_sim_after_per_layer[i]:+.2e}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "w_V_per_layer": w_V_per_layer,                 # list[29] of [2048], unit norm, primary axis
            "w_A_per_layer": w_A_per_layer,                 # list[29] of [2048], unit norm, orthogonal to w_V
            "cos_sim_before_per_layer": cos_sim_before_per_layer,  # list[29] floats
            "cos_sim_after_per_layer": cos_sim_after_per_layer,    # list[29] floats, ~0
            "leakage_per_layer": leakage_per_layer,          # list[29] floats, w_A_raw . w_V before projection
            "k": payload.get("k"),
            "alpha": payload.get("alpha"),
            "emotions": payload.get("emotions"),
            "r2_V_per_layer": payload.get("r2_V_per_layer"),
            "r2_A_per_layer": payload.get("r2_A_per_layer"),
            "mu_per_layer": payload.get("mu_per_layer"),
            "model": payload.get("model"),
            "source": str(args.input),
            "formula": (
                "w_V = w_V_raw / ||w_V_raw||; "
                "w_A_proj = w_A_raw - (w_A_raw . w_V) * w_V; "
                "w_A = w_A_proj / ||w_A_proj||  (Valence is the primary, unrotated axis)"
            ),
        },
        args.output,
    )
    print(f"\nSaved orthogonalized directions to {args.output}")
    print(
        "Access example: payload['w_V_per_layer'][layer]  # [2048] unit valence axis  "
        "or payload['w_A_per_layer'][layer]  # [2048] unit arousal axis, orthogonal to valence"
    )


if __name__ == "__main__":
    main()