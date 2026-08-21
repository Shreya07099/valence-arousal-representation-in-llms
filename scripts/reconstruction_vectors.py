"""Reconstruct valence/arousal steering directions in the model's native
2048-dim residual stream from the k-dimensional ridge coefficients.

For each layer, given:
    beta_V   [k]        -- valence ridge coefficients (PC-space)
    beta_A   [k]        -- arousal ridge coefficients (PC-space)
    Uk       [k, 2048]  -- top-k PCA directions (components_per_layer[layer])

Reconstruct the raw (un-normalized) direction in activation space:

    w_V_raw = beta_V @ Uk   # [2048]
    w_A_raw = beta_A @ Uk   # [2048]

Then normalize to unit length for steering:

    w_V_hat = w_V_raw / ||w_V_raw||
    w_A_hat = w_A_raw / ||w_A_raw||

Applied independently at every layer, same as centering/PCA/ridge.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1] if (Path(__file__).resolve().parent.name == "scripts") else Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_INPUT = ROOT / "data" / "emotion_va_ridge_betas.pt"
DEFAULT_OUTPUT = ROOT / "data" / "emotion_va_directions.pt"
DEFAULT_CHECK_LAYER = 14


def reconstruct_layer(beta_V: torch.Tensor, beta_A: torch.Tensor, Uk: torch.Tensor):
    """beta: [k], Uk: [k, 2048] -> raw and unit-normalized [2048] directions."""
    w_V_raw = beta_V @ Uk   # [2048]
    w_A_raw = beta_A @ Uk   # [2048]

    w_V_hat = w_V_raw / w_V_raw.norm()
    w_A_hat = w_A_raw / w_A_raw.norm()

    return w_V_raw, w_A_raw, w_V_hat, w_A_hat


def reconstruct_all_layers(
    beta_V_per_layer: list[torch.Tensor],
    beta_A_per_layer: list[torch.Tensor],
    components_per_layer: list[torch.Tensor],
):
    w_V_raw_per_layer = []
    w_A_raw_per_layer = []
    w_V_hat_per_layer = []
    w_A_hat_per_layer = []

    for beta_V, beta_A, Uk in zip(beta_V_per_layer, beta_A_per_layer, components_per_layer):
        w_V_raw, w_A_raw, w_V_hat, w_A_hat = reconstruct_layer(beta_V, beta_A, Uk)
        w_V_raw_per_layer.append(w_V_raw)
        w_A_raw_per_layer.append(w_A_raw)
        w_V_hat_per_layer.append(w_V_hat)
        w_A_hat_per_layer.append(w_A_hat)

    return w_V_raw_per_layer, w_A_raw_per_layer, w_V_hat_per_layer, w_A_hat_per_layer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reconstruct valence/arousal directions in 2048-dim activation space from ridge betas."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT,
                         help=f"Path to emotion_va_ridge_betas.pt (default: {DEFAULT_INPUT}).")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
                         help=f"Output .pt path (default: {DEFAULT_OUTPUT}).")
    parser.add_argument("--check-layer", type=int, default=DEFAULT_CHECK_LAYER,
                         help=f"Layer index to print a sanity check for (default: {DEFAULT_CHECK_LAYER}).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = torch.load(args.input, map_location="cpu", weights_only=False)

    required = ("beta_V_per_layer", "beta_A_per_layer", "components_per_layer")
    missing = [key for key in required if key not in payload]
    if missing:
        raise ValueError(
            f"{args.input} is missing keys: {missing}. "
            "Did you run fit_ridge_va.py (with components_per_layer carried forward)?"
        )

    beta_V_per_layer = payload["beta_V_per_layer"]
    beta_A_per_layer = payload["beta_A_per_layer"]
    components_per_layer = payload["components_per_layer"]
    n_layers = len(beta_V_per_layer)

    if not (0 <= args.check_layer < n_layers):
        raise ValueError(f"--check-layer {args.check_layer} out of range for {n_layers} layers.")

    w_V_raw_per_layer, w_A_raw_per_layer, w_V_hat_per_layer, w_A_hat_per_layer = reconstruct_all_layers(
        beta_V_per_layer, beta_A_per_layer, components_per_layer
    )

    layer = args.check_layer
    print(f"Loaded {args.input}")
    print(f"w_V_raw[{layer}].shape = {tuple(w_V_raw_per_layer[layer].shape)}")
    print(f"w_V_hat[{layer}] norm  = {w_V_hat_per_layer[layer].norm().item():.6f}  (should be 1.0)")
    print(f"w_A_hat[{layer}] norm  = {w_A_hat_per_layer[layer].norm().item():.6f}  (should be 1.0)")
    cos_sim = torch.dot(w_V_hat_per_layer[layer], w_A_hat_per_layer[layer]).item()
    print(f"cos_sim(w_V_hat, w_A_hat) at layer {layer} = {cos_sim:.4f}  (near 0 = well-separated axes)")

    print()
    print("Raw direction norms across layers (valence / arousal) -- shows scale before normalization:")
    for i in range(n_layers):
        print(f"  layer {i:2d}: |w_V_raw|={w_V_raw_per_layer[i].norm().item():.4f}  |w_A_raw|={w_A_raw_per_layer[i].norm().item():.4f}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "w_V_raw_per_layer": w_V_raw_per_layer,   # list[29] of [2048], un-normalized
            "w_A_raw_per_layer": w_A_raw_per_layer,   # list[29] of [2048], un-normalized
            "w_V_hat_per_layer": w_V_hat_per_layer,   # list[29] of [2048], unit norm -- use for steering
            "w_A_hat_per_layer": w_A_hat_per_layer,   # list[29] of [2048], unit norm -- use for steering
            "k": payload["k"],
            "alpha": payload.get("alpha"),
            "emotions": payload["emotions"],
            "r2_V_per_layer": payload.get("r2_V_per_layer"),
            "r2_A_per_layer": payload.get("r2_A_per_layer"),
            "mu_per_layer": payload["mu_per_layer"],
            "model": payload.get("model"),
            "source": str(args.input),
            "formula": (
                "w_raw = beta @ Uk  (Uk = components_per_layer[layer], [k,2048]); "
                "w_hat = w_raw / ||w_raw||_2"
            ),
        },
        args.output,
    )
    print(f"\nSaved directions to {args.output}")
    print(
        "Access example: payload['w_V_hat_per_layer'][layer]  # [2048] unit valence direction  "
        "or payload['w_A_hat_per_layer'][layer]  # [2048] unit arousal direction"
    )


if __name__ == "__main__":
    main()