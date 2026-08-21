"""Fit per-layer Ridge regressions mapping PCA scores (Z) to
mean-centered Valence / Arousal targets.

For each layer, given:
    Z              [27, k]   -- PC scores (from emotion_pca_vectors.pt)
    yV_centered    [27]      -- mean-centered valence targets
    yA_centered    [27]      -- mean-centered arousal targets

Fit two separate Ridge models (alpha=1.0, fit_intercept=False, since
both Z and y are already mean-centered):

    Valence: Ridge().fit(Z, yV_centered)  -> beta_V in R^k
    Arousal: Ridge().fit(Z, yA_centered)  -> beta_A in R^k

beta_V / beta_A are the valence/arousal directions expressed in the
k-dimensional PC subspace. Fit independently per layer, same as
centering and PCA.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from sklearn.linear_model import Ridge

ROOT = Path(__file__).resolve().parents[1] if (Path(__file__).resolve().parent.name == "scripts") else Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_INPUT = ROOT / "data" / "emotion_pca_vectors.pt"
DEFAULT_TARGETS = ROOT / "data" / "qwen-native-coordinates.json"
DEFAULT_OUTPUT = ROOT / "data" / "emotion_va_ridge_betas.pt"
DEFAULT_ALPHA = 1.0
DEFAULT_CHECK_LAYER = 14


def load_targets(path: Path, emotions: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
    """Load {emotion: {valence, arousal}} JSON, align to `emotions` order,
    drop any extra keys (e.g. 'neutral'), and mean-center.

    Returns (yV_centered, yA_centered), each [len(emotions)].
    """
    with open(path) as f:
        raw = json.load(f)

    missing = [e for e in emotions if e not in raw]
    if missing:
        raise ValueError(f"{path} is missing targets for: {missing}")

    yV = torch.tensor([raw[e]["valence"] for e in emotions], dtype=torch.float32)
    yA = torch.tensor([raw[e]["arousal"] for e in emotions], dtype=torch.float32)

    yV_centered = yV - yV.mean()
    yA_centered = yA - yA.mean()
    return yV_centered, yA_centered


def fit_layer(Z: torch.Tensor, yV_centered: torch.Tensor, yA_centered: torch.Tensor, alpha: float):
    """Fit Ridge(alpha, fit_intercept=False) for V and A on one layer's [27, k] scores."""
    Z_np = Z.numpy()

    ridge_V = Ridge(alpha=alpha, fit_intercept=False)
    ridge_V.fit(Z_np, yV_centered.numpy())
    beta_V = torch.tensor(ridge_V.coef_, dtype=torch.float32)  # [k]

    ridge_A = Ridge(alpha=alpha, fit_intercept=False)
    ridge_A.fit(Z_np, yA_centered.numpy())
    beta_A = torch.tensor(ridge_A.coef_, dtype=torch.float32)  # [k]

    # R^2 on training data (in-sample fit quality; k=10 params, n=27 rows -- watch for overfitting)
    r2_V = ridge_V.score(Z_np, yV_centered.numpy())
    r2_A = ridge_A.score(Z_np, yA_centered.numpy())

    return beta_V, beta_A, r2_V, r2_A


def fit_all_layers(
    scores_per_layer: list[torch.Tensor],
    yV_centered: torch.Tensor,
    yA_centered: torch.Tensor,
    alpha: float,
):
    beta_V_per_layer = []
    beta_A_per_layer = []
    r2_V_per_layer = []
    r2_A_per_layer = []

    for Z in scores_per_layer:
        beta_V, beta_A, r2_V, r2_A = fit_layer(Z, yV_centered, yA_centered, alpha)
        beta_V_per_layer.append(beta_V)
        beta_A_per_layer.append(beta_A)
        r2_V_per_layer.append(r2_V)
        r2_A_per_layer.append(r2_A)

    return beta_V_per_layer, beta_A_per_layer, r2_V_per_layer, r2_A_per_layer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fit per-layer Ridge regressions from PCA scores to mean-centered V/A targets."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT,
                         help=f"Path to emotion_pca_vectors.pt (default: {DEFAULT_INPUT}).")
    parser.add_argument("--targets", type=Path, default=DEFAULT_TARGETS,
                         help=f"Path to {{emotion: {{valence, arousal}}}} JSON (default: {DEFAULT_TARGETS}).")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
                         help=f"Output .pt path (default: {DEFAULT_OUTPUT}).")
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA,
                         help=f"Ridge L2 penalty (default: {DEFAULT_ALPHA}).")
    parser.add_argument("--check-layer", type=int, default=DEFAULT_CHECK_LAYER,
                         help=f"Layer index to print a sanity check for (default: {DEFAULT_CHECK_LAYER}).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = torch.load(args.input, map_location="cpu", weights_only=False)

    required = ("scores_per_layer", "emotions", "k")
    missing = [key for key in required if key not in payload]
    if missing:
        raise ValueError(f"{args.input} is missing keys: {missing}")

    emotions = payload["emotions"]
    scores_per_layer = payload["scores_per_layer"]
    n_layers = len(scores_per_layer)

    if not (0 <= args.check_layer < n_layers):
        raise ValueError(f"--check-layer {args.check_layer} out of range for {n_layers} layers.")

    yV_centered, yA_centered = load_targets(args.targets, emotions)

    beta_V_per_layer, beta_A_per_layer, r2_V_per_layer, r2_A_per_layer = fit_all_layers(
        scores_per_layer, yV_centered, yA_centered, args.alpha
    )

    layer = args.check_layer
    print(f"Loaded {args.input}")
    print(f"Loaded targets from {args.targets} ({len(emotions)} emotions, mean-centered)")
    print(f"alpha = {args.alpha}, k = {payload['k']}")
    print(f"beta_V[{layer}].shape = {tuple(beta_V_per_layer[layer].shape)}")
    print(f"beta_A[{layer}].shape = {tuple(beta_A_per_layer[layer].shape)}")
    print(f"layer {layer}: R^2(valence) = {r2_V_per_layer[layer]:.3f}, R^2(arousal) = {r2_A_per_layer[layer]:.3f}")
    print()
    print("R^2 across layers (valence / arousal):")
    for i in range(n_layers):
        print(f"  layer {i:2d}: V={r2_V_per_layer[i]:.3f}  A={r2_A_per_layer[i]:.3f}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "alpha": args.alpha,
            "k": payload["k"],
            "emotions": emotions,
            "beta_V_per_layer": beta_V_per_layer,   # list[29] of [k]
            "beta_A_per_layer": beta_A_per_layer,   # list[29] of [k]
            "r2_V_per_layer": r2_V_per_layer,       # list[29] floats, in-sample
            "r2_A_per_layer": r2_A_per_layer,       # list[29] floats, in-sample
            "yV_centered": yV_centered,             # [27]
            "yA_centered": yA_centered,              # [27]
            "components_per_layer": payload["components_per_layer"],  # carried forward
            "mu_per_layer": payload["mu_per_layer"],                  # carried forward
            "model": payload.get("model"),
            "source": str(args.input),
            "targets_source": str(args.targets),
            "formula": (
                "beta_V, beta_A = Ridge(alpha, fit_intercept=False).fit(Z, y_centered).coef_ "
                "per layer, where Z = scores_per_layer[layer] [27,k], "
                "y_centered = target - target.mean() over the 27 emotions."
            ),
        },
        args.output,
    )
    print(f"\nSaved ridge betas to {args.output}")
    print(
        "Access example: payload['beta_V_per_layer'][layer]  # [k]  "
        "or payload['beta_A_per_layer'][layer]  # [k]"
    )


if __name__ == "__main__":
    main()