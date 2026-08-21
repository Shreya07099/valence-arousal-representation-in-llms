"""
Sanity check for emotion_centered_vectors.pt, produced by the
mean-centering script (V_centered = V - mean_e(V), per layer,
over the 27 non-neutral GoEmotions labels).

Checks:
  1. Structural sanity: 27 emotions, correct shapes, emotions list
     matches GO_EMOTIONS_LABELS order, neutral excluded.
  2. Matrix form (V_centered_per_layer) vs dict form (centered_per_layer)
     contain identical numbers.
  3. Zero-mean property: mean over the 27 rows is ~0 at every layer
     (this is the actual definition of mean-centering).
  4. Reconstruction against the source file: loads the original
     contrast_per_layer from payload["source"] and checks
     V - mu == V_centered, and mu == mean(V), for every layer.

Usage:
    python check_centering.py [path/to/emotion_centered_vectors.pt]

If no path is given, defaults to data/emotion_centered_vectors.pt
relative to the project root (same default the mean-centering script uses).
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import GO_EMOTIONS_LABELS, NEUTRAL_LABEL

DEFAULT_CENTERED = ROOT / "data" / "emotion_centered_vectors.pt"
ATOL = 1e-4


def check_structure(payload) -> bool:
    print("\n[1] Structural sanity")
    ok = True

    emotions = payload["emotions"]
    if emotions != list(GO_EMOTIONS_LABELS):
        print(f"  FAIL: emotions list doesn't match GO_EMOTIONS_LABELS order")
        ok = False
    else:
        print(f"  emotions: {len(emotions)} labels, order matches GO_EMOTIONS_LABELS  [ok]")

    if NEUTRAL_LABEL in emotions:
        print(f"  FAIL: neutral label '{NEUTRAL_LABEL}' present in emotions list")
        ok = False
    else:
        print(f"  neutral ('{NEUTRAL_LABEL}') correctly excluded  [ok]")

    n_layers_mat = len(payload["V_centered_per_layer"])
    n_layers_dict = len(payload["centered_per_layer"])
    n_layers_mu = len(payload["mu_per_layer"])
    if n_layers_mat != n_layers_dict or n_layers_mat != n_layers_mu:
        print(f"  FAIL: layer count mismatch (matrix={n_layers_mat}, "
              f"dict={n_layers_dict}, mu={n_layers_mu})")
        ok = False
    else:
        print(f"  {n_layers_mat} layers, consistent across all three fields  [ok]")

    for layer in range(n_layers_mat):
        mat = payload["V_centered_per_layer"][layer]
        mu = payload["mu_per_layer"][layer]
        if mat.shape[0] != len(emotions):
            print(f"  Layer {layer}: FAIL shape {tuple(mat.shape)}, "
                  f"expected {len(emotions)} rows")
            ok = False
        if mu.shape[0] != mat.shape[1]:
            print(f"  Layer {layer}: FAIL mu shape {tuple(mu.shape)} "
                  f"!= hidden dim {mat.shape[1]}")
            ok = False

    if ok:
        hidden = payload["V_centered_per_layer"][0].shape[1]
        print(f"  all layer shapes consistent: [{len(emotions)}, {hidden}]  [ok]")

    print("  -> PASS" if ok else "  -> FAIL")
    return ok


def check_matrix_dict_consistency(payload) -> bool:
    print("\n[2] Matrix vs dict consistency")
    emotions = payload["emotions"]
    n_layers = len(payload["V_centered_per_layer"])
    ok = True

    for layer in range(n_layers):
        mat = payload["V_centered_per_layer"][layer]
        dct = payload["centered_per_layer"][layer]
        for i, name in enumerate(emotions):
            if name not in dct:
                print(f"  Layer {layer}: '{name}' missing from dict form")
                ok = False
                continue
            if not torch.allclose(mat[i], dct[name], atol=ATOL):
                diff = (mat[i] - dct[name]).abs().max().item()
                print(f"  Layer {layer}, '{name}': max abs diff = {diff:.6f}")
                ok = False

    print("  -> PASS: matrix and dict forms match" if ok else "  -> FAIL")
    return ok


def check_zero_mean(payload) -> bool:
    print("\n[3] Zero-mean check (mean over 27 emotions, per layer)")
    n_layers = len(payload["V_centered_per_layer"])
    ok = True

    for layer in range(n_layers):
        mat = payload["V_centered_per_layer"][layer]
        max_abs = mat.mean(dim=0).abs().max().item()
        layer_ok = max_abs < ATOL
        ok &= layer_ok
        print(f"  Layer {layer:2d}: max|mean| = {max_abs:.3e}  "
              f"[{'ok' if layer_ok else 'FAIL'}]")

    print("  -> PASS: every layer centered to ~0" if ok else "  -> FAIL")
    return ok


def check_source_reconstruction(payload):
    print("\n[4] Reconstruction against source file")
    source = payload.get("source")
    if not source or not Path(source).exists():
        print(f"  Source file not found ({source!r}) — skipping.")
        print("  -> SKIPPED")
        return None

    raw_payload = torch.load(source, map_location="cpu", weights_only=False)
    if "contrast_per_layer" not in raw_payload:
        print(f"  {source} has no 'contrast_per_layer' key — skipping.")
        print("  -> SKIPPED")
        return None

    contrast_per_layer = raw_payload["contrast_per_layer"]
    emotions = payload["emotions"]
    n_layers = len(payload["V_centered_per_layer"])
    ok = True

    for layer in range(n_layers):
        layer_dict = contrast_per_layer[layer]
        V = torch.stack([layer_dict[e] for e in emotions], dim=0)
        mu = payload["mu_per_layer"][layer]
        V_centered = payload["V_centered_per_layer"][layer]

        recon_ok = torch.allclose(V - mu, V_centered, atol=ATOL)
        mu_ok = torch.allclose(V.mean(dim=0), mu, atol=ATOL)
        ok &= recon_ok & mu_ok

        if not recon_ok:
            diff = (V - mu - V_centered).abs().max().item()
            print(f"  Layer {layer}: V - mu != V_centered, max abs diff = {diff:.6f}")
        if not mu_ok:
            diff2 = (V.mean(dim=0) - mu).abs().max().item()
            print(f"  Layer {layer}: mu != mean(V), max abs diff = {diff2:.6f}")

    print("  -> PASS: mu and V_centered exactly reproduce V from source"
          if ok else "  -> FAIL")
    return ok


def main() -> None:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_CENTERED
    if not path.exists():
        print(f"File not found: {path}")
        sys.exit(1)

    payload = torch.load(path, map_location="cpu", weights_only=False)
    print(f"Loaded {path}")

    r1 = check_structure(payload)
    r2 = check_matrix_dict_consistency(payload)
    r3 = check_zero_mean(payload)
    r4 = check_source_reconstruction(payload)

    def fmt(r):
        return "SKIPPED" if r is None else ("PASS" if r else "FAIL")

    print("\n=== Summary ===")
    print(f"  Structural sanity          : {fmt(r1)}")
    print(f"  Matrix/dict consistency    : {fmt(r2)}")
    print(f"  Zero-mean centering        : {fmt(r3)}")
    print(f"  Source reconstruction      : {fmt(r4)}")

    if all(r in (True, None) for r in (r1, r2, r3, r4)):
        print("\nAll checks passed (or skipped where source unavailable).")
    else:
        print("\nSome checks FAILED — see details above.")
        sys.exit(1)


if __name__ == "__main__":
    main()