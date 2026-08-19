

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import ALL_LABELS

DEFAULT_MODEL = "Qwen/Qwen3-1.7B"
DEFAULT_OUTPUT = "outputs/qwen_native_coordinates.json"


EMOTIONS = ALL_LABELS


TEMPLATES = {
    "template_1": (
        'Rate the emotion label "{label}" on two continuous scales.\n'
        'Return ONLY a JSON object with numeric fields: {{"valence": <number>, "arousal": <number>}}\n\n'
        "Scale definitions (BOTH inclusive):\n"
        "- valence in [-1.00, +1.00]: -1.00 very unpleasant, +1.00 very pleasant\n"
        "- arousal in [-1.00, +1.00]: -1.00 very calm/deactivated, +1.00 very activated/intense\n\n"
        "Constraints:\n"
        "- Use decimals with at most 2 digits after the decimal.\n"
        "- Values must be within the ranges exactly (inclusive)."
    ),
    "template_2": (
        "You are scoring affective properties of emotion words on [-1, +1] scales.\n"
        'Emotion: "{label}"\n'
        "Valence (pleasantness): -1.00 = extremely unpleasant, 0.00 = neutral/mixed, +1.00 = extremely pleasant\n"
        "Arousal (activation/intensity): -1.00 = very calm/deactivated, 0.00 = neutral, +1.00 = very activated/intense\n\n"
        'Return ONLY JSON: {{"valence": x, "arousal": y}}\n'
        "x and y must be in [-1.00, +1.00] inclusive, with at most 2 decimals."
    ),
    "template_3": (
        'Give your best guess for the affective coordinates of the emotion label "{label}".\n'
        "Hard constraints (inclusive):\n"
        "- valence must be between -1.00 and +1.00\n"
        "- arousal must be between -1.00 and +1.00\n"
        "- use at most 2 decimals\n\n"
        'Return ONLY JSON: {{"valence": <number>, "arousal": <number>}}'
    ),
}

JSON_PATTERNS = (
    r'\{[^{}]*"valence"\s*:\s*[-+]?\d*\.?\d+[^{}]*"arousal"\s*:\s*[-+]?\d*\.?\d+[^{}]*\}',
    r'\{[^{}]*"arousal"\s*:\s*[-+]?\d*\.?\d+[^{}]*"valence"\s*:\s*[-+]?\d*\.?\d+[^{}]*\}',
)


def strip_thinking(text: str) -> str:
    """Remove Qwen3 chain-of-thought blocks if present."""
    think_end = "<" + "/think>"
    if think_end.lower() in text.lower():
        text = re.split(re.escape(think_end), text, flags=re.IGNORECASE)[-1]
    return text.strip()


def extract_json(text: str) -> tuple[float, float] | None:
    """Parse valence/arousal JSON from model output."""
    text = strip_thinking(text)

    candidates = []
    for pattern in JSON_PATTERNS:
        candidates.extend(re.findall(pattern, text, flags=re.DOTALL | re.IGNORECASE))

    if not candidates:
        match = re.search(r"\{.*?\}", text, re.DOTALL)
        if match:
            candidates.append(match.group(0))

    for candidate in candidates:
        try:
            data = json.loads(candidate)
            data = {str(k).lower(): float(v) for k, v in data.items()}
            if "valence" in data and "arousal" in data:
                return data["valence"], data["arousal"]
        except (TypeError, ValueError, json.JSONDecodeError):
            continue

    return None


def get_model_device(model: torch.nn.Module) -> torch.device:
    return next(model.parameters()).device


def build_chat_text(tokenizer, prompt: str) -> str:
    messages = [
        {
            "role": "system",
            "content": "You are a precise assistant that outputs JSON coordinates exactly as instructed.",
        },
        {"role": "user", "content": prompt},
    ]
    kwargs = {
        "tokenize": False,
        "add_generation_prompt": True,
    }
    # Qwen3 supports a thinking mode, disavle that
    if "enable_thinking" in tokenizer.chat_template or "qwen3" in tokenizer.name_or_path.lower():
        kwargs["enable_thinking"] = False

    try:
        return tokenizer.apply_chat_template(messages, **kwargs)
    except TypeError:
        kwargs.pop("enable_thinking", None)
        return tokenizer.apply_chat_template(messages, **kwargs)


def query_qwen_for_coordinates(
    model_name: str = DEFAULT_MODEL,
    output_path: Path = Path(DEFAULT_OUTPUT),
) -> dict[str, dict[str, float]]:
    print(f"Loading model and tokenizer: {model_name}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    device = get_model_device(model)

    model_native_coordinates: dict[str, dict[str, float]] = {}

    for emotion in EMOTIONS:
        print(f"\nQuerying coordinates for: {emotion}")
        template_scores: list[tuple[float, float]] = []

        for temp_id, temp_str in TEMPLATES.items():
            prompt = temp_str.format(label=emotion)
            text = build_chat_text(tokenizer, prompt)
            model_inputs = tokenizer([text], return_tensors="pt").to(device)

            with torch.no_grad():
                generated_ids = model.generate(
                    **model_inputs,
                    max_new_tokens=128,
                    do_sample=False,
                )

            generated_ids = [
                output_ids[len(input_ids) :]
                for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]
            response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

            coords = extract_json(response)
            if coords:
                template_scores.append(coords)
            else:
                print(
                    f"  Warning: failed to parse {temp_id}. "
                    f"Raw output: {response.strip()!r}"
                )

        if template_scores:
            avg_v = sum(v for v, _ in template_scores) / len(template_scores)
            avg_a = sum(a for _, a in template_scores) / len(template_scores)
            avg_v = max(-1.0, min(1.0, round(avg_v, 4)))
            avg_a = max(-1.0, min(1.0, round(avg_a, 4)))
            model_native_coordinates[emotion] = {"valence": avg_v, "arousal": avg_a}
            print(
                f"  Result -> valence: {avg_v:.2f}, arousal: {avg_a:.2f} "
                f"(averaged over {len(template_scores)} templates)"
            )
        else:
            print(f"  Error: no valid coordinates for {emotion}; using (0.0, 0.0)")
            model_native_coordinates[emotion] = {"valence": 0.0, "arousal": 0.0}

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(model_native_coordinates, f, indent=4)

    print(f"\nSaved {len(model_native_coordinates)} labels to {output_path}")
    return model_native_coordinates


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Query Qwen for model-native valence-arousal coordinates."
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Hugging Face model id (default: {DEFAULT_MODEL}).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(DEFAULT_OUTPUT),
        help=f"Output JSON path (default: {DEFAULT_OUTPUT}).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    query_qwen_for_coordinates(model_name=args.model, output_path=args.output)


if __name__ == "__main__":
    main()
