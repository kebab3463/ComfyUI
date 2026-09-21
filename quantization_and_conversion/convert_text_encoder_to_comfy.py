#!/usr/bin/env python3
"""
Convert HuggingFace / diffusers text encoder weights to ComfyUI-compatible safetensors.

ComfyUI expects a single .safetensors file under models/text_encoders/ with tensor keys
that match its loaders. Many modern models (Qwen2.5-VL, Llama, etc.) already use the
same key names as HuggingFace; the main work is merging sharded checkpoints.

Classic SD/SDXL diffusers text encoders need key renaming via comfy.diffusers_convert.

Examples:
  # Full Qwen2.5-VL (text + vision) for CLIPLoader / Qwen Image workflows
  python convert_text_encoder_to_comfy.py \\
    models/text_encoders/Qwen2.5VL/Qwen2.5-VL-7B-Instruct-heretic \\
    -o models/text_encoders/Qwen2.5VL/Qwen2.5-VL-7B-Instruct-heretic.safetensors \\
    --mode vl

  # Text-only subset for WanVideoWrapper QwenLoader
  python convert_text_encoder_to_comfy.py \\
    models/text_encoders/Qwen2.5VL/Qwen2.5-VL-7B-Instruct-heretic \\
    -o models/text_encoders/Qwen2.5VL/Qwen2.5-VL-7B-Instruct-heretic-text.safetensors \\
    --mode text_only

  # SDXL diffusers pipeline text_encoder folder
  python convert_text_encoder_to_comfy.py /path/to/sdxl-diffusers --mode auto
"""

from __future__ import annotations

import argparse
import json
import os
import sys

# Walk up to the ComfyUI root (dir containing comfy/) so this works from any subfolder.
COMFY_ROOT = os.path.dirname(os.path.abspath(__file__))
while COMFY_ROOT != os.path.dirname(COMFY_ROOT) and not os.path.isdir(os.path.join(COMFY_ROOT, "comfy")):
    COMFY_ROOT = os.path.dirname(COMFY_ROOT)
if COMFY_ROOT not in sys.path:
    sys.path.insert(0, COMFY_ROOT)

import comfy.utils
from comfy import diffusers_convert


TEXT_ENCODER_FILENAMES = [
    "model.fp16.safetensors",
    "model.safetensors",
    "pytorch_model.fp16.bin",
    "pytorch_model.bin",
]


def first_file(path: str, filenames: list[str]) -> str | None:
    for name in filenames:
        candidate = os.path.join(path, name)
        if os.path.exists(candidate):
            return candidate
    return None


def resolve_input_path(path: str) -> str:
    path = os.path.abspath(path)
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    if os.path.isfile(path):
        return path

    index_path = os.path.join(path, "model.safetensors.index.json")
    if os.path.isfile(index_path):
        return path

    for subdir in ("text_encoder", "text_encoder_2"):
        te_dir = os.path.join(path, subdir)
        te_file = first_file(te_dir, TEXT_ENCODER_FILENAMES)
        if te_file is not None:
            return te_file

    single = first_file(path, TEXT_ENCODER_FILENAMES)
    if single is not None:
        return single

    raise FileNotFoundError(
        f"Could not find text encoder weights in {path}. "
        "Expected a .safetensors file, model.safetensors.index.json, or text_encoder/ subfolder."
    )


def load_state_dict(path: str) -> dict:
    if os.path.isfile(path):
        return comfy.utils.load_torch_file(path)

    index_path = os.path.join(path, "model.safetensors.index.json")
    if os.path.isfile(index_path):
        with open(index_path, encoding="utf-8") as f:
            weight_map = json.load(f)["weight_map"]
        sd: dict = {}
        for shard in sorted(set(weight_map.values())):
            shard_path = os.path.join(path, shard)
            sd.update(comfy.utils.load_torch_file(shard_path))
        return sd

    return comfy.utils.load_torch_file(path)


def detect_format(state_dict: dict) -> str:
    keys = state_dict.keys()
    if any(k.startswith("text_model.encoder.layers.") for k in keys):
        return "diffusers_clip"
    if any(k.startswith("model.language_model.") for k in keys):
        return "qwen35_nested"
    if any(k.startswith("visual.") for k in keys):
        return "qwen_vl"
    if any(k.startswith("model.layers.") for k in keys):
        return "llama_style"
    return "unknown"


def convert_keys(state_dict: dict) -> dict:
    fmt = detect_format(state_dict)

    if fmt == "diffusers_clip":
        prefix = ""
        if any(k.startswith("conditioner.embedders.") for k in state_dict):
            # Keep only the text encoder embedder weights if a full checkpoint was passed in.
            state_dict = {
                k: v
                for k, v in state_dict.items()
                if ".text_model." in k or k.endswith("text_projection.weight")
            }
        return diffusers_convert.convert_text_enc_state_dict_v20(state_dict, prefix=prefix)

    if fmt == "qwen35_nested":
        return comfy.utils.state_dict_prefix_replace(
            state_dict,
            {
                "model.language_model.": "model.",
                "model.visual.": "visual.",
                "lm_head.": "model.lm_head.",
            },
        )

    # Qwen2.5-VL, Llama, Gemma, etc. already use ComfyUI-compatible names.
    return state_dict


def filter_mode(state_dict: dict, mode: str) -> dict:
    if mode == "full":
        return state_dict

    if mode == "text_only":
        allowed_prefixes = ("model.",)
        allowed_exact = {"lm_head.weight", "model.lm_head.weight"}
        return {
            k: v
            for k, v in state_dict.items()
            if k.startswith(allowed_prefixes) or k in allowed_exact
        }

    if mode == "vl":
        allowed_prefixes = ("model.", "visual.")
        allowed_exact = {"lm_head.weight", "model.lm_head.weight"}
        return {
            k: v
            for k, v in state_dict.items()
            if k.startswith(allowed_prefixes) or k in allowed_exact
        }

    raise ValueError(f"Unknown mode: {mode}")


def default_output_path(input_path: str, mode: str) -> str:
    if os.path.isfile(input_path):
        base, ext = os.path.splitext(input_path)
        suffix = "" if mode == "full" else f"-{mode}"
        return f"{base}-comfy{suffix}{ext or '.safetensors'}"

    name = os.path.basename(input_path.rstrip(os.sep))
    suffix = "" if mode == "full" else f"-{mode}"
    return os.path.join(input_path, f"{name}-comfy{suffix}.safetensors")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input", help="HF model directory, sharded checkpoint dir, or .safetensors file")
    parser.add_argument("-o", "--output", help="Output .safetensors path")
    parser.add_argument(
        "--mode",
        choices=("auto", "full", "text_only", "vl"),
        default="auto",
        help="auto: keep all keys; text_only: model.+lm_head only; vl: model.+visual.+lm_head",
    )
    parser.add_argument("--dry-run", action="store_true", help="Inspect keys only, do not write output")
    args = parser.parse_args()

    resolved = resolve_input_path(args.input)
    print(f"Loading from: {resolved}")
    state_dict = load_state_dict(resolved)
    print(f"Loaded {len(state_dict)} tensors")

    detected = detect_format(state_dict)
    print(f"Detected format: {detected}")

    converted = convert_keys(state_dict)
    mode = "full" if args.mode == "auto" else args.mode
    if args.mode == "auto" and detected == "qwen_vl":
        mode = "vl"
        print("Auto-selected mode: vl")

    filtered = filter_mode(converted, mode)
    print(f"Writing {len(filtered)} tensors (mode={mode})")

    prefixes: dict[str, int] = {}
    for key in filtered:
        prefix = key.split(".", 1)[0]
        prefixes[prefix] = prefixes.get(prefix, 0) + 1
    print("Key prefixes:", ", ".join(f"{k}={v}" for k, v in sorted(prefixes.items())))

    if args.dry_run:
        print("Dry run complete.")
        return 0

    output = args.output or default_output_path(resolved, mode)
    output = os.path.abspath(output)
    os.makedirs(os.path.dirname(output), exist_ok=True)
    print(f"Saving to: {output}")
    comfy.utils.save_torch_file(filtered, output)
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
