"""Convert ComfyUI Z-Image Turbo safetensors into a diffusers pipeline folder.

Takes three separate safetensors (diffusion transformer, Qwen3 text encoder, VAE)
and writes a layout compatible with diffusers / ai-toolkit (arch: zimage).

Requires diffusers with Z-Image support. Install from the local clone:
  /home/csprunger/ComfyUI/.venv/bin/pip install -e quantization_and_conversion/diffusers

Examples:
  # Full conversion (downloads configs/tokenizer/scheduler from the base repo first)
  python convert_zimage_to_diffusers.py \\
    --transformer models/diffusion_models/ZIB/my_finetune.safetensors \\
    --text-encoder models/text_encoders/ZIT/qwen_3_4b.safetensors \\
    --vae models/vae/ZIT/ae.safetensors \\
    -o /path/to/my-zimage-diffusers

  # Skip skeleton download if you already have the output folder populated
  python convert_zimage_to_diffusers.py ... -o /path/to/out --no-download-skeleton

  # Inspect key prefixes without writing weights
  python convert_zimage_to_diffusers.py ... --dry-run
"""

from __future__ import annotations

import argparse
import os
import sys

TE_PREFIXES = (
    "qwen3_4b.transformer.",
    "text_encoders.qwen3_4b.transformer.",
)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--transformer",
        required=True,
        help="ComfyUI diffusion model .safetensors (keys under model.diffusion_model.*)",
    )
    ap.add_argument(
        "--text-encoder",
        required=True,
        help="Qwen3 text encoder .safetensors (keys under model.*)",
    )
    ap.add_argument(
        "--vae",
        required=True,
        help="VAE .safetensors (keys under encoder.* / decoder.*)",
    )
    ap.add_argument(
        "-o",
        "--output",
        required=True,
        help="Output diffusers pipeline directory",
    )
    ap.add_argument(
        "--base-repo",
        default="Tongyi-MAI/Z-Image-Turbo",
        help="HF repo id for configs, tokenizer, and scheduler (default: %(default)s)",
    )
    ap.add_argument(
        "--no-download-skeleton",
        action="store_true",
        help="Do not download non-weight files from --base-repo",
    )
    ap.add_argument(
        "--transformer-dtype",
        choices=("bf16", "fp32"),
        default="bf16",
        help="dtype for the saved transformer weights (default: %(default)s)",
    )
    ap.add_argument(
        "--text-encoder-dtype",
        choices=("bf16", "fp32"),
        default="bf16",
        help="dtype for the saved text encoder weights (default: %(default)s)",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Print key-prefix checks only; do not download or write weights",
    )
    return ap.parse_args()


def torch_dtype(name: str):
    import torch

    return torch.bfloat16 if name == "bf16" else torch.float32


def top_namespaces(path: str) -> dict[str, int]:
    from safetensors import safe_open

    counts: dict[str, int] = {}
    with safe_open(path, framework="pt", device="cpu") as f:
        for key in f.keys():
            ns = key.split(".", 1)[0]
            counts[ns] = counts.get(ns, 0) + 1
    return counts


def inspect_inputs(args: argparse.Namespace) -> None:
    for label, path in (
        ("transformer", args.transformer),
        ("text_encoder", args.text_encoder),
        ("vae", args.vae),
    ):
        path = os.path.abspath(path)
        if not os.path.isfile(path):
            raise FileNotFoundError(path)
        ns = top_namespaces(path)
        print(f"{label}: {path}")
        print(f"  top-level namespaces: {dict(sorted(ns.items()))}")


def download_skeleton(output: str, base_repo: str) -> None:
    from huggingface_hub import snapshot_download

    print(f"Downloading configs/tokenizer/scheduler from {base_repo} ...")
    snapshot_download(
        base_repo,
        local_dir=output,
        ignore_patterns=["*.safetensors", "*.bin"],
    )


def normalize_text_encoder_state_dict(state_dict: dict) -> tuple[dict, str | None]:
    for prefix in TE_PREFIXES:
        if any(k.startswith(prefix) for k in state_dict):
            return (
                {
                    k[len(prefix) :] if k.startswith(prefix) else k: v
                    for k, v in state_dict.items()
                },
                prefix,
            )
    return state_dict, None


def convert_transformer(args: argparse.Namespace, output: str) -> None:
    import torch
    from diffusers import ZImageTransformer2DModel

    dtype = torch_dtype(args.transformer_dtype)
    print("Converting transformer ...")
    transformer = ZImageTransformer2DModel.from_single_file(
        os.path.abspath(args.transformer),
        config=args.base_repo,
        subfolder="transformer",
        torch_dtype=dtype,
    )
    transformer.save_pretrained(
        os.path.join(output, "transformer"),
        safe_serialization=True,
    )
    del transformer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("  wrote transformer/")


def convert_text_encoder(args: argparse.Namespace, output: str) -> None:
    import torch
    from safetensors.torch import load_file
    from transformers import Qwen3ForCausalLM

    dtype = torch_dtype(args.text_encoder_dtype)
    print("Converting text encoder ...")
    te = Qwen3ForCausalLM.from_pretrained(
        args.base_repo,
        subfolder="text_encoder",
        torch_dtype=dtype,
    )
    sd = load_file(os.path.abspath(args.text_encoder))
    sd, stripped = normalize_text_encoder_state_dict(sd)
    if stripped:
        print(f"  stripped ComfyUI prefix: {stripped!r}")

    missing, unexpected = te.load_state_dict(sd, strict=False)
    print(f"  load_state_dict: missing={len(missing)}, unexpected={len(unexpected)}")
    if missing:
        print(f"  missing sample: {missing[:5]}")
    if unexpected:
        print(f"  unexpected sample: {unexpected[:5]}")
    if missing:
        raise SystemExit(
            "Text encoder keys did not match Qwen3ForCausalLM. "
            "Run with --dry-run and inspect key prefixes."
        )

    te.save_pretrained(
        os.path.join(output, "text_encoder"),
        safe_serialization=True,
    )
    del te, sd
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("  wrote text_encoder/")


def convert_vae(args: argparse.Namespace, output: str) -> None:
    import torch
    from diffusers import AutoencoderKL

    print("Converting VAE ...")
    vae = AutoencoderKL.from_single_file(
        os.path.abspath(args.vae),
        config=args.base_repo,
        subfolder="vae",
        torch_dtype=torch.float32,
    )
    vae.save_pretrained(
        os.path.join(output, "vae"),
        safe_serialization=True,
    )
    del vae
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("  wrote vae/")


def main() -> int:
    args = parse_args()
    output = os.path.abspath(args.output)

    inspect_inputs(args)
    if args.dry_run:
        print("Dry run complete.")
        return 0

    os.makedirs(output, exist_ok=True)
    if not args.no_download_skeleton:
        download_skeleton(output, args.base_repo)

    convert_transformer(args, output)
    # convert_text_encoder(args, output)
    # convert_vae(args, output)

    print(f"Done. Diffusers folder: {output}")
    print("Point ai-toolkit at it with:")
    print(f"  model.name_or_path: {output}")
    print("  model.arch: zimage")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
