#!/usr/bin/env python
"""Isolate one top-level component (VAE, text-embedding projection, ...) from an
LTX-2 full checkpoint into a standalone safetensors, optionally renaming its prefix
so it loads in the matching ComfyUI node.

Run with the ComfyUI venv.

Examples:
  # Video VAE -> standalone ComfyUI VAE (strip the 'vae.' prefix; Load VAE expects that)
  extract_components.py -i checkpoint.safetensors -o ltx_vae_bf16.safetensors \
      --prefix "vae." --to ""

  # text_embedding_projection -> standalone (keep the prefix; the LTX TE loader looks for it)
  extract_components.py -i checkpoint.safetensors -o ltx_te_proj_bf16.safetensors \
      --prefix "text_embedding_projection."
"""
import argparse, os, sys
from safetensors import safe_open

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # import comfy
import comfy.utils

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", required=True)
    ap.add_argument("-o", "--output", required=True)
    ap.add_argument("--prefix", required=True,
                    help="select keys starting with this prefix, e.g. 'vae.'")
    ap.add_argument("--to", default=None,
                    help="rename the matched prefix to this (default: keep prefix; "
                         "pass '' to strip it).")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    to = args.prefix if args.to is None else args.to

    out = {}
    with safe_open(args.input, framework="pt", device="cpu") as f:
        meta = f.metadata() or {}
        keys = [k for k in f.keys() if k.startswith(args.prefix)]
        if not keys:
            sys.exit(f"no keys under prefix {args.prefix!r}")
        print(f"selected {len(keys)} keys under {args.prefix!r}; prefix -> {to!r}")
        for k in keys:
            nk = to + k[len(args.prefix):]
            if args.dry_run:
                if len(out) < 6:
                    print(f"  {k}  ->  {nk}")
                out[nk] = None
            else:
                out[nk] = f.get_tensor(k).contiguous()

    if args.dry_run:
        print(f"(dry-run) would write {len(keys)} tensors")
        return
    comfy.utils.save_torch_file(out, args.output, metadata=meta)
    print(f"wrote {args.output}  ({os.path.getsize(args.output)/1e9:.2f} GB, {len(out)} tensors)")

if __name__ == "__main__":
    main()
