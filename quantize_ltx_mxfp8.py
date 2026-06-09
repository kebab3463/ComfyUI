#!/usr/bin/env python
"""Quantize an LTX 2.3 (or any DiT/LLM) safetensors to ComfyUI-native MXFP8.

Produces a ComfyUI-native quantized checkpoint: each quantized layer stores `weight`
(float8_e4m3fn) + `weight_scale` (e8m0 block scales, block 32, saved as uint8), plus a
`_quantization_metadata` entry in the safetensors header. ComfyUI's loader detects this
and keeps the weights packed in VRAM (dequantizing per-matmul) instead of upcasting to
bf16 -- so the file stays compact when loaded. Works for both the diffusion model
(Load Diffusion Model node) and text encoders (CLIP loader).

Selection rule: quantize Linear weights only -- `.weight`, 2D, both dims >= --min-dim
(default 1024). Norms (1D), biases, conv (4D), and small in/out projections fall out
automatically. Anything matching a --keep-bf16 substring is forced to stay bf16.

ALWAYS run with the ComfyUI venv (needs torch cu130 + comfy_kitchen MXFP8):
  /home/csprunger/ComfyUI/.venv/bin/python quantize_ltx_mxfp8.py ...

Flags:
  -i/--input, -o/--output   safetensors paths
  --min-dim N               quantize 2D linears with min(shape) >= N (default 1024)
  --diffusion-prefix P      override auto-detected diffusion prefix. Auto-detect keeps
                            only that prefix and DROPS everything else (VAE/CLIP), then
                            strips it -> a clean diffusion-only file. Pass "" to keep the
                            WHOLE file with no prefix strip and drop nothing (text encoders).
  --keep-bf16 "a,b,c"       comma-separated substrings of layers to leave in bf16
                            (default: adaln,timestep_embedder,embeddings_connector)
  --quant-connectors        drop 'embeddings_connector' from the keep list (quantize them)
  --dry-run                 print the kept/dropped/quant plan, write nothing (header-only,
                            works on a partially-downloaded file)
  --check-error             print per-layer relative dequant error during the run
  --verify [N]              after writing, reload N sampled layers (default 64) and compare
                            dequant vs source to confirm the file round-trips

====================================================================================
RECIPES (configurations used so far)
====================================================================================

1) LTX 2.3 DIFFUSION MODEL -- recommended (matches known-good + official LTX recipe).
   Keeps adaLN/modulation + timestep + both embeddings_connectors in bf16, quantizes
   attn (q/k/v/out) + ff in all transformer_blocks. ~24GB out, best identity. Drops
   VAE/audio_vae/vocoder/text_embedding_projection (auto, via prefix detection):
     quantize_ltx_mxfp8.py \
       -i sulphur_distil_bf16.safetensors -o sulphur_distil_mxfp8.safetensors --verify

2) LTX 2.3 DIFFUSION MODEL -- smaller, more aggressive (connectors quantized, ~21.7GB).
   Use to A/B whether the connectors actually matter for your outputs:
     quantize_ltx_mxfp8.py \
       -i sulphur_distil_bf16.safetensors -o sulphur_distil_mxfp8_qconn.safetensors \
       --quant-connectors --verify

3) LTX 2.3 DIFFUSION MODEL -- also spare first/last blocks (official "audio" refinement;
   helps input-audio artifacts, not image identity):
     quantize_ltx_mxfp8.py -i in.safetensors -o out.safetensors --verify \
       --keep-bf16 "adaln,timestep_embedder,embeddings_connector,transformer_blocks.0.,transformer_blocks.1.,transformer_blocks.46.,transformer_blocks.47."

4) GEMMA 3 TEXT ENCODER (or any LLM TE) -- keep the WHOLE file (vision_model +
   multi_modal_projector preserved for I2V prompt enhancement), quantize the LLM +
   vision-encoder linears, keep embedding tables / projector / norms in bf16. ~13GB out,
   stays compact in ComfyUI (no need for --fp8_e4m3fn-text-enc):
     quantize_ltx_mxfp8.py \
       -i gemma-3-12b-it-heretic-v2.safetensors \
       -o gemma-3-12b-it-heretic-v2_mxfp8_comfy.safetensors --verify \
       --diffusion-prefix "" \
       --keep-bf16 "embed_tokens,position_embedding,multi_modal_projector,norm"
   (add 'vision_model' to --keep-bf16 to leave the vision tower bf16 for extra I2V fidelity)

Notes:
  - Inspect any safetensors fast by reading only the front header (works mid-download).
  - NVFP4/FP4 halves size but wrecks i2v reference-image fidelity -- avoid for the
    diffusion model. (Fine for text encoders, where quantization is low-risk.)
  - This script does naive round-to-nearest PTQ. The official LTX recipe also uses naive
    casting, so quality comes from LAYER SELECTION, not rounding. For learned rounding /
    bias correction / nvfp4, use convert_to_quant (ctq) at ./convert_to_quant.
"""
import argparse, json, os, sys
from collections import Counter
import torch
from safetensors import safe_open

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # import comfy
from comfy.quant_ops import QuantizedTensor, QUANT_ALGOS, get_layout_class
from comfy import model_detection
import comfy.utils

FORMAT = "mxfp8"
LAYOUT = "TensorCoreMXFP8Layout"
# Identity-sensitive layers kept in bf16 by default. This keep-set matches a known-good
# LTX 2.3 mxfp8 quant (official distilled 1.1, ~24GB): the adaLN/modulation + timestep
# MLP (per-channel shift/scale/gate feeding every block), and BOTH embeddings_connectors
# (the bridge from caption/register conditioning into the model). Quantizing the
# connectors is the main thing that distinguished our weaker output from the good file.
# Cheap relative to quality: ~2.3GB. (in/out projections patchify_proj/proj_out stay
# bf16 automatically via the 1024 min-dim threshold.)
DEFAULT_KEEP = "adaln,timestep_embedder,embeddings_connector"

def is_linear_weight(key, t, min_dim):
    # Linear weights only. In LTX 2.3 every `.weight` that is 2D and >=1024 on both
    # dims is a Linear: norms are 1D, the in/out projections (patchify_proj/proj_out)
    # have a 128 dim so they fall below the threshold, and the only embedding-like
    # params (learnable_registers, scale_shift_table) do not end in `.weight`.
    if not key.endswith(".weight"):
        return False
    if t.dim() != 2:                      # norms/biases (1D), conv (4D/5D) -> bf16
        return False
    if min(t.shape) < min_dim:            # tiny / in-out projections -> bf16
        return False
    if t.dtype not in (torch.bfloat16, torch.float16, torch.float32):
        return False
    return True

def top_ns(keys):
    """First dotted segment of each key, for a human-readable namespace breakdown."""
    return Counter(k.split(".", 1)[0] for k in keys)

def verify(input_path, output_path, prefix, layers, device, n):
    """Reload sampled quantized layers from the saved file and compare their
    dequantized weights to the source bf16, to confirm the on-disk format round-trips
    (rules out a save/load bug vs. a recipe issue)."""
    names = sorted(layers)
    step = max(1, len(names) // n)
    sample = names[::step][:n]
    layout_cls = get_layout_class(LAYOUT)
    print(f"\nverifying {len(sample)} of {len(names)} quantized layers (reload vs source)...")
    errs = []
    with safe_open(input_path, framework="pt", device="cpu") as fsrc, \
         safe_open(output_path, framework="pt", device="cpu") as fout:
        out_keys = set(fout.keys())
        for layer in sample:
            wk, sk = f"{layer}.weight", f"{layer}.weight_scale"
            if wk not in out_keys or sk not in out_keys:
                print(f"  MISSING in output: {layer}")
                errs.append((1.0, layer))
                continue
            w_fp8 = fout.get_tensor(wk).to(device)                 # float8_e4m3fn (may be 32x-padded)
            block_scale = fout.get_tensor(sk).to(device).view(torch.float8_e8m0fnu)
            src = fsrc.get_tensor(f"{prefix}{layer}.weight").to(device).float()
            # orig_shape is the TRUE (unpadded) logical shape -- mxfp8 pads dims to a
            # multiple of 32 on disk; ComfyUI's loader unpads via the layer dims, so we
            # must too (use the source shape, not the stored/padded weight shape).
            params = layout_cls.Params(scale=block_scale, orig_dtype=torch.bfloat16,
                                       orig_shape=tuple(src.shape))
            deq = QuantizedTensor(w_fp8, LAYOUT, params).dequantize().float()
            errs.append((((deq - src).abs().mean() / src.abs().mean().clamp_min(1e-8)).item(), layer))
    e = sorted(r for r, _ in errs)
    mean = sum(e) / len(e)
    print(f"  reload relerr  mean={mean:.4f}  median={e[len(e)//2]:.4f}  max={e[-1]:.4f}")
    print("  worst layers:")
    for r, layer in sorted(errs, reverse=True)[:5]:
        print(f"    {r:.4f}  {layer}")
    if e[-1] > 0.10:
        print("  WARNING: max reload error >0.10 - investigate save/load, not just recipe.")
    else:
        print("  OK: reload matches in-memory error range -> on-disk format is correct.")

def main():
    assert FORMAT in QUANT_ALGOS, "comfy_kitchen lacks MXFP8 — update it"
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", required=True)
    ap.add_argument("-o", "--output", required=True)
    ap.add_argument("--min-dim", type=int, default=1024)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--diffusion-prefix", default=None,
                    help="override the auto-detected diffusion-model prefix (e.g. "
                         "'model.diffusion_model.'). Use '' to treat the whole file as "
                         "the diffusion model.")
    ap.add_argument("--keep-bf16", default=DEFAULT_KEEP,
                    help="comma-separated substrings; matching Linear layers stay bf16. "
                         f"Default: {DEFAULT_KEEP!r}. Pass '' to quantize everything, or add "
                         "e.g. 'attn2,to_video_attn,to_audio_attn' to also spare cross-attention.")
    ap.add_argument("--quant-connectors", action="store_true",
                    help="also quantize the audio/video embeddings_connector layers (drops "
                         "'embeddings_connector' from --keep-bf16). Smaller file (~-2.3GB); the "
                         "known-good and official LTX recipes keep these in bf16.")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--check-error", action="store_true",
                    help="report mean relative dequant error per quantized layer")
    ap.add_argument("--verify", type=int, nargs="?", const=64, default=0,
                    help="after writing, reload N sampled layers (default 64) and compare "
                         "dequant vs source to confirm the file round-trips")
    args = ap.parse_args()
    keep_patterns = [p for p in args.keep_bf16.lower().split(",") if p]
    if args.quant_connectors:
        keep_patterns = [p for p in keep_patterns if p != "embeddings_connector"]

    with safe_open(args.input, framework="pt", device="cpu") as f:
        src_meta = f.metadata() or {}
        keys = list(f.keys())

        prefix = (args.diffusion_prefix
                  if args.diffusion_prefix is not None
                  else model_detection.unet_prefix_from_state_dict(keys))
        kept = [k for k in keys if k.startswith(prefix)]
        dropped = [k for k in keys if not k.startswith(prefix)]

        print(f"detected diffusion prefix: {prefix!r}")
        print(f"keep-bf16 patterns: {keep_patterns}")
        print(f"diffusion keys: {len(kept)}   dropped (VAE/CLIP/other): {len(dropped)}")
        print(f"  kept namespaces:    {dict(top_ns(k[len(prefix):] for k in kept))}")
        print(f"  dropped namespaces: {dict(top_ns(dropped))}")
        if not kept:
            sys.exit("ERROR: no keys under the diffusion prefix — pass --diffusion-prefix explicitly.")

        out_sd, layers = {}, {}
        n_q = n_pass = n_keep_sensitive = 0
        for k in kept:
            t = f.get_tensor(k)
            ok = k[len(prefix):]                          # strip prefix for the output
            qable = is_linear_weight(ok, t, args.min_dim)
            sensitive = qable and any(p in ok.lower() for p in keep_patterns)
            if qable and not sensitive:
                layer = ok[:-len(".weight")]
                n_q += 1
                if args.dry_run:
                    print(f"  QUANT  {layer:64s} {tuple(t.shape)}")
                    continue
                w = t.to(device=args.device, dtype=torch.bfloat16)
                qt = QuantizedTensor.from_float(w, LAYOUT)
                part = qt.state_dict(f"{layer}.weight")
                for pk, pv in part.items():
                    if pv.dtype == torch.float8_e8m0fnu:
                        pv = pv.view(torch.uint8)        # e8m0 not safetensors-storable
                    out_sd[pk] = pv.cpu().contiguous()
                layers[layer] = {"format": FORMAT}
                if args.check_error:
                    err = ((qt.dequantize().float() - w.float()).abs().mean()
                           / w.float().abs().mean().clamp_min(1e-8)).item()
                    print(f"  QUANT  {layer:64s} relerr={err:.4f}")
                del w, qt
            else:
                n_pass += 1
                if sensitive:
                    n_keep_sensitive += 1
                    if args.dry_run:
                        print(f"  KEEP   {ok[:-len('.weight')]:64s} {tuple(t.shape)} (sensitive)")
                if not args.dry_run:
                    out_sd[ok] = t.contiguous()           # pass through (bf16/etc.)

    print(f"\nquantized={n_q}  passthrough(bf16)={n_pass}  "
          f"(of which kept-sensitive linears={n_keep_sensitive})")
    if args.dry_run:
        return

    meta = {k: v for k, v in src_meta.items()             # carry over non-quant metadata
            if k != "_quantization_metadata"}
    meta["_quantization_metadata"] = json.dumps(
        {"format_version": "1.0", "layers": layers})
    comfy.utils.save_torch_file(out_sd, args.output, metadata=meta)
    sz = os.path.getsize(args.output) / 1e9
    print(f"wrote {args.output}  ({sz:.1f} GB)")

    if args.verify:
        verify(args.input, args.output, prefix, layers, args.device, args.verify)

if __name__ == "__main__":
    main()
