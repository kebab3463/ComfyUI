#!/usr/bin/env python
"""Quantize an LTX 2.3 (or any DiT/LLM) safetensors to ComfyUI-native MXFP8.

Produces a ComfyUI-native quantized checkpoint: each quantized layer stores `weight`
(float8_e4m3fn) + `weight_scale` (e8m0 block scales, block 32, saved as uint8), plus a
`_quantization_metadata` entry in the safetensors header. ComfyUI's loader detects this
and keeps the weights packed in VRAM (dequantizing per-matmul) instead of upcasting to
bf16 -- so the file stays compact when loaded. Works for both the diffusion model
(Load Diffusion Model node) and text encoders (CLIP loader).

EVERYTHING AUTO-DETECTS. A bare run picks the right diffusion prefix and the right
sensitive layers to keep in bf16 -- for LTX, FLUX, Gemma/LLM text encoders, and unseen
diffusion/LLM architectures alike:
  quantize_mxfp8.py -i model_bf16.safetensors -o model_mxfp8.safetensors --verify

How it generalizes (no per-model registry):
  - Prefix: 'model.diffusion_model.' if present -> extract the diffusion model, drop
    VAE/TE; otherwise '' -> keep the whole file (raw diffusion model, or an LLM TE with
    its vision_model/projector preserved).
  - Quantize Linear weights only: `.weight`, 2D, both dims >= --min-dim (default 1024).
    Norms (1D), biases, conv (4D), and small in/out projections fall out automatically.
  - Keep in bf16: a universal sensitive-layer name list (embeddings, modulation/adaLN,
    time/guidance conditioning, patch/in/out projections, multimodal projectors) that is
    disjoint from bulk attn/mlp names, plus a structural rule (vocab-sized dim >= 50k =>
    embedding table). This reproduces the hand-tuned LTX/FLUX/Gemma recipes and extends
    to new models. Verified: LTX quantized=1344, FLUX=112, Gemma=498 with no flags.

ALWAYS run with the ComfyUI venv (needs torch cu130 + comfy_kitchen MXFP8):
  /home/csprunger/ComfyUI/.venv/bin/python quantize_mxfp8.py ...

Flags:
  -i/--input, -o/--output   safetensors paths
  --min-dim N               quantize 2D linears with min(shape) >= N (default 1024)
  --diffusion-prefix P      override the auto prefix (e.g. force "" to keep a full file)
  --keep-bf16 "a,b,c"       override the universal keep-list with your own substrings
  --quant-connectors        LTX: force-quantize the embeddings_connectors (~-2.3GB; the
                            default keeps them bf16, matching the known-good/official recipe)
  --dry-run                 print the kept/dropped/quant plan, write nothing (header-only,
                            works on a partially-downloaded file)
  --check-error             print per-layer relative dequant error during the run
  --verify [N]              after writing, reload N sampled layers (default 64) and compare
                            dequant vs source to confirm the file round-trips

Overrides for special cases:
  - LTX, spare first/last blocks too (official "audio" refinement, not image identity):
      --keep-bf16 "embed,modulation,adaln,time_,transformer_blocks.0.,transformer_blocks.1.,transformer_blocks.46.,transformer_blocks.47."
  - Gemma TE, also keep the vision tower bf16 (extra I2V-prompt fidelity, ~+0.5GB):
      append ',vision_model' to a custom --keep-bf16.

Notes:
  - Inspect any safetensors fast by reading only the front header (works mid-download).
  - NVFP4/FP4 halves size but wrecks i2v reference-image fidelity -- avoid for the
    diffusion model. (Fine for text encoders, where quantization is low-risk.)
  - This script does naive round-to-nearest PTQ. The official LTX recipe also uses naive
    casting, so quality comes from LAYER SELECTION, not rounding. For learned rounding /
    bias correction / nvfp4, use convert_to_quant (ctq) at ./convert_to_quant.
"""
import argparse
import datetime
import json
import os
import re
import sys
import threading
import time
from collections import Counter, defaultdict
import torch
from safetensors import safe_open

# Find the ComfyUI root (the dir containing comfy/) by walking up from this file, so the
# script works from any subfolder (e.g. quantization_and_conversion/).
_root = os.path.dirname(os.path.abspath(__file__))
while _root != os.path.dirname(_root) and not os.path.isdir(os.path.join(_root, "comfy")):
    _root = os.path.dirname(_root)
sys.path.insert(0, _root)
from comfy.quant_ops import QuantizedTensor, QUANT_ALGOS, get_layout_class
import comfy.utils

try:                                                 # progress bar (graceful no-op if absent)
    from tqdm import tqdm
    _HAVE_TQDM = True
except ImportError:
    _HAVE_TQDM = False
    def tqdm(it=None, **kw):
        return it
_log = tqdm.write if _HAVE_TQDM else print           # print without breaking the bar

FORMAT_LAYOUT = {"mxfp8": "TensorCoreMXFP8Layout", "nvfp4": "TensorCoreNVFP4Layout"}
# nvfp4 per-tensor scale denominator = F8_E4M3_MAX * F4_E2M1_MAX (= 448 * 6); used to turn a
# calibrated activation amax into an input_scale (and matches the weight "recalculate" path).
NVFP4_SCALE_DENOM = 448.0 * 6.0

# Universal "keep in bf16" substrings: the union of identity-sensitive layer names across
# diffusion (LTX, FLUX, ...) and LLM/vision text encoders. These names are disjoint from
# the heavy matmuls that SHOULD be quantized (attn/qkv/proj/mlp/ff/fc/linear1/linear2/
# gate/up/down), so the union both reproduces the per-arch recipes we hand-tuned AND
# generalizes to unseen architectures: it keeps embeddings, modulation/adaLN, time/guidance
# conditioning, patch/in/out projections, and multimodal projectors in bf16 while
# quantizing the bulk. Two structural rules back it up:
#   - small in/out projections self-exclude via --min-dim (a data-sized dim < 1024);
#   - any 2D weight with a vocab-sized dim (>= VOCAB_DIM) is treated as an embedding table
#     and kept bf16 even when its name has no telltale (it must never be Linear-quantized).
# We keep GLOBAL modulation bf16 (a handful of layers: FLUX.2 *_stream_modulation, LTX
# adaln_single, final_layer, norm_out). PER-BLOCK modulation (Qwen-Image/FLUX.1 img_mod/txt_mod)
# is format-aware: QUANTIZED at mxfp8 (8-bit, near-lossless, ~1/3 of those models' params) but
# kept bf16 at nvfp4 (4-bit is sensitive to modulation error) -- see NVFP4_KEEP_EXTRA below.
# Override the whole list with --keep-bf16 (disables the format-aware add); force connectors
# back in with --quant-connectors.
DEFAULT_KEEP = ("embed,wte,modulation,adaln,ada_ln,time_,guidance,"
                "final_layer,norm_out,img_in,txt_in,patch,projector,merger")
# Appended to the default keep-list for nvfp4 only: per-block modulation stays bf16 at 4-bit.
NVFP4_KEEP_EXTRA = "img_mod.,txt_mod."
VOCAB_DIM = 50000  # FFN dims top out ~36k in these models; >=50k => embedding/vocab table

def detect_prefix(keys):
    """Generic: a full checkpoint exposes the diffusion model under 'model.diffusion_model.'
    -> use it (extract the diffusion model, dropping VAE/TE). A raw diffusion model or an
    LLM text encoder has no such prefix -> "" keeps the whole file (nothing dropped)."""
    P = "model.diffusion_model."
    return P if any(k.startswith(P) for k in keys) else ""

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

def verify(input_path, output_path, prefix, layers, device, n, fmt, layout):
    """Reload sampled quantized layers from the saved file and compare their
    dequantized weights to the source bf16, to confirm the on-disk format round-trips
    (rules out a save/load bug vs. a recipe issue). Reconstructs the QuantizedTensor
    exactly as ComfyUI's loader does, so passing it is strong evidence the file loads."""
    names = sorted(layers)
    step = max(1, len(names) // n)
    sample = names[::step][:n]
    layout_cls = get_layout_class(layout)
    print(f"\nverifying {len(sample)} of {len(names)} quantized layers ({fmt}, reload vs source)...")
    errs = []
    with safe_open(input_path, framework="pt", device="cpu") as fsrc, \
         safe_open(output_path, framework="pt", device="cpu") as fout:
        out_keys = set(fout.keys())
        for layer in tqdm(sample, desc="verifying", unit="layer", disable=not _HAVE_TQDM):
            wk, sk = f"{layer}.weight", f"{layer}.weight_scale"
            if wk not in out_keys or sk not in out_keys:
                _log(f"  MISSING in output: {layer}")
                errs.append((1.0, layer))
                continue
            qdata = fout.get_tensor(wk).to(device)                 # mxfp8: fp8 (maybe padded); nvfp4: uint4-packed uint8
            src = fsrc.get_tensor(f"{prefix}{layer}.weight").to(device).float()
            # orig_shape is the TRUE (unpadded) logical shape -- the loader unpads via the
            # layer dims, so we use the source shape, not the stored (possibly padded) one.
            if fmt == "nvfp4":
                bs = fout.get_tensor(sk).to(device)                # weight_scale: fp8 block scale
                ts = fout.get_tensor(f"{layer}.weight_scale_2").to(device)  # weight_scale_2: f32 global
                params = layout_cls.Params(scale=ts, block_scale=bs, orig_dtype=torch.bfloat16,
                                           orig_shape=tuple(src.shape))
            else:  # mxfp8: e8m0 block scale stored as uint8
                bs = fout.get_tensor(sk).to(device).view(torch.float8_e8m0fnu)
                params = layout_cls.Params(scale=bs, orig_dtype=torch.bfloat16,
                                           orig_shape=tuple(src.shape))
            deq = QuantizedTensor(qdata, layout, params).dequantize().float()
            errs.append((((deq - src).abs().mean() / src.abs().mean().clamp_min(1e-8)).item(), layer))
    e = sorted(r for r, _ in errs)
    mean = sum(e) / len(e)
    worst = sorted(errs, reverse=True)[:5]
    print(f"  reload relerr  mean={mean:.4f}  median={e[len(e)//2]:.4f}  max={e[-1]:.4f}")
    print("  worst layers:")
    for r, layer in worst:
        print(f"    {r:.4f}  {layer}")
    if e[-1] > 0.10:
        print("  WARNING: max reload error >0.10 - investigate save/load, not just recipe.")
    else:
        print("  OK: reload matches in-memory error range -> on-disk format is correct.")
    return {"sampled": len(sample), "mean": mean, "median": e[len(e) // 2], "max": e[-1],
            "worst": worst}

def classify(key, t, min_dim, keep_patterns, force_quant):
    """Single source of truth for what gets quantized and WHY. Returns (action, reason)
    where action is 'quant' or 'keep'. The loop and the report both use this."""
    if not key.endswith(".weight"):
        return "keep", "non-weight tensor (bias / scale_shift / param)"
    if t.dim() != 2:
        return "keep", f"{t.dim()}D weight (norm / conv — not a Linear)"
    if min(t.shape) < min_dim:
        return "keep", f"min dim {min(t.shape)} < {min_dim} (small in/out projection)"
    if t.dtype not in (torch.bfloat16, torch.float16, torch.float32):
        return "keep", f"dtype {t.dtype} (not a float weight)"
    name = key.lower()                                    # quantizable 2D Linear from here
    if any(fp in name for fp in force_quant):
        return "quant", None                              # force-quant overrides keep
    if max(t.shape) >= VOCAB_DIM:
        return "keep", f"vocab/embedding table (dim {max(t.shape)} >= {VOCAB_DIM})"
    matched = next((p for p in keep_patterns if p in name), None)
    if matched is not None:
        return "keep", f"sensitive name: matched '{matched}'"
    return "quant", None

def _table(rows):
    """rows: list of (pattern, count, shape). -> markdown table string."""
    out = ["| layer pattern | count | example shape |", "|---|---:|---|"]
    for pat, cnt, shape in rows:
        out.append(f"| `{pat}` | {cnt} | `{tuple(shape)}` |")
    return "\n".join(out)

def write_report(path, *, args, fmt, prefix, dropped_ns, rep_q, rep_k, q_params, bf_params,
                 in_size, out_size, n_q, n_pass, calib, n_input_scale, verify_stats):
    """Write a Markdown report stating exactly what is and isn't quantized (and why)."""
    total = q_params + bf_params
    pq = 100 * q_params / total if total else 0
    fmt_desc = {"mxfp8": "block-32 microscaling FP8 (E4M3 + E8M0 block scales)",
                "nvfp4": "4-bit NVFP4 (E2M1 + group-16 FP8 block scale + per-tensor FP32 scale)"}[fmt]
    L = []
    L.append(f"# Quantization report — `{os.path.basename(args.output)}`\n")
    L.append(f"- **Source:** `{os.path.basename(args.input)}` — {in_size:.2f} GB (bf16)")
    L.append(f"- **Output:** `{os.path.basename(args.output)}` — {out_size:.2f} GB "
             f"({100*out_size/in_size:.0f}% of source)")
    L.append(f"- **Format:** {fmt} — {fmt_desc}; ComfyUI-native (`comfy_quant` metadata, no custom nodes)")
    act = "post-training round-to-nearest, weight-only" + (
        ", + calibrated activation `input_scale`" if fmt == "nvfp4" and calib else "")
    L.append(f"- **Method:** {act}")
    L.append(f"- **Generated:** {datetime.date.today().isoformat()} by `quantize_model_blackwell.py`")
    L.append(f"- **Command:** `{' '.join(os.path.basename(a) if i == 0 else a for i, a in enumerate(sys.argv))}`\n")

    L.append("## Summary\n")
    L.append("| disposition | layers | parameters |")
    L.append("|---|---:|---:|")
    L.append(f"| Quantized ({fmt}) | {n_q} | {q_params/1e9:.2f} B ({pq:.0f}%) |")
    L.append(f"| Kept in bf16 | {n_pass} | {bf_params/1e9:.2f} B ({100-pq:.0f}%) |")
    L.append(f"| **Total** | {n_q+n_pass} | {total/1e9:.2f} B |\n")
    L.append(f"- **Diffusion prefix:** `{prefix!r}`"
             + ("" if not dropped_ns else f" — dropped (not in output): {dropped_ns}"))
    L.append(f"- **Selection rule:** quantize 2D Linear `.weight` with min(dim) ≥ {args.min_dim}, "
             "except names in the keep-list and vocab/embedding tables.")
    keep_eff = "DEFAULT_KEEP" + (" + NVFP4_KEEP_EXTRA" if fmt == "nvfp4" and args.keep_bf16 is None else "")
    L.append(f"- **keep-bf16 patterns** ({keep_eff}): "
             f"`{args.keep_bf16 if args.keep_bf16 is not None else DEFAULT_KEEP + (',' + NVFP4_KEEP_EXTRA if fmt=='nvfp4' else '')}`\n")
    if fmt == "nvfp4" and calib:
        L.append(f"- **Calibration:** `input_scale` applied to {n_input_scale}/{n_q} quantized layers "
                 f"from `{os.path.basename(args.calib)}` ({len(calib)} entries).\n")

    L.append(f"## ✅ Quantized layers ({fmt})\n")
    qrows = sorted(((p, c[0], c[1]) for p, c in rep_q.items()), key=lambda r: r[0])
    L.append(_table(qrows) + "\n")

    L.append("## ⏸️ Kept in bf16 (NOT quantized), by reason\n")
    for reason in sorted(rep_k, key=lambda r: -sum(c[0] for c in rep_k[r].values())):
        d = rep_k[reason]
        tot = sum(c[0] for c in d.values())
        L.append(f"### {reason}  — {tot} tensor(s)\n")
        L.append(_table(sorted(((p, c[0], c[1]) for p, c in d.items()), key=lambda r: r[0])) + "\n")

    if verify_stats:
        v = verify_stats
        floor = ("~0.02 is near the mxfp8 representational floor" if fmt == "mxfp8"
                 else "~0.09 is expected for 4-bit nvfp4")
        L.append("## Quality — reload verification\n")
        L.append(f"Round-trip dequantization error over {v['sampled']} sampled quantized layers, "
                 "reconstructed exactly as ComfyUI's loader does (so it doubles as a load check):\n")
        L.append(f"- relative L1 error: **mean {v['mean']:.4f}, median {v['median']:.4f}, "
                 f"max {v['max']:.4f}** ({floor}).")
        L.append("- worst layers: " + ", ".join(f"`{lay}` {r:.4f}" for r, lay in v["worst"]) + "\n")

    is_te = any("model.layers" in p or p.startswith("visual.") for p in rep_q)
    loader = "**CLIPLoader**" if is_te else "**Load Diffusion Model**"
    L.append("## Loading in ComfyUI\n")
    L.append(f"Load with {loader}; the `comfy_quant` metadata is auto-detected and the weights stay "
             "quantized in VRAM (no custom nodes required). Requires a Blackwell GPU "
             f"({'nvfp4' if fmt == 'nvfp4' else 'mxfp8'} tensor-core compute).\n")

    with open(path, "w") as fh:
        fh.write("\n".join(L))
    print(f"wrote quantization report -> {path}")

def save_with_progress(out_sd, path, metadata):
    """Save the safetensors with a disk-write progress bar. safetensors writes the whole
    dict in one opaque call, so we poll the growing output file from a background thread."""
    if not _HAVE_TQDM:
        comfy.utils.save_torch_file(out_sd, path, metadata=metadata)
        return
    total = sum(t.numel() * t.element_size() for t in out_sd.values())
    stop = threading.Event()

    def monitor():
        bar = tqdm(total=total, unit="B", unit_scale=True, unit_divisor=1024, desc="writing -> disk")
        last = 0
        while not stop.wait(0.25):
            cur = os.path.getsize(path) if os.path.exists(path) else 0
            if cur > last:
                bar.update(cur - last)
                last = cur
        cur = os.path.getsize(path) if os.path.exists(path) else last
        if cur > last:
            bar.update(cur - last)
        bar.n = min(bar.total, bar.n)            # avoid a cosmetic >100% from header bytes
        bar.refresh()
        bar.close()

    th = threading.Thread(target=monitor, daemon=True)
    th.start()
    try:
        comfy.utils.save_torch_file(out_sd, path, metadata=metadata)
    finally:
        stop.set()
        th.join(timeout=3)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", required=True)
    ap.add_argument("-o", "--output", required=True)
    ap.add_argument("--format", choices=list(FORMAT_LAYOUT), default="mxfp8",
                    help="mxfp8 (8-bit, default, ~RTN-optimal) or nvfp4 (4-bit, needs mixed "
                         "precision + ideally --calib for activation input_scale).")
    ap.add_argument("--calib", default=None,
                    help="nvfp4 only: a calibration safetensors of {layer: activation_amax} "
                         "(from the ComfyUI calibration node) -> writes a per-layer input_scale "
                         "for better 4-bit activation quant. Omit for dynamic activation scaling.")
    ap.add_argument("--min-dim", type=int, default=1024)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--diffusion-prefix", default=None,
                    help="override the auto-detected diffusion-model prefix. Auto: "
                         "'model.diffusion_model.' if present (extract diffusion, drop VAE/TE), "
                         "else '' (keep whole file). Pass '' to force whole-file.")
    ap.add_argument("--keep-bf16", default=None,
                    help="comma-separated substrings; matching Linear layers stay bf16. Default: a "
                         "universal sensitive-layer list that auto-generalizes across diffusion/LLM "
                         "architectures (and for nvfp4 also keeps per-block modulation). Pass '' to "
                         "quantize everything, or your own list (disables the format-aware add).")
    ap.add_argument("--quant-connectors", action="store_true",
                    help="force-quantize the audio/video embeddings_connector layers (LTX) even "
                         "though the default keeps them bf16. Smaller file (~-2.3GB); the "
                         "known-good and official LTX recipes keep these in bf16.")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--check-error", action="store_true",
                    help="report mean relative dequant error per quantized layer")
    ap.add_argument("--verify", type=int, nargs="?", const=64, default=0,
                    help="after writing, reload N sampled layers (default 64) and compare "
                         "dequant vs source to confirm the file round-trips")
    ap.add_argument("--report", nargs="?", const="", default=None,
                    help="write a Markdown report listing exactly which layers are quantized vs "
                         "kept bf16 (and why), for sharing (e.g. a HuggingFace model card). "
                         "Bare --report -> '<output>.quant_report.md'; or give a path. Combine with "
                         "--verify to include the round-trip quality numbers.")
    args = ap.parse_args()
    assert args.format in QUANT_ALGOS, f"comfy_kitchen lacks {args.format} support — update it"
    fmt, layout = args.format, FORMAT_LAYOUT[args.format]
    if args.keep_bf16 is not None:
        keep_src = args.keep_bf16                 # explicit override: use exactly as given
    else:
        keep_src = DEFAULT_KEEP                    # default + format-aware additions
        if fmt == "nvfp4":
            keep_src += "," + NVFP4_KEEP_EXTRA     # keep per-block modulation bf16 at 4-bit
    keep_patterns = [p for p in keep_src.lower().split(",") if p]
    force_quant = ["embeddings_connector"] if args.quant_connectors else []

    # nvfp4 calibration: {layer -> activation amax} -> per-layer input_scale
    calib = {}
    if args.calib:
        with safe_open(args.calib, framework="pt", device="cpu") as cf:
            calib = {k: cf.get_tensor(k).float() for k in cf.keys()}
        print(f"loaded calibration for {len(calib)} layers from {args.calib}")

    with safe_open(args.input, framework="pt", device="cpu") as f:
        src_meta = f.metadata() or {}
        keys = list(f.keys())

        prefix = (args.diffusion_prefix
                  if args.diffusion_prefix is not None
                  else detect_prefix(keys))
        kept = [k for k in keys if k.startswith(prefix)]
        dropped = [k for k in keys if not k.startswith(prefix)]

        print(f"detected diffusion prefix: {prefix!r}")
        print(f"keep-bf16 patterns: {keep_patterns}{'  + force-quant '+str(force_quant) if force_quant else ''}")
        print(f"diffusion keys: {len(kept)}   dropped (VAE/CLIP/other): {len(dropped)}")
        print(f"  kept namespaces:    {dict(top_ns(k[len(prefix):] for k in kept))}")
        print(f"  dropped namespaces: {dict(top_ns(dropped))}")
        if not kept:
            sys.exit("ERROR: no keys under the diffusion prefix — pass --diffusion-prefix explicitly.")

        out_sd, layers = {}, {}
        n_q = n_pass = n_keep_sensitive = n_input_scale = q_params = bf_params = 0
        rep_q, rep_k = {}, defaultdict(dict)              # report: quantized / kept-by-reason
        pbar = tqdm(kept, desc=f"quantizing -> {fmt}", unit="layer", disable=args.dry_run)
        for k in pbar:
            t = f.get_tensor(k)
            ok = k[len(prefix):]                          # strip prefix for the output
            action, reason = classify(ok, t, args.min_dim, keep_patterns, force_quant)
            pat = re.sub(r"\.\d+\.", ".N.", ok)           # collapse block indices for grouping
            if action == "quant":
                layer = ok[:-len(".weight")]
                n_q += 1; q_params += t.numel()
                rep_q.setdefault(pat, [0, tuple(t.shape)])[0] += 1
                if args.dry_run:
                    print(f"  QUANT  {layer:64s} {tuple(t.shape)}")
                    continue
                w = t.to(device=args.device, dtype=torch.bfloat16)
                qt = QuantizedTensor.from_float(w, layout)   # nvfp4 recalculates its global scale
                part = qt.state_dict(f"{layer}.weight")
                for pk, pv in part.items():
                    if pv.dtype == torch.float8_e8m0fnu:
                        pv = pv.view(torch.uint8)        # mxfp8 e8m0 not safetensors-storable
                    out_sd[pk] = pv.cpu().contiguous()
                layers[layer] = {"format": fmt}
                if fmt == "nvfp4" and layer in calib:    # calibrated activation scale
                    out_sd[f"{layer}.input_scale"] = (calib[layer] / NVFP4_SCALE_DENOM).float().cpu()
                    n_input_scale += 1
                if args.check_error:
                    err = ((qt.dequantize().float() - w.float()).abs().mean()
                           / w.float().abs().mean().clamp_min(1e-8)).item()
                    _log(f"  QUANT  {layer:64s} relerr={err:.4f}")
                del w, qt
            else:
                n_pass += 1; bf_params += t.numel()
                rep_k[reason].setdefault(pat, [0, tuple(t.shape)])[0] += 1
                if reason.startswith(("sensitive", "vocab")):
                    n_keep_sensitive += 1
                    if args.dry_run:
                        print(f"  KEEP   {ok[:-len('.weight')]:64s} {tuple(t.shape)} ({reason})")
                if not args.dry_run:
                    out_sd[ok] = t.contiguous()           # pass through (bf16/etc.)
            if _HAVE_TQDM and not args.dry_run:
                pbar.set_postfix(quantized=n_q, kept_bf16=n_pass, refresh=False)

    print(f"\nquantized={n_q}  passthrough(bf16)={n_pass}  "
          f"(of which kept-sensitive linears={n_keep_sensitive})")
    if calib:
        print(f"input_scale: matched {n_input_scale}/{n_q} quantized layers "
              f"(calib file had {len(calib)} entries)")
        if n_input_scale == 0:
            print("  WARNING: no calib names matched quantizer layer names — the calibration node's "
                  "module names differ from the quantizer keys; input_scale was NOT applied.")
        elif n_input_scale < n_q:
            missing = n_q - n_input_scale
            print(f"  note: {missing} quantized layers had no calib entry (those use dynamic "
                  "activation scaling at runtime).")
    if args.dry_run:
        return

    meta = {k: v for k, v in src_meta.items()             # carry over non-quant metadata
            if k != "_quantization_metadata"}
    meta["_quantization_metadata"] = json.dumps(
        {"format_version": "1.0", "layers": layers})
    meta["quantization_summary"] = json.dumps(            # self-documenting (visible on HF)
        {"tool": "quantize_model_blackwell.py", "format": fmt, "quantized_layers": n_q,
         "bf16_layers": n_pass, "quantized_params": q_params, "bf16_params": bf_params,
         "min_dim": args.min_dim, "diffusion_prefix": prefix})
    save_with_progress(out_sd, args.output, meta)
    out_size = os.path.getsize(args.output) / 1e9
    print(f"wrote {args.output}  ({out_size:.1f} GB)")

    verify_stats = None
    if args.verify:
        verify_stats = verify(args.input, args.output, prefix, layers, args.device,
                              args.verify, fmt, layout)

    if args.report is not None:
        report_path = args.report or (os.path.splitext(args.output)[0] + ".quant_report.md")
        write_report(report_path, args=args, fmt=fmt, prefix=prefix,
                     dropped_ns=dict(top_ns(dropped)) if dropped else None,
                     rep_q=rep_q, rep_k=rep_k, q_params=q_params, bf_params=bf_params,
                     in_size=os.path.getsize(args.input) / 1e9, out_size=out_size,
                     n_q=n_q, n_pass=n_pass, calib=calib, n_input_scale=n_input_scale,
                     verify_stats=verify_stats)

if __name__ == "__main__":
    main()
