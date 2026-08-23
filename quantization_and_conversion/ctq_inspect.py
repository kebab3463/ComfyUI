"""Inspect a ctq-quantized .safetensors: per-layer format and ConvRot group sizes.

Reads only the safetensors header plus the tiny .comfy_quant marker tensors,
so it is fast even on a 20 GB checkpoint.

Usage:
    python ctq_inspect.py model.safetensors            # summary
    python ctq_inspect.py model.safetensors --per-layer
    python ctq_inspect.py model.safetensors --skipped  # layers with no rotation
"""

import argparse
import json
from collections import Counter, defaultdict

from safetensors import safe_open


def tensor_to_dict(t):
    return json.loads(bytes(t.tolist()).decode("utf-8"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument(
        "--per-layer", action="store_true", help="print every layer"
    )
    ap.add_argument(
        "--skipped",
        action="store_true",
        help="list quantized layers that did NOT get a rotation",
    )
    args = ap.parse_args()

    configs = {}  # base_name -> comfy_quant dict
    in_features = {}  # base_name -> weight.shape[1]
    header_meta = None

    with safe_open(args.path, framework="pt") as f:
        meta = f.metadata() or {}
        if "_quantization_metadata" in meta:
            header_meta = json.loads(meta["_quantization_metadata"])

        keys = list(f.keys())
        for k in keys:
            if k.endswith(".comfy_quant"):
                configs[k[: -len(".comfy_quant")]] = tensor_to_dict(
                    f.get_tensor(k)
                )
            elif k.endswith(".weight"):
                shape = f.get_slice(k).get_shape()
                if len(shape) == 2:
                    in_features[k[: -len(".weight")]] = shape[1]

    if not configs:
        print(
            "No .comfy_quant tensors found — this file was not written with "
            "--comfy_quant, or is not a ctq output."
        )
        if header_meta:
            print(
                "It does carry a _quantization_metadata header; showing that instead."
            )
            layers = header_meta.get("layers", {})
            print(
                f"  {len(layers)} layer entries, "
                f"format_version={header_meta.get('format_version')}"
            )
        return

    print(
        f"{len(configs)} quantized layers  |  "
        f"{len(in_features) - len(configs)} 2-D weights left unquantized"
    )
    if header_meta is not None:
        n = len(header_meta.get("layers", {}))
        print(f"_quantization_metadata header present ({n} entries)")
        print(
            "  note: the regenerated header may omit convrot keys — the "
            ".comfy_quant tensors below are authoritative"
        )
    print()

    fmt_counts = Counter(c.get("format", "?") for c in configs.values())
    print("formats:")
    for fmt, n in fmt_counts.most_common():
        print(f"  {fmt:<28} {n:>5}")
    print()

    gs_counts = Counter()
    gs_by_dim = defaultdict(Counter)
    unrotated = []
    for name, cfg in configs.items():
        n_in = in_features.get(name)
        if cfg.get("convrot"):
            g = cfg.get("convrot_groupsize")
            gs_counts[g] += 1
            gs_by_dim[n_in][g] += 1
        else:
            gs_counts[None] += 1
            gs_by_dim[n_in][None] += 1
            unrotated.append((name, n_in, cfg.get("format")))

    total = len(configs)
    rotated = total - gs_counts.get(None, 0)
    print(
        f"ConvRot: {rotated}/{total} quantized layers rotated "
        f"({100 * rotated / total:.1f}%)"
    )
    for g, n in sorted(
        gs_counts.items(), key=lambda x: (x[0] is None, x[0] or 0)
    ):
        label = "NOT rotated" if g is None else f"group_size {g}"
        print(f"  {label:<20} {n:>5}")
    print()

    print("by in_features:")
    print(f"  {'in_features':>12} {'layers':>7}  group sizes chosen")
    for n_in in sorted(gs_by_dim, key=lambda d: (d is None, d or 0)):
        counts = gs_by_dim[n_in]
        detail = ", ".join(
            f"{'none' if g is None else g}\u00d7{c}"
            for g, c in sorted(
                counts.items(), key=lambda x: (x[0] is None, x[0] or 0)
            )
        )
        print(f"  {str(n_in):>12} {sum(counts.values()):>7}  {detail}")

    if args.skipped and unrotated:
        print(f"\nunrotated quantized layers ({len(unrotated)}):")
        for name, n_in, fmt in sorted(
            unrotated, key=lambda r: (r[1] or 0, r[0])
        ):
            print(f"  in={str(n_in):>6}  {fmt:<20} {name}")

    if args.per_layer:
        print("\nper-layer:")
        for name in sorted(configs):
            cfg = configs[name]
            g = cfg.get("convrot_groupsize") if cfg.get("convrot") else None
            print(
                f"  in={str(in_features.get(name)):>6}  "
                f"{cfg.get('format', '?'):<20} "
                f"convrot={'-' if g is None else g:<6} {name}"
            )


if __name__ == "__main__":
    main()
