import json, struct, os
p = "/home/csprunger/models/diffusion_models/Sulphur2/sulphur_distil_bf16.safetensors"
fsize = os.path.getsize(p)
with open(p, "rb") as f:
    n = struct.unpack("<Q", f.read(8))[0]
    if n + 8 > fsize:
        raise SystemExit(f"header ({n} bytes) not downloaded yet (file is {fsize} bytes) - wait a bit")
    hdr = json.loads(f.read(n))
meta = hdr.pop("__metadata__", None)
with open("inspect_ltx.txt", "w") as f:
    f.write(f"file so far: {fsize/1e9:.1f} GB ; header: {n/1e6:.2f} MB ; tensors: {len(hdr)}")
    f.write(f"metadata: {meta}")
    from collections import Counter
    f.write("\n== top-level namespaces ==\n")
    for ns,c in Counter(k.split('.',1)[0] for k in hdr).most_common():
        f.write(f"  {ns:30s} {c}\n")
    f.write("\n== full key list (name | dtype | shape) ==\n")
    for k in sorted(hdr):
        f.write(f"{k}\t{hdr[k]['dtype']}\t{hdr[k]['shape']}\n")

