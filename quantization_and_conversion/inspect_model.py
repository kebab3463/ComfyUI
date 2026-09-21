from argparse import ArgumentParser
import safetensors
import json
import struct
import os
import re
from collections import Counter


def inspect_model_safe(file_path):
    with safetensors.safe_open(file_path, framework="pt", device="cpu") as f:
        meta = f.metadata() or {}
        keys = [k for k in f.keys()]
        print(f"metadata: {meta}")
        print("\n== top-level namespaces ==\n")
        for ns,c in Counter(k.split('.',1)[0] for k in keys).most_common():
            print(f"  {ns:30s} {c}\n")
        print("\n== full key list (name | dtype | shape) ==\n")
        for k in sorted(keys):
            print(f"{k}\t{f.get_tensor(k).dtype}\t{f.get_tensor(k).shape}\n")

def inspect_model(file_path):
    if not os.path.exists(file_path): print("NOT FOUND", file_path); raise SystemExit
    fsize=os.path.getsize(file_path)
    with open(file_path,"rb") as f:
        n=struct.unpack("<Q",f.read(8))[0]; hdr=json.loads(f.read(n))
    meta=hdr.pop("__metadata__",{}) or {}
    print(f"size {fsize/1e9:.2f} GB ; tensors {len(hdr)} ; meta keys {list(meta)[:10]}")
    print("dtypes:", dict(Counter(v['dtype'] for v in hdr.values())))
    print("already-quant markers:", sum(1 for k in hdr if k.endswith((".weight_scale",".comfy_quant",".scale_weight"))))
    print("top-level namespaces:", dict(Counter(k.split('.',1)[0] for k in hdr)))
    norm=lambda k: re.sub(r"\.\d+\.",".N.",k)
    w2d=[(k,v["shape"]) for k,v in hdr.items() if k.endswith(".weight") and len(v["shape"])==2]
    big=[(k,s) for k,s in w2d if min(s)>=1024]
    print(f"\n2D .weight: {len(w2d)} ; >=1024 both dims: {len(big)}")
    print("\n== distinct 2D>=1024 .weight patterns (quant candidates) ==")
    for nm,c in sorted(Counter(norm(k) for k,_ in big).items()):
        ex=next(s for k,s in big if norm(k)==nm)
        pad32 = "" if (ex[0]%32==0 and ex[1]%32==0) else " !PAD(not 32-aligned)"
        print(f"  x{c:<4} {nm:62s} {ex}{pad32}")
    print("\n== 2D<1024 .weight (kept by min-dim) ==")
    for nm,c in sorted(Counter(norm(k) for k,_ in w2d if min(_ if False else next(s for kk,s in w2d if kk==k))<1024 for _ in [0]).items())[:0]:
        pass
    small=[(k,s) for k,s in w2d if min(s)<1024]
    for nm,c in sorted(Counter(norm(k) for k,_ in small).items()):
        ex=next(s for k,s in small if norm(k)==nm)
        print(f"  x{c:<4} {nm:62s} {ex}")
    print("\n== possible embedding tables (vocab-sized dim >=50000) ==")
    for k,s in w2d:
        if max(s)>=50000:
            print(f"  {norm(k):62s} {s}")
    print("\n== sensitive-name candidates (embed/mod/adaln/time/norm/img_in/txt_in/final/patch/proj) ==")
    seen=set()
    for k,v in hdr.items():
        nm=norm(k)
        if nm in seen: continue
        if any(t in k.lower() for t in ("embed","modul","adaln","ada_ln","time","img_in","txt_in","final","patch","norm","guidance","cond","_mod.","register","scale_shift")):
            seen.add(nm)
            print(f"  {nm:62s} {v['dtype']} {v['shape']}")

def main():
    parser = ArgumentParser()
    parser.add_argument("input", type=str, help="Input safetensors file")
    args = parser.parse_args()
    inspect_model(args.input)

if __name__ == "__main__":
    main()
