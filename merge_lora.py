"""
Merge a LoRA adapter into the base weights, producing a standalone HF model
that can be converted to GGUF for LM Studio.

peft's merge_and_unload() materialises the whole 3B model in fp16 (~6.2 GB).
This machine has ~7 GB of RAM total and under 1 GB free, so that is not an
option. LoRA only touches specific projection matrices, and each merge is
local to one tensor:

    W_merged = W_base + (B @ A) * (alpha / r)

so we stream the base one tensor at a time, merge where an adapter pair
exists, and write out. Peak memory is one tensor, not one model.

    python merge_lora.py [adapter_dir] [out_dir]
"""

import glob
import json
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = _HERE
import shutil
import struct
import sys

import torch
from safetensors.torch import load_file, save_file

ADAPTER = sys.argv[1] if len(sys.argv) > 1 else os.path.join(_ROOT, "adapters", "economics_both")
OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.join(_ROOT, "models", "qwen3b-economics-merged")
SHARD_BYTES = 900 * 1024 * 1024

DTYPES = {
    "BF16": torch.bfloat16, "F16": torch.float16, "F32": torch.float32,
    "F64": torch.float64, "I8": torch.int8, "U8": torch.uint8,
    "I16": torch.int16, "I32": torch.int32, "I64": torch.int64, "BOOL": torch.bool,
}


def read_header(fh):
    n = struct.unpack("<Q", fh.read(8))[0]
    return json.loads(fh.read(n)), 8 + n


def iter_tensors(path):
    """Stream tensors with ordinary reads -- no mmap, bounded memory."""
    with open(path, "rb") as fh:
        header, data_start = read_header(fh)
        entries = [(k, v) for k, v in header.items() if k != "__metadata__"]
        entries.sort(key=lambda kv: kv[1]["data_offsets"][0])
        for name, meta in entries:
            start, end = meta["data_offsets"]
            fh.seek(data_start + start)
            raw = bytearray(end - start)
            if fh.readinto(raw) != len(raw):
                raise SystemExit(f"short read: {name}")
            t = torch.frombuffer(raw, dtype=DTYPES[meta["dtype"]])
            yield name, t.reshape(meta["shape"]).clone()


def main():
    cfg_path = os.path.join(ADAPTER, "adapter_config.json")
    with open(cfg_path, encoding="utf-8") as f:
        cfg = json.load(f)
    base_dir = cfg["base_model_name_or_path"]
    scaling = cfg["lora_alpha"] / cfg["r"]
    print(f"adapter : {ADAPTER}")
    print(f"base    : {base_dir}")
    print(f"r={cfg['r']} alpha={cfg['lora_alpha']} -> scaling={scaling}")

    if not os.path.isdir(base_dir):
        raise SystemExit(f"base model dir not found: {base_dir}")

    lora = load_file(os.path.join(ADAPTER, "adapter_model.safetensors"))

    # peft names look like:
    #   base_model.model.<module path>.lora_A.weight
    # the corresponding base tensor is  <module path>.weight
    pairs = {}
    for key in lora:
        if ".lora_A" not in key:
            continue
        module = key.split(".lora_A")[0]
        if module.startswith("base_model.model."):
            module = module[len("base_model.model."):]
        b_key = key.replace(".lora_A", ".lora_B")
        if b_key in lora:
            pairs[module + ".weight"] = (key, b_key)
    print(f"adapter modules to merge: {len(pairs)}")

    os.makedirs(OUT, exist_ok=True)
    for old in glob.glob(os.path.join(OUT, "*.safetensors")):
        os.remove(old)

    shards = sorted(glob.glob(os.path.join(base_dir, "*.safetensors")))
    weight_map, total_size, merged_count = {}, 0, 0
    group, group_bytes, index = {}, 0, 0

    def flush():
        nonlocal group, group_bytes, index
        if not group:
            return
        index += 1
        name = f"model-{index:05d}.safetensors"
        save_file(group, os.path.join(OUT, name), metadata={"format": "pt"})
        for k in group:
            weight_map[k] = name
        print(f"  wrote {name} ({group_bytes/1e6:.0f} MB, {len(group)} tensors)", flush=True)
        group, group_bytes = {}, 0

    for path in shards:
        for name, tensor in iter_tensors(path):
            if name in pairs:
                a_key, b_key = pairs[name]
                A = lora[a_key].float()          # [r, in]
                B = lora[b_key].float()          # [out, r]
                delta = (B @ A) * scaling        # [out, in]
                if delta.shape != tensor.shape:
                    raise SystemExit(
                        f"shape mismatch on {name}: {tuple(delta.shape)} vs {tuple(tensor.shape)}")
                tensor = (tensor.float() + delta).to(tensor.dtype)
                merged_count += 1
                del A, B, delta
            nbytes = tensor.numel() * tensor.element_size()
            total_size += nbytes
            if group_bytes + nbytes > SHARD_BYTES:
                flush()
            group[name] = tensor
            group_bytes += nbytes
    flush()

    if merged_count != len(pairs):
        print(f"  !! WARNING: merged {merged_count} of {len(pairs)} adapter modules "
              f"-- some adapter keys found no matching base tensor")
    else:
        print(f"  merged all {merged_count} modules")

    with open(os.path.join(OUT, "model.safetensors.index.json"), "w") as f:
        json.dump({"metadata": {"total_size": total_size},
                   "weight_map": weight_map}, f, indent=2)

    for fn in ("config.json", "generation_config.json", "tokenizer.json",
               "tokenizer_config.json", "vocab.json", "merges.txt",
               "chat_template.jinja", "special_tokens_map.json"):
        src = os.path.join(base_dir, fn)
        if os.path.exists(src):
            shutil.copy2(src, OUT)
    # tokenizer may live with the adapter instead
    for fn in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja"):
        dst = os.path.join(OUT, fn)
        src = os.path.join(ADAPTER, fn)
        if not os.path.exists(dst) and os.path.exists(src):
            shutil.copy2(src, OUT)

    print(f"\nmerged model -> {OUT}")
    print(f"  {index} shards, {total_size/1e9:.2f} GB")


if __name__ == "__main__":
    main()
