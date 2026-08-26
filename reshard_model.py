"""
One-time utility: rewrite a cached HF model into small safetensors shards.

Why: this machine's commit limit is pinned (C: is nearly full, so the pagefile
cannot grow). Loading Qwen2.5-3B fails with a bare access violation / OSError
1455 because its largest shard is 3.97 GB. Empirically the crash tracks shard
size -- a 0.99 GB shard loads fine, 3.09 GB and 3.97 GB do not.

safetensors' own reader memory-maps the entire file, which is exactly what
fails here, so this parses the container directly with ordinary buffered reads
and never holds more than one tensor plus one output group in memory.

Run once:   python reshard_model.py
Then:       set QWEN_MODEL=models\Qwen2.5-3B-Instruct-resharded
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
from safetensors.torch import save_file

CACHE_DIR = os.environ.get("HF_HOME")
# Optional arg, e.g.  python reshard_model.py Qwen/Qwen2.5-1.5B-Instruct
SRC_MODEL = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen2.5-3B-Instruct"
# Tokenizer files live with the 0.5B snapshot; the 3B snapshot is weights-only.
TOKENIZER_SRC = "Qwen/Qwen2.5-0.5B-Instruct"
DST_DIR = os.path.join(_ROOT, "models", SRC_MODEL.split("/")[-1] + "-resharded")
SHARD_BYTES = 250 * 1024 * 1024

DTYPES = {
    "BF16": torch.bfloat16, "F16": torch.float16, "F32": torch.float32,
    "F64": torch.float64, "I8": torch.int8, "U8": torch.uint8,
    "I16": torch.int16, "I32": torch.int32, "I64": torch.int64,
    "BOOL": torch.bool,
}


def read_header(fh):
    """safetensors layout: u64 header length, then that many bytes of JSON,
    then the raw tensor data."""
    n = struct.unpack("<Q", fh.read(8))[0]
    header = json.loads(fh.read(n))
    return header, 8 + n


def iter_tensors(path):
    """Yield (name, tensor) using ordinary reads -- no mmap."""
    with open(path, "rb") as fh:
        header, data_start = read_header(fh)
        entries = [(k, v) for k, v in header.items() if k != "__metadata__"]
        # Read in on-disk order so the file is walked sequentially.
        entries.sort(key=lambda kv: kv[1]["data_offsets"][0])
        for name, meta in entries:
            start, end = meta["data_offsets"]
            fh.seek(data_start + start)
            raw = bytearray(end - start)          # writable -> no torch warning
            if fh.readinto(raw) != len(raw):
                raise SystemExit(f"short read for {name} in {path}")
            dtype = DTYPES[meta["dtype"]]
            t = torch.frombuffer(raw, dtype=dtype)
            yield name, t.reshape(meta["shape"]).clone()


def snapshot_dir(model_name):
    repo = "models--" + model_name.replace("/", "--")
    hits = sorted(glob.glob(os.path.join(CACHE_DIR, repo, "snapshots", "*")))
    if not hits:
        raise SystemExit(f"{model_name} not found in {CACHE_DIR}")
    return hits[-1]


def main():
    src = snapshot_dir(SRC_MODEL)
    tok_src = snapshot_dir(TOKENIZER_SRC)
    os.makedirs(DST_DIR, exist_ok=True)

    shards = sorted(glob.glob(os.path.join(src, "*.safetensors")))
    print(f"source: {src}", flush=True)
    print(f"  {len(shards)} shard(s), largest "
          f"{max(os.path.getsize(p) for p in shards)/1e9:.2f} GB", flush=True)

    weight_map = {}
    total_size = 0
    state = {"group": {}, "bytes": 0, "index": 0}

    def flush():
        if not state["group"]:
            return
        state["index"] += 1
        name = f"model-{state['index']:05d}.safetensors"
        save_file(state["group"], os.path.join(DST_DIR, name),
                  metadata={"format": "pt"})
        for k in state["group"]:
            weight_map[k] = name
        print(f"  wrote {name}  ({state['bytes']/1e6:.0f} MB, "
              f"{len(state['group'])} tensors)", flush=True)
        state["group"], state["bytes"] = {}, 0

    for path in shards:                       # one source file open at a time
        print(f"  reading {os.path.basename(path)} "
              f"({os.path.getsize(path)/1e9:.2f} GB)...", flush=True)
        for name, tensor in iter_tensors(path):
            nbytes = tensor.numel() * tensor.element_size()
            total_size += nbytes
            if state["bytes"] + nbytes > SHARD_BYTES:
                flush()
            state["group"][name] = tensor
            state["bytes"] += nbytes
    flush()

    with open(os.path.join(DST_DIR, "model.safetensors.index.json"), "w") as f:
        json.dump({"metadata": {"total_size": total_size},
                   "weight_map": weight_map}, f, indent=2)

    # config from the model; tokenizer from the 0.5B snapshot (shared vocab).
    for fn in ("config.json", "generation_config.json"):
        p = os.path.join(src, fn)
        if os.path.exists(p):
            shutil.copy2(p, DST_DIR)
    copied = []
    for fn in ("tokenizer.json", "tokenizer_config.json", "vocab.json",
               "merges.txt", "chat_template.jinja"):
        p = os.path.join(tok_src, fn)
        if os.path.exists(p):
            shutil.copy2(p, DST_DIR)
            copied.append(fn)
    print(f"  copied tokenizer files: {', '.join(copied)}", flush=True)

    new = glob.glob(os.path.join(DST_DIR, "*.safetensors"))
    print(f"\nDone -> {DST_DIR}")
    print(f"  {len(new)} shards, largest "
          f"{max(os.path.getsize(p) for p in new)/1e6:.0f} MB")
    print(f"  total {total_size/1e9:.2f} GB")


if __name__ == "__main__":
    main()
