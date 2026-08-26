"""
Publish the trained weights to the Hugging Face Hub.

    python publish_to_hf.py --user YOUR_HF_USERNAME              # adapters (345 MB)
    python publish_to_hf.py --user YOUR_HF_USERNAME --gguf       # + Q4_K_M (1.9 GB)
    python publish_to_hf.py --user YOUR_HF_USERNAME --dry-run    # show, upload nothing

Authenticate first — a WRITE token from https://huggingface.co/settings/tokens:

    huggingface-cli login

Uploads resume, so a dropped connection is not fatal: rerun the same command.

Licence note: the adapters derive from Qwen2.5-3B-Instruct, which ships under the
Qwen RESEARCH LICENSE (non-commercial). This script fetches that licence and uploads
it alongside the weights, which the agreement requires when distributing derivatives.
"""

import argparse
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
GGUF_DIR = Path("D:/Models/gguf")

QWEN_LICENSE_URL = (
    "https://huggingface.co/Qwen/Qwen2.5-3B-Instruct/raw/main/LICENSE"
)

# The probe run was a 2-step smoke test - no reason to publish it.
ADAPTERS = ["economics_both", "economics_books", "economics_qwen"]

# f16 is 6.2 GB and reconstructible from Q4_K_M's source; not worth the upload.
GGUF_FILES = ["qwen3b-economics-Q4_K_M.gguf"]


def human(n):
    return f"{n/1e9:.2f} GB" if n >= 1e9 else f"{n/1e6:.0f} MB"


def fetch_license(dest):
    if dest.exists():
        return dest
    print(f"  fetching Qwen RESEARCH LICENSE -> {dest.name}")
    with urllib.request.urlopen(QWEN_LICENSE_URL, timeout=60) as r:
        dest.write_bytes(r.read())
    return dest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--user", required=True, help="your Hugging Face username")
    ap.add_argument("--gguf", action="store_true", help="also publish the GGUF quant")
    ap.add_argument("--private", action="store_true", help="create private repos")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    adapter_repo = f"{args.user}/qwen2.5-3b-austrian-economics-lora"
    gguf_repo = f"{args.user}/qwen2.5-3b-austrian-economics-GGUF"

    # ---- collect what we are about to send -------------------------------
    payload = []
    total = 0
    for name in ADAPTERS:
        d = ROOT / "adapters" / name
        if not d.is_dir():
            sys.exit(f"missing adapter: {d}")
        for f in sorted(d.iterdir()):
            if f.name == "README.md":       # empty HF boilerplate
                continue
            size = f.stat().st_size
            payload.append((f, f"{name}/{f.name}", size))
            total += size

    gguf_payload = []
    if args.gguf:
        for fn in GGUF_FILES:
            f = GGUF_DIR / fn
            if not f.exists():
                sys.exit(f"missing GGUF: {f}")
            gguf_payload.append((f, fn, f.stat().st_size))

    print(f"\nAdapter repo: {adapter_repo}")
    for _, path, size in payload:
        print(f"    {path:<45} {human(size):>9}")
    print(f"    {'':<45} {human(total):>9} total")

    if gguf_payload:
        gtotal = sum(s for _, _, s in gguf_payload)
        print(f"\nGGUF repo: {gguf_repo}")
        for _, path, size in gguf_payload:
            print(f"    {path:<45} {human(size):>9}")
        print(f"    {'':<45} {human(gtotal):>9} total")

    if args.dry_run:
        print("\n--dry-run: nothing uploaded.")
        return 0

    from huggingface_hub import HfApi
    api = HfApi()
    try:
        who = api.whoami()["name"]
    except Exception:
        sys.exit("Not logged in. Run:  huggingface-cli login\n"
                 "Create a WRITE token at https://huggingface.co/settings/tokens")
    print(f"\nauthenticated as {who}")
    if who != args.user:
        print(f"  ! --user is {args.user} but you are {who}; using {args.user} "
              f"(fine if it is an org you can write to)")

    card = ROOT / "MODEL_CARD.md"
    lic = fetch_license(ROOT / "LICENSE.qwen")

    # ---- adapters --------------------------------------------------------
    print(f"\ncreating {adapter_repo}")
    api.create_repo(adapter_repo, repo_type="model",
                    private=args.private, exist_ok=True)

    readme = card.read_text(encoding="utf-8").replace("REPO_ID", adapter_repo)
    api.upload_file(path_or_fileobj=readme.encode("utf-8"),
                    path_in_repo="README.md", repo_id=adapter_repo)
    api.upload_file(path_or_fileobj=str(lic),
                    path_in_repo="LICENSE", repo_id=adapter_repo)

    for f, path_in_repo, size in payload:
        print(f"  uploading {path_in_repo} ({human(size)})")
        api.upload_file(path_or_fileobj=str(f), path_in_repo=path_in_repo,
                        repo_id=adapter_repo)
    print(f"  done -> https://huggingface.co/{adapter_repo}")

    # ---- gguf ------------------------------------------------------------
    if gguf_payload:
        print(f"\ncreating {gguf_repo}")
        api.create_repo(gguf_repo, repo_type="model",
                        private=args.private, exist_ok=True)
        gcard = (f"---\nbase_model: Qwen/Qwen2.5-3B-Instruct\n"
                 f"license: other\nlicense_name: qwen-research\n"
                 f"license_link: {QWEN_LICENSE_URL}\ntags:\n  - gguf\n"
                 f"  - llama.cpp\n  - economics\n---\n\n"
                 f"# Qwen2.5-3B Austrian Economics — GGUF\n\n"
                 f"**Built with Qwen.** Q4_K_M quantization of the merged "
                 f"`economics_both` adapter.\n\n"
                 f"Adapters: https://huggingface.co/{adapter_repo}\n"
                 f"Code: https://github.com/deepspace28/austrian-economics-lora\n\n"
                 f"Non-commercial use only (Qwen RESEARCH LICENSE). See the model "
                 f"card on the adapter repo for training details and limitations.\n")
        api.upload_file(path_or_fileobj=gcard.encode("utf-8"),
                        path_in_repo="README.md", repo_id=gguf_repo)
        api.upload_file(path_or_fileobj=str(lic),
                        path_in_repo="LICENSE", repo_id=gguf_repo)
        for f, path_in_repo, size in gguf_payload:
            print(f"  uploading {path_in_repo} ({human(size)}) - this takes a while")
            api.upload_file(path_or_fileobj=str(f), path_in_repo=path_in_repo,
                            repo_id=gguf_repo)
        print(f"  done -> https://huggingface.co/{gguf_repo}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
