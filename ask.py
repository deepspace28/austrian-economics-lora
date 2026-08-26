"""
Run a single prompt through the fine-tuned model.

    python ask.py "your question here"           # chat mode (Q&A)
    python ask.py --raw "Prices are signals that" # raw continuation (book style)
    python ask.py --base "..."                   # also show the base model

    set ADAPTER=adapters\economics_books & python ask.py "..."
"""

import os
import sys

_ROOT = os.path.dirname(os.path.abspath(__file__))

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

args = sys.argv[1:]
raw = "--raw" in args
show_base = "--base" in args
tokens = 200
for a in list(args):
    if a.startswith("--tokens="):
        tokens = int(a.split("=", 1)[1])
        args.remove(a)
prompt = " ".join(a for a in args if not a.startswith("--")).strip()
if not prompt:
    raise SystemExit('usage: python ask.py "your prompt"   [--raw] [--base] [--tokens=N]')

ADAPTER = os.environ.get("ADAPTER", os.path.join(_ROOT, "adapters", "economics_both"))
CACHE = os.environ.get("HF_HOME")

with open(os.path.join(ADAPTER, "adapter_config.json"), encoding="utf-8") as f:
    base_name = json.load(f)["base_model_name_or_path"]

tok = AutoTokenizer.from_pretrained(base_name, cache_dir=CACHE, local_files_only=True)
if len(tok) < 100_000 or not tok.chat_template:
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct",
                                        cache_dir=CACHE, local_files_only=True)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

print(f"adapter: {os.path.basename(ADAPTER)} | mode: "
      f"{'raw continuation' if raw else 'chat'} | loading...", flush=True)

model = AutoModelForCausalLM.from_pretrained(
    base_name, cache_dir=CACHE, local_files_only=True, low_cpu_mem_usage=True,
    device_map={"": 0},
    quantization_config=BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True),
)
model = PeftModel.from_pretrained(model, ADAPTER)
model.eval()

text = prompt if raw else tok.apply_chat_template(
    [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)


def gen(tuned):
    ids = tok(text, return_tensors="pt").to(model.device)

    def run():
        with torch.no_grad():
            return model.generate(**ids, max_new_tokens=tokens, do_sample=True,
                                  temperature=0.7, top_p=0.9,
                                  repetition_penalty=1.1,
                                  pad_token_id=tok.pad_token_id)

    out = run() if tuned else None
    if out is None:
        with model.disable_adapter():
            out = run()
    return tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True).strip()


print("\n" + "=" * 68)
print(f"PROMPT: {prompt}")
print("=" * 68)
if show_base:
    print(f"\n[BASE]\n{gen(False)}")
print(f"\n[FINE-TUNED]\n{gen(True)}\n")
