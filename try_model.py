"""
Side-by-side demo: base Qwen vs the fine-tuned adapter, same weights.

Tests both things we trained for:
  1. Q&A     -- does it answer in the register of the books?
  2. Prose   -- given the opening of a sentence, does it continue like the books?

    python try_model.py                 # economics_both
    set ADAPTER=...economics_books & python try_model.py
"""

import os

_ROOT = os.path.dirname(os.path.abspath(__file__))

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

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

print(f"adapter : {os.path.basename(ADAPTER)}")
print(f"base    : {base_name}\nloading...", flush=True)

model = AutoModelForCausalLM.from_pretrained(
    base_name, cache_dir=CACHE, local_files_only=True, low_cpu_mem_usage=True,
    device_map={"": 0},
    quantization_config=BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True),
)
model = PeftModel.from_pretrained(model, ADAPTER)
model.eval()


def gen(text, tuned, n=110):
    ids = tok(text, return_tensors="pt").to(model.device)
    def run():
        with torch.no_grad():
            return model.generate(**ids, max_new_tokens=n, do_sample=True,
                                  temperature=0.7, top_p=0.9,
                                  repetition_penalty=1.1,
                                  pad_token_id=tok.pad_token_id)
    if tuned:
        out = run()
    else:
        with model.disable_adapter():
            out = run()
    return tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True).strip()


QA = "Why do people so often judge an economic policy by its immediate effects alone?"
PROSE = ("The essential difference between the market economy and socialism "
         "is that")

print("\n" + "=" * 68)
print("TEST 1 - question answering")
print("=" * 68)
prompt = tok.apply_chat_template([{"role": "user", "content": QA}],
                                 tokenize=False, add_generation_prompt=True)
print(f"Q: {QA}\n")
print(f"[BASE]\n{gen(prompt, False)}\n")
print(f"[FINE-TUNED]\n{gen(prompt, True)}\n")

print("=" * 68)
print("TEST 2 - prose continuation (raw, no chat template)")
print("=" * 68)
print(f"Prompt: {PROSE}...\n")
print(f"[BASE]\n{gen(PROSE, False)}\n")
print(f"[FINE-TUNED]\n{gen(PROSE, True)}\n")
