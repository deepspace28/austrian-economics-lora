import os

_ROOT = os.path.dirname(os.path.abspath(__file__))

# Must precede the transformers import to have any effect.
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

import json

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

# Which adapter to test: economics_qwen (Q&A only), economics_books (prose
# style), or economics_both (prose style + Q&A).
ADAPTER_PATH = os.environ.get(
    "ADAPTER", os.path.join(_ROOT, "adapters", "economics_both")
)
CACHE_DIR = os.environ.get("HF_HOME")
TOKENIZER_FALLBACK = "Qwen/Qwen2.5-0.5B-Instruct"


def resolve_base_model():
    """Read the base model out of the adapter itself. Hardcoding it here is how
    you end up loading a 3B base under an adapter trained on 0.5B, which fails
    with an unhelpful shape mismatch."""
    override = os.environ.get("QWEN_MODEL")
    if override:
        return override
    cfg_path = os.path.join(ADAPTER_PATH, "adapter_config.json")
    if os.path.exists(cfg_path):
        with open(cfg_path, encoding="utf-8") as f:
            base = json.load(f).get("base_model_name_or_path")
        if base:
            return base
    return "Qwen/Qwen2.5-3B-Instruct"


def load_tokenizer(model_name):
    """Same validation as train.py: an incomplete tokenizer loads without
    raising but has vocab size 1 and no chat template."""
    for name in (model_name, TOKENIZER_FALLBACK):
        try:
            tok = AutoTokenizer.from_pretrained(
                name, cache_dir=CACHE_DIR, local_files_only=True
            )
        except Exception:
            continue
        if len(tok) >= 100_000 and tok.chat_template:
            if tok.pad_token is None:
                tok.pad_token = tok.eos_token
            tok.padding_side = "left"  # correct side for batched generation
            return tok
    raise SystemExit("No usable tokenizer found in the local cache.")


def load_model(base_model_name):
    """Load the base once and attach the adapter. Loading a second full copy to
    compare against does not fit in this machine's memory -- use
    `model.disable_adapter()` to get base behaviour from the same weights."""
    use_cuda = torch.cuda.is_available()
    kwargs = dict(cache_dir=CACHE_DIR, local_files_only=True, low_cpu_mem_usage=True)
    if use_cuda:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        kwargs["device_map"] = {"": 0}
    else:
        kwargs["dtype"] = torch.float32
        kwargs["device_map"] = {"": "cpu"}

    print(f"Loading {base_model_name} on {'cuda (4-bit)' if use_cuda else 'cpu'}...")
    model = AutoModelForCausalLM.from_pretrained(base_model_name, **kwargs)

    if not os.path.exists(os.path.join(ADAPTER_PATH, "adapter_config.json")):
        raise SystemExit(f"No adapter at {ADAPTER_PATH}. Run train.py first.")
    model = PeftModel.from_pretrained(model, ADAPTER_PATH)
    model.eval()
    return model


def generate_response(model, tokenizer, question, finetuned=True):
    messages = [{"role": "user", "content": question}]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    def _gen():
        with torch.no_grad():
            return model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tokenizer.pad_token_id,
            )

    if finetuned:
        output_ids = _gen()
    else:
        # Same weights, adapter switched off -- no second model in memory.
        with model.disable_adapter():
            output_ids = _gen()

    response_ids = output_ids[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(response_ids, skip_special_tokens=True)


def compare(model, tokenizer, question):
    print("\n" + "=" * 60)
    print(f"Q: {question}")
    print("=" * 60)
    print(f"[Base]:\n{generate_response(model, tokenizer, question, finetuned=False)}\n")
    print("-" * 60)
    print(f"[Finetuned]:\n{generate_response(model, tokenizer, question, finetuned=True)}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    base_model_name = resolve_base_model()
    tokenizer = load_tokenizer(base_model_name)
    model = load_model(base_model_name)

    compare(model, tokenizer,
            "What is the basic reason bad economists and demagogues seem convincing?")

    while True:
        try:
            user_input = input("Ask a question (or 'exit' to quit): ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if user_input.lower() in {"exit", "quit"}:
            break
        if user_input:
            compare(model, tokenizer, user_input)
