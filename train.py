import os

_ROOT = os.path.dirname(os.path.abspath(__file__))

# NOTE: these must be set BEFORE transformers/torch are imported, otherwise the
# library has already read them at import time and they are silently ignored.
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

import glob
import json
import math
import shutil

import torch
from torch.optim import AdamW
from torch.utils.data import Dataset as TorchDataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    get_cosine_schedule_with_warmup,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

MODEL_NAME = os.environ.get("QWEN_MODEL", "Qwen/Qwen2.5-3B-Instruct")
CACHE_DIR = os.environ.get("HF_HOME")

# "chat" -> instruction/response pairs, prompt masked, chat template applied.
# "text" -> raw book prose packed into fixed blocks, every token supervised.
#           This is what teaches the model to *write like* the books; the chat
#           template and prompt masking would both be wrong for it.
MODE = os.environ.get("TRAIN_MODE", "chat").lower()

DATASET_PATH = os.path.join(_ROOT, "datasets", "training_dataset.jsonl")
BOOKS_DIR = os.path.join(_ROOT, "datasets", "books_clean")
OUTPUT_DIR = os.environ.get(
    "OUTPUT_DIR",
    os.path.join(_ROOT, "adapters", "economics_books") if MODE == "text"
    else os.path.join(_ROOT, "adapters", "economics_qwen"),
)
# Stage 2: continue training an existing adapter instead of starting fresh.
RESUME_ADAPTER = os.environ.get("RESUME_ADAPTER", "")
# Fraction of data held out to distinguish learning from memorisation.
EVAL_FRACTION = float(os.environ.get("EVAL_FRACTION", 0.02))

# Longest example in training_dataset.jsonl is 166 tokens, so 192 truncates nothing.
MAX_LENGTH = 192
# Block length for packed prose. Longer gives the model more context to learn
# sentence rhythm from, but activation memory scales with it and this GPU has
# 4 GB. Probe with MAX_STEPS=2 before raising.
BLOCK_SIZE = int(os.environ.get("BLOCK_SIZE", 384))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", 1))
GRAD_ACCUM = int(os.environ.get("GRAD_ACCUM", 16))   # effective batch size 16
EPOCHS = int(os.environ.get("EPOCHS", 2 if MODE == "text" else 3))
LEARNING_RATE = float(os.environ.get("LEARNING_RATE", 2e-4))
WARMUP_RATIO = 0.03
LOG_EVERY = int(os.environ.get("LOG_EVERY", 10))     # optimizer steps
# Cap optimizer steps for a quick smoke test, e.g. set MAX_STEPS=3
MAX_STEPS = int(os.environ.get("MAX_STEPS", 0)) or None


# --------------------------------------------------------------------------
# Preflight: this machine has 7.3 GB RAM and a 4 GB GPU. Loading a large
# safetensors shard with too little free memory kills the process with a bare
# access violation / "paging file is too small" instead of a Python error, so
# check up front and say something useful.
# --------------------------------------------------------------------------
def preflight(model_name):
    if os.path.isdir(model_name):
        shards = glob.glob(os.path.join(model_name, "*.safetensors"))
    else:
        repo = "models--" + model_name.replace("/", "--")
        shards = glob.glob(os.path.join(CACHE_DIR, repo, "snapshots", "*", "*.safetensors"))
    if not shards:
        raise SystemExit(
            f"No .safetensors found for {model_name} in {CACHE_DIR}.\n"
            f"This script runs with local_files_only=True, so the model must "
            f"already be downloaded."
        )
    largest_gb = max(os.path.getsize(p) for p in shards) / 1e9

    free_gb = None
    try:
        import ctypes

        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        stat = MEMORYSTATUSEX()
        stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
        free_gb = stat.ullAvailPhys / 1e9
        commit_gb = stat.ullAvailPageFile / 1e9
    except Exception:
        return largest_gb

    print(f"  largest shard : {largest_gb:.2f} GB")
    print(f"  free RAM      : {free_gb:.2f} GB")
    print(f"  free commit   : {commit_gb:.2f} GB")

    # Free *commit* is the binding constraint, not free physical RAM: the shard
    # is memory-mapped and pageable, but the load still has to reserve commit
    # charge for it plus a quantization workspace. Running out surfaces as
    # OSError 1455 ("paging file is too small") or a bare access violation.
    needed = largest_gb * 1.5
    if commit_gb < needed:
        smaller = [m for m in ("Qwen/Qwen2.5-1.5B-Instruct", "Qwen/Qwen2.5-0.5B-Instruct")
                   if m != model_name]
        print(
            f"\n  !! WARNING: free commit ({commit_gb:.2f} GB) is under the ~{needed:.2f} GB\n"
            f"     this load is likely to need. It may die with an access violation or\n"
            f"     'paging file is too small' -- a hard crash, not a Python traceback.\n"
            f"     Fix: close PyCharm / Chrome to free memory, or train a smaller model:\n"
            f"         set QWEN_MODEL={smaller[0]}\n"
        )
    return largest_gb


class JsonlDataset(TorchDataset):
    """Tokenizes each example and masks the prompt so loss is computed on the
    assistant's answer only."""

    def __init__(self, file_path, tokenizer, max_length=MAX_LENGTH):
        self.samples = []
        skipped = 0
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                messages = self.to_messages(json.loads(line))
                if messages is None:
                    skipped += 1
                    continue

                # Full conversation, and the prompt-only prefix that precedes
                # the assistant turn. The difference is what we train on.
                full = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=False
                )
                prompt = tokenizer.apply_chat_template(
                    messages[:-1], tokenize=False, add_generation_prompt=True
                )

                ids = tokenizer(full, truncation=True, max_length=max_length)["input_ids"]
                n_prompt = len(tokenizer(prompt, truncation=True, max_length=max_length)["input_ids"])

                labels = list(ids)
                for i in range(min(n_prompt, len(labels))):
                    labels[i] = -100  # don't train on the question

                if all(t == -100 for t in labels):
                    skipped += 1  # answer was entirely truncated away
                    continue

                self.samples.append({"input_ids": ids, "labels": labels})

        if not self.samples:
            raise SystemExit(f"No usable examples parsed from {file_path}")
        if skipped:
            print(f"  skipped {skipped} unusable example(s)")

    @staticmethod
    def to_messages(sample):
        """Dataset mixes two schemas: {'messages': [...]} and
        {'instruction', 'response'}. Handle both."""
        if "messages" in sample:
            msgs = sample["messages"]
            return msgs if msgs and msgs[-1].get("role") == "assistant" else None
        if "instruction" in sample and "response" in sample:
            return [
                {"role": "user", "content": sample["instruction"]},
                {"role": "assistant", "content": sample["response"]},
            ]
        return None

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


class PackedTextDataset(TorchDataset):
    """Concatenate the cleaned books into one token stream and slice it into
    equal blocks.

    Packing rather than padding means no wasted compute and no padding in the
    loss. Every token is supervised -- for prose continuation there is no
    prompt to mask, the objective is simply "predict the next word as this
    author would".
    """

    def __init__(self, folder, tokenizer, block_size=BLOCK_SIZE):
        paths = sorted(glob.glob(os.path.join(folder, "*.txt")))
        if not paths:
            raise SystemExit(
                f"No .txt files in {folder}. Run prepare_books.py first."
            )
        eos = tokenizer.eos_token_id
        stream = []
        for path in paths:
            with open(path, encoding="utf-8") as f:
                text = f.read()
            n_before = len(stream)
            # Tokenise paragraph by paragraph to keep peak memory bounded.
            for par in text.split("\n\n"):
                par = par.strip()
                if par:
                    stream.extend(tokenizer(par + "\n\n",
                                            add_special_tokens=False)["input_ids"])
            stream.append(eos)   # book boundary
            print(f"    {os.path.basename(path):<44} "
                  f"{len(stream)-n_before:>9,} tokens", flush=True)

        n_blocks = len(stream) // block_size
        if n_blocks == 0:
            raise SystemExit(f"Corpus is smaller than one {block_size}-token block.")
        self.blocks = [stream[i * block_size:(i + 1) * block_size]
                       for i in range(n_blocks)]
        self.total_tokens = len(stream)

    def __len__(self):
        return len(self.blocks)

    def __getitem__(self, idx):
        ids = self.blocks[idx]
        return {"input_ids": ids, "labels": list(ids)}


def make_collator(pad_token_id):
    """Pad to the longest example in the batch rather than to MAX_LENGTH.
    Padded positions get label -100 so they contribute nothing to the loss."""

    def collate(batch):
        width = max(len(b["input_ids"]) for b in batch)
        input_ids, labels, attention = [], [], []
        for b in batch:
            pad = width - len(b["input_ids"])
            input_ids.append(b["input_ids"] + [pad_token_id] * pad)
            labels.append(b["labels"] + [-100] * pad)
            attention.append([1] * len(b["input_ids"]) + [0] * pad)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attention, dtype=torch.long),
        }

    return collate


print("Preflight:")
preflight(MODEL_NAME)

use_cuda = torch.cuda.is_available()
device = "cuda" if use_cuda else "cpu"
print(f"\nDevice: {device}" + (f" ({torch.cuda.get_device_name(0)})" if use_cuda else ""))

# ---------------------------------------------------------------------------
# Tokenizer. The cached 3B snapshot ships weights only (no tokenizer.json), so
# fall back to the 0.5B tokenizer files. This is safe: every Qwen2.5-Instruct
# model shares one tokenizer and vocab_size=151936 (verified against both
# config.json files).
# ---------------------------------------------------------------------------
TOKENIZER_FALLBACK = "Qwen/Qwen2.5-0.5B-Instruct"


def load_tokenizer():
    """A missing vocab does NOT raise -- transformers happily returns a
    Qwen2Tokenizer with vocab size 1 and no chat template, which would tokenize
    the whole dataset into garbage. So validate rather than catch."""
    for name in (MODEL_NAME, TOKENIZER_FALLBACK):
        try:
            tok = AutoTokenizer.from_pretrained(
                name, cache_dir=CACHE_DIR, local_files_only=True
            )
        except Exception as exc:
            print(f"  {name}: failed to load ({type(exc).__name__})")
            continue
        if len(tok) < 100_000 or not tok.chat_template:
            print(f"  {name}: incomplete (vocab={len(tok)}, "
                  f"chat_template={'yes' if tok.chat_template else 'no'}) -- skipping")
            continue
        if name != MODEL_NAME:
            print(f"Tokenizer: {name} (shared Qwen2.5 vocab; "
                  f"{MODEL_NAME} ships weights only)")
        else:
            print(f"Tokenizer: {name}")
        return tok
    raise SystemExit("No usable tokenizer found in the local cache.")


tokenizer = load_tokenizer()

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

print(f"\nLoading {MODEL_NAME}...")
load_kwargs = dict(
    cache_dir=CACHE_DIR,
    local_files_only=True,
    low_cpu_mem_usage=True,
)
if use_cuda:
    # This is what the original bnb_config was for -- it was built but never
    # passed, so the model silently loaded in fp32 on the CPU.
    load_kwargs["quantization_config"] = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    load_kwargs["device_map"] = {"": 0}
else:
    load_kwargs["dtype"] = torch.float32
    load_kwargs["device_map"] = {"": "cpu"}

model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, **load_kwargs)

if use_cuda:
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
model.config.use_cache = False          # incompatible with gradient checkpointing
model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
model.enable_input_require_grads()

if RESUME_ADAPTER:
    # Stage 2 continues the stage-1 adapter so the prose style learned from the
    # books is carried into instruction tuning rather than discarded.
    from peft import PeftModel

    if not os.path.exists(os.path.join(RESUME_ADAPTER, "adapter_config.json")):
        raise SystemExit(f"No adapter at {RESUME_ADAPTER}")
    print(f"Resuming adapter from {RESUME_ADAPTER}...")
    model = PeftModel.from_pretrained(model, RESUME_ADAPTER, is_trainable=True)
    model.print_trainable_parameters()
else:
    print("Applying LoRA...")
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        # Attention + MLP projections. The original targeted only q_proj/v_proj,
        # which leaves most of the model untouched.
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

print(f"\nLoading and tokenizing dataset (mode={MODE})...")
if MODE == "text":
    dataset = PackedTextDataset(BOOKS_DIR, tokenizer)
    print(f"  {dataset.total_tokens:,} tokens -> {len(dataset)} blocks "
          f"of {BLOCK_SIZE}")
else:
    dataset = JsonlDataset(DATASET_PATH, tokenizer)
    print(f"  {len(dataset)} examples")

# Hold out a slice so we can tell genuine learning from memorisation. Training
# loss alone cannot distinguish them, especially over multiple epochs.
generator = torch.Generator().manual_seed(42)
n_eval = max(1, int(len(dataset) * EVAL_FRACTION)) if EVAL_FRACTION > 0 else 0
n_train = len(dataset) - n_eval
train_set, eval_set = torch.utils.data.random_split(
    dataset, [n_train, n_eval], generator=generator
)
collate = make_collator(tokenizer.pad_token_id)
dataloader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True,
                        collate_fn=collate)
eval_loader = DataLoader(eval_set, batch_size=BATCH_SIZE, shuffle=False,
                         collate_fn=collate)
print(f"  split: {n_train} train / {n_eval} held out")


@torch.no_grad()
def evaluate():
    model.eval()
    total, count = 0.0, 0
    for batch in eval_loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        total += model(**batch).loss.item()
        count += 1
    model.train()
    mean = total / max(count, 1)
    return mean, math.exp(min(mean, 20))

steps_per_epoch = math.ceil(len(dataloader) / GRAD_ACCUM)
total_steps = steps_per_epoch * EPOCHS
if MAX_STEPS:
    total_steps = min(total_steps, MAX_STEPS)

# Only the LoRA adapters are trainable; handing the frozen base weights to the
# optimizer wastes memory and is what the original code did.
trainable = [p for p in model.parameters() if p.requires_grad]
try:
    from bitsandbytes.optim import PagedAdamW8bit

    optimizer = PagedAdamW8bit(trainable, lr=LEARNING_RATE)
    print("Optimizer: PagedAdamW8bit")
except Exception:
    optimizer = AdamW(trainable, lr=LEARNING_RATE)
    print("Optimizer: AdamW")

scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=max(1, int(total_steps * WARMUP_RATIO)),
    num_training_steps=total_steps,
)

print(f"\nTraining: {EPOCHS} epochs x {steps_per_epoch} steps = {total_steps} optimizer steps")
print(f"(effective batch {BATCH_SIZE * GRAD_ACCUM}, seq len <= {MAX_LENGTH})\n")

model.train()
step = 0
running = 0.0
counted = 0

for epoch in range(EPOCHS):
    optimizer.zero_grad(set_to_none=True)
    for i, batch in enumerate(dataloader):
        batch = {k: v.to(device) for k, v in batch.items()}

        loss = model(**batch).loss
        running += loss.item()
        counted += 1
        (loss / GRAD_ACCUM).backward()

        is_last = (i + 1) == len(dataloader)
        if (i + 1) % GRAD_ACCUM == 0 or is_last:
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1

            if step % LOG_EVERY == 0 or step == total_steps:
                vram = f" | vram {torch.cuda.memory_allocated()/1e9:.2f}GB" if use_cuda else ""
                print(
                    f"epoch {epoch+1}/{EPOCHS} | step {step}/{total_steps} "
                    f"| loss {running/max(counted,1):.4f} "
                    f"| lr {scheduler.get_last_lr()[0]:.2e}{vram}"
                )
                running, counted = 0.0, 0

            if MAX_STEPS and step >= MAX_STEPS:
                break

    # Save at every epoch boundary. A crash at step 190 of 201 should not throw
    # away the whole run.
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    model.save_pretrained(OUTPUT_DIR)
    if n_eval:
        loss, ppl = evaluate()
        print(f"-- epoch {epoch+1} done | eval loss {loss:.4f} "
              f"| perplexity {ppl:.2f} | checkpoint saved", flush=True)
    else:
        print(f"-- epoch {epoch+1} done | checkpoint saved", flush=True)

    if MAX_STEPS and step >= MAX_STEPS:
        print(f"Stopping early at MAX_STEPS={MAX_STEPS}")
        break

print(f"\nSaving adapter to {OUTPUT_DIR}...")
os.makedirs(OUTPUT_DIR, exist_ok=True)
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print("\n==============================")
print(f"Training complete. Adapter saved to {OUTPUT_DIR}")
print("==============================\n")
