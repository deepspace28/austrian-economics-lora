---
base_model: Qwen/Qwen2.5-3B-Instruct
license: other
license_name: qwen-research
license_link: https://huggingface.co/Qwen/Qwen2.5-3B-Instruct/blob/main/LICENSE
library_name: peft
pipeline_tag: text-generation
tags:
  - lora
  - qlora
  - peft
  - economics
  - austrian-economics
  - qwen2.5
language:
  - en
---

# Qwen2.5-3B Austrian Economics LoRA

**Built with Qwen.**

LoRA adapters for **Qwen2.5-3B-Instruct**, fine-tuned on the Austrian school canon —
Mises, Rothbard, Hazlitt, Hoppe, Bagus. Trained end to end on a laptop with 7.3 GB of
system RAM and a 4 GB GPU, peaking at **2.85 GB of VRAM**.

Pipeline, training code, and reproduction instructions:
**https://github.com/deepspace28/austrian-economics-lora**

## Adapters

| Adapter | Training | Use it for |
|---|---|---|
| `economics_both` | prose → then Q&A | **Default.** Best of both. |
| `economics_books` | raw prose only | Continuation in the style of the texts |
| `economics_qwen` | Q&A only, from base | Instruction following, no prose transfer |

All are r=16, α=32, dropout 0.05, targeting all seven projection modules —
**29,933,568 trainable parameters, 0.96%** of the 3.09B base.

## Usage

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

base = "Qwen/Qwen2.5-3B-Instruct"
tok = AutoTokenizer.from_pretrained(base)
model = AutoModelForCausalLM.from_pretrained(base, device_map="auto")
model = PeftModel.from_pretrained(model, "REPO_ID", subfolder="economics_both")

msgs = [{"role": "user",
         "content": "Why does credit expansion cause a boom that must end in a bust?"}]
ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt")
print(tok.decode(model.generate(ids.to(model.device), max_new_tokens=200)[0]))
```

## Training

Two stages, teaching different things.

**Stage 1 — continued pretraining on raw prose.** Book text packed into fixed 384-token
blocks, every token supervised. This teaches the model to *write like* the sources; a
chat template and prompt masking would both be wrong for it.

| | |
|---|---|
| Corpus | 1,664,612 tokens → 4,334 blocks |
| Split | 4,248 train / 86 held out |
| Eval loss (epoch 1) | 2.4918 — perplexity **12.08** |

**Stage 2 — instruction tuning**, resuming from the stage 1 adapter. Prompt tokens
masked, chat template applied.

| | |
|---|---|
| Examples | 1,057 pairs (1,036 train / 21 held out) |
| Train loss | 2.5196 → **1.8040** |
| Eval loss | 1.9944 — perplexity **7.35** |
| Peak VRAM | 2.85 GB |

**Hyperparameters:** 4-bit NF4 with double quant and fp16 compute, PagedAdamW8bit,
LR 2e-4 cosine with 3% warmup, effective batch 16 (batch 1 × grad accum 16), gradient
checkpointing (non-reentrant).

## Limitations

Read these before trusting anything it says.

- **Stage 1 crashed in epoch 2** with a CUDA illegal memory access. The epoch 1
  checkpoint is what stage 2 resumed from, so these adapters reflect one epoch of
  prose training, not the intended two.
- **1,057 instruction pairs is a small dataset.** The model picks up the register and
  vocabulary of the sources convincingly. It is **not** a reliable authority on what
  Mises or Rothbard actually argued, and it will state invented claims in their voice.
- **No benchmark evaluation.** Held-out loss and perplexity is all that was measured.
  No MMLU, no factuality testing, no human eval.
- **It argues one position.** The corpus is a single school of economics, so the output
  is that school stated confidently — not a survey of the field, and not a neutral
  source on contested questions.
- A 3B model quantized to 4 bits. Reasoning is limited regardless of fine-tuning.

## Training data

Seven texts: Mises (*Human Action*, *Interventionism*), Rothbard (*Man, Economy, and
State*, *For a New Liberty*), Hazlitt (*Economics in One Lesson*), Hoppe (*A Theory of
Socialism and Capitalism*), Bagus (*The Tragedy of the Euro*).

The corpus and derived dataset are **not** distributed. Most Mises Institute editions
are CC BY-NC-ND, and the ND term forbids redistributing derivative works; Hazlitt is
separately still in copyright. `fetch_books.py` in the GitHub repo resolves the sources
from mises.org so the corpus can be rebuilt.

## Licence

These adapters are derived from Qwen2.5-3B-Instruct and are released under the
**Qwen RESEARCH LICENSE AGREEMENT** — **non-commercial use only**. A copy is included
as `LICENSE` in this repository. Commercial use requires a separate licence from
Alibaba Cloud.

> Qwen is licensed under the Qwen RESEARCH LICENSE AGREEMENT,
> Copyright (c) Alibaba Cloud. All Rights Reserved.

Modifications: LoRA adapter weights trained on the corpus described above. The base
model weights are unmodified and are not redistributed here.

The training *code* in the GitHub repository is MIT; this licence covers the weights.
