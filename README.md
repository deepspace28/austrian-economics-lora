# austrian-economics-lora

A QLoRA fine-tune of **Qwen2.5-3B-Instruct** on the Austrian school canon — Mises,
Rothbard, Hazlitt, Hoppe, Bagus — trained end to end on a laptop with **7.3 GB of
system RAM and a 4 GB GPU**.

The repository is the whole pipeline, not just the training script: PDF extraction,
font-aware cleaning, chapter detection, semantic chunking, synthetic Q&A generation,
two-stage training, adapter merging, and GGUF conversion.

The interesting constraint was the hardware. A 3B model in 4-bit with gradient
checkpointing and a paged 8-bit optimizer fits in **2.85 GB of VRAM**. Most of the
engineering here is about not running out of memory — including a preflight check
that measures free *commit charge* rather than free physical RAM, because on Windows
that is the constraint that actually kills the load, and it fails as a bare access
violation rather than a Python exception.

## Results

Two stages, because they teach different things.

**Stage 1 — continued pretraining on raw prose.** Book text packed into fixed
384-token blocks with every token supervised. This is what teaches the model to
*write like* the source material; a chat template and prompt masking would both be
wrong for it.

| | |
|---|---|
| Corpus | 1,664,612 tokens → 4,334 blocks |
| Split | 4,248 train / 86 held out |
| Schedule | 2 epochs × 266 steps |
| Eval loss (epoch 1) | 2.4918 — perplexity **12.08** |

**Stage 2 — instruction tuning on generated Q&A**, resuming from the stage 1 adapter.
Prompt tokens masked, chat template applied.

| | |
|---|---|
| Examples | 1,057 pairs (1,036 train / 21 held out) |
| Schedule | 1 epoch × 65 steps |
| Train loss | 2.5196 → **1.8040** |
| Eval loss | 1.9944 — perplexity **7.35** |
| Peak VRAM | 2.85 GB |

Held-out evaluation is deliberately part of the loop. With a corpus this small the
failure mode is memorisation, and a training curve alone will not show you that.

### Honest limitations

- Stage 1 **crashed in epoch 2** with a CUDA illegal memory access. The epoch 1
  checkpoint was saved and is what stage 2 resumes from, so the published adapters
  reflect one epoch of prose training, not two.
- 1,057 instruction pairs is a small dataset. The model adopts the register and
  vocabulary of the source texts convincingly; it is not a reliable authority on
  what any of these authors actually argued.
- No benchmark evaluation. Loss and perplexity on a held-out split is all that was
  measured.
- The model reproduces the perspective of its corpus. That is the point of the
  exercise, but it means the output is one school's position stated confidently,
  not a survey of economics.

## Adapters

| Adapter | Training |
|---|---|
| `economics_books` | Stage 1 only — raw prose |
| `economics_qwen` | Chat pairs only, from base |
| `economics_both` | Stage 1 → stage 2 (**the one to use**) |
| `economics_books_probe` | Short smoke-test run |

All four are r=16, α=32, dropout 0.05, targeting all seven projection modules
(`q/k/v/o_proj`, `gate/up/down_proj`) — **29,933,568 trainable parameters, 0.96%** of
the 3.12B total.

Weights are **not** in this repository. Each adapter is 115 MB, over GitHub's 100 MB
file limit, and the merged model and GGUF quants are 6–8 GB. Training reproduces them
in a couple of hours.

## Pipeline

```
fetch_books.py      download the source PDFs from mises.org
  └─ src/extract.py         PyMuPDF text extraction
  └─ src/cleaner.py         font-size body detection, header/footer removal
  └─ src/chapter_detector.py
  └─ src/semantic_chunker.py
  └─ src/dataset_generator.py   3-8 Q&A pairs per 1,500-char chunk
  └─ src/validator.py
train.py            two-stage QLoRA (TRAIN_MODE=text | chat)
merge_lora.py       fold the adapter into the base weights
ask.py              single prompt, chat or raw-continuation mode
```

The cleaner is font-aware rather than regex-based: it finds the modal body font size
and drops anything that deviates, which removes running heads, page numbers, and
footnotes without hand-tuning per book. On *Human Action* it kept 914 of 952 pages.

## Setup

```bash
pip install -r requirements.txt
python fetch_books.py          # get the source PDFs
python src/main.py             # extract -> clean -> chunk -> dataset

set TRAIN_MODE=text  & python train.py    # stage 1
set TRAIN_MODE=chat  & set RESUME_ADAPTER=adapters/economics_books & python train.py

python ask.py "Why does credit expansion cause a boom that must end in a bust?"
python ask.py --raw "Prices are signals that"
```

Training is configured by environment variable — `QWEN_MODEL`, `TRAIN_MODE`,
`BLOCK_SIZE`, `BATCH_SIZE`, `GRAD_ACCUM`, `EPOCHS`, `LEARNING_RATE`, `MAX_STEPS`.
Set `MAX_STEPS=3` for a smoke test before committing to a long run.

If you have less memory than this was built for, `QWEN_MODEL=Qwen/Qwen2.5-1.5B-Instruct`
works and the preflight check will suggest it.

## Hyperparameters

| | |
|---|---|
| Quantization | 4-bit NF4, double quant, fp16 compute |
| Optimizer | PagedAdamW8bit |
| Learning rate | 2e-4, cosine schedule, 3% warmup |
| Effective batch | 16 (batch 1 × grad accum 16) |
| Sequence length | 192 chat / 384 packed prose |
| Gradient checkpointing | on, non-reentrant |

## Source texts and licensing

The PDFs and everything derived from them — extracted text, cleaned corpus, generated
dataset — are **excluded from this repository**.

Most Mises Institute editions are published under CC BY-NC-ND. That licence makes the
books free to read and copy, but the **ND** term forbids distributing derivative works,
and a cleaned corpus or a generated Q&A set is a derivative work. Hazlitt's *Economics
in One Lesson* is separately still in copyright.

`fetch_books.py` resolves each text from mises.org so you can build the corpus
yourself. Two resolve automatically; the rest print a search link.

- Ludwig von Mises — *Human Action*, *Interventionism*
- Murray N. Rothbard — *Man, Economy, and State*, *For a New Liberty*
- Henry Hazlitt — *Economics in One Lesson*
- Hans-Hermann Hoppe — *A Theory of Socialism and Capitalism*
- Philipp Bagus — *The Tragedy of the Euro*

**Base model licence:** Qwen2.5-3B is licensed differently from most of the Qwen2.5
family — check the [model card](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct)
before using derived weights commercially. The code in this repository is MIT.

## Licence

MIT for the pipeline code. See [LICENSE](LICENSE). The source texts retain their own
licences and are not distributed here.
