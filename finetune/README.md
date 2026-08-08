# Fine-Tuning the Definition Writer

The gleaned rows from 拾穗 (Shí Suì) are the grain. This directory holds the
tools that grind it into the mill: a small model, tuned to write the
flashcard definitions that the `rewrite` stage once asked an API to write.
The goal is a model that does that work on the machine itself, offline, and
free.

## The two phases

There are two distinct seasons in this work, and they use different tools.

### 1. Training (on a Colab VM, GPU)

`finetune_run.py` runs on a rented Colab runtime. It takes the rewritten
training rows (word + phrase + meanings in, `quality` / `simple_meaning` /
`selected_meaning` out), attaches a LoRA adapter to `google/gemma-3-270m-it`,
and saves only the adapter — a few megabytes, not the whole model.

Why a VM? A 270M model trains in minutes on a free T4 but would take the
better part of a day on the 2-core i3 at home. The rental is cheap; the tea
is worth it.

The adapter, the training data, and the HuggingFace token are **data** — they
live in the gitignored `tmp/` dir, never in this repository.

### 2. Conversion (local, CPU)

The scripts here merge the adapter into the base model and package it as a
quantized GGUF that `llama.cpp` can serve:

| Step | Tool | What it does |
|------|------|--------------|
| merge | `merge.py` | bakes the LoRA deltas into the base weights (`merge_and_unload`) |
| convert | llama.cpp `convert_hf_to_gguf.py` | HF directory -> unquantized GGUF (f16) |
| quantize | `llama-quantize` | f16 GGUF -> Q6_K (~270 MB) |
| patch | `patch_gguf_token_types.py` | repairs two metadata bytes (see `BUG-NOTES.md`) |

A 2-core CPU can merge and quantize a 270M model comfortably. The merge peaks
around 0.6 GB of RAM.

## Files

| File | Purpose |
|------|---------|
| `finetune_run.py` | Colab training script (LoRA SFT on gemma-3-270m-it) |
| `merge.py` | merge LoRA adapter into the base model (local) |
| `patch_gguf_token_types.py` | fix `<start_of_turn>` token metadata in a GGUF |
| `compare_raw_vs_finetuned.py` | raw vs finetuned answers on a few rows |
| `compare_all.py` | full prompt + answer for every row, to a file |
| `test_unseen_word.py` | finetuned answer on a word never seen in training |
| `chat_request_census.json` | canned census prompt for llama-server verification |
| `BUG-NOTES.md` | the token-splitting bug, its source, and the 2-byte fix |
| `BUILD.md` | the logged conversion run (commands, sizes, deviations) |

The comparison and test scripts need the adapter and data from `tmp/`; pass
`--data-dir` to point elsewhere, or rely on the default of `<repo>/tmp`.

## The quick path

Train on Colab:

```bash
colab new -s finetune --gpu T4
colab install -s finetune transformers peft accelerate datasets
colab upload -s finetune finetune_run.py
colab upload -s finetune tmp/wiki100.sample20.rewritten.jsonl
colab exec -s finetune -f finetune_run.py --timeout 600
colab download -s finetune /content/lora-adapter.zip ./lora-adapter.zip
colab stop -s finetune
```

Merge and package locally:

```bash
# venv with torch-cpu + transformers + peft (see BUILD.md)
tmp/venv/bin/python finetune/merge.py tmp/gguf-build/base tmp/lora-adapter tmp/gguf-build/merged
# clone llama.cpp at the installed build tag (b10223), then:
tmp/venv/bin/python tmp/gguf-build/llama.cpp/convert_hf_to_gguf.py tmp/gguf-build/merged \
  --outfile tmp/gguf-build/gemma-3-270m-it-f16.gguf --outtype f16
llama-quantize tmp/gguf-build/gemma-3-270m-it-f16.gguf tmp/gguf-build/gemma-3-270m-it-q6_k.gguf Q6_K
cp tmp/gguf-build/gemma-3-270m-it-q6_k.gguf tmp/gguf-build/gemma-3-270m-it-q6_k-patched.gguf
python finetune/patch_gguf_token_types.py tmp/gguf-build/gemma-3-270m-it-q6_k-patched.gguf
```

Serve:

```bash
llama-server -m tmp/gguf-build/gemma-3-270m-it-q6_k-patched.gguf -t 2 --port 8080
```

## The honest caveat

Twenty rows teach a 270M model the *shape* of the task — the JSON, the
fields, the behavior — remarkably well. They do not teach it the *judgment*
of which meaning fits which phrase. A real dataset of thousands of rows is
what sharpens that. This directory is the mill; the grain is still growing.

See `BUG-NOTES.md` for the one sharp edge found along the way: llama.cpp's
tokenizer byte-splits Gemma's `<start_of_turn>` unless its GGUF metadata marks
it a control token, and the two-byte patch that mends it.
