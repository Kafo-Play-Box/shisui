# BUILD.md — LoRA → Q6_K GGUF pipeline log

This is a log of one specific run on one specific machine (the machine this
project was developed on: a 2-core i3, 7.5 GB RAM, CPU-only). The commands
generalize to any machine; the sizes and counts below are from that run.
All paths relative to the project's gitignored `tmp/` dir.
Deliverable: `gemma-3-270m-it-q6_k.gguf` (merged LoRA finetune, Q6_K).
Toolchain: llama.cpp b10223 (commit 11924d4c17, tag `b10223`) — runtime and converter match exactly.

## 1. Download base model — exit 0
```
HF_TOKEN=$(cat tmp/hf_token.txt)
tmp/venv/bin/hf download google/gemma-3-270m-it --local-dir tmp/gguf-build/base --token "$HF_TOKEN"
```
Key output: base model downloaded/cached under `tmp/gguf-build/base` (verified complete: config.json, model.safetensors, tokenizer.model, tokenizer.json, chat_template.jinja).

## 2. Merge LoRA — exit 0
```
tmp/venv/bin/python finetune/merge.py tmp/gguf-build/base lora-adapter tmp/gguf-build/merged
```
`merge.py`: load base `dtype=torch.bfloat16` (transformers v5), wrap `PeftModel.from_pretrained`, `merge_and_unload()`, save with `safe_serialization=True`, save tokenizer, copy `tokenizer.model` + `chat_template.jinja` from base, assert all 5 required files.
Key output: `MERGED_DIR: tmp/gguf-build/merged` — contains model.safetensors (536,223,056 B), config.json, tokenizer.model, chat_template.jinja, tokenizer_config.json (+ tokenizer.json, generation_config.json).

## 3. Clone llama.cpp pinned to installed build — exit 0
```
git clone --depth 1 --branch b10223 https://github.com/ggml-org/llama.cpp tmp/gguf-build/llama.cpp
```
Tag `b10223` exists; no substitution needed. `git describe --tags --exact-match HEAD` = `b10223`, commit `11924d4`; runtime `llama-cli --version` = `10223 (11924d4c17)` — exact match.

## 4. Convert HF → GGUF f16 — exit 0
```
tmp/venv/bin/python tmp/gguf-build/llama.cpp/convert_hf_to_gguf.py \
  tmp/gguf-build/merged --outfile tmp/gguf-build/gemma-3-270m-it-f16.gguf --outtype f16
```
No `gguf` import issue (gguf-py is in the venv path; PYTHONPATH not needed).
Key output: `n_tensors = 236, total_size = 536.3M` … `Model successfully exported`.
File: `gemma-3-270m-it-f16.gguf`, 542,835,840 B (f16, reference).

## 5. Quantize to Q6_K — exit 0
```
llama-quantize tmp/gguf-build/gemma-3-270m-it-f16.gguf tmp/gguf-build/gemma-3-270m-it-q6_k.gguf Q6_K
```
Key output:
```
llama_model_quantize_impl: model size  = 511.46 MiB (16.00 BPW)
llama_model_quantize_impl: quant size  = 263.64 MiB (8.25 BPW)
llama_model_quantize_impl: WARNING: 90 of 236 tensor(s) required fallback quantization
```
Size: **282,975,360 B ≈ 270 MB** (not ~220 MB — see deviations). The warning: gemma-3-270m's matrix shapes (640-wide dims on o_proj/down_proj, etc.) don't fit Q6_K's 256-block structure; llama-quantize falls back to q8_0 for 90 tensors (norms stay f32: 109 f32 + 91 q8_0 + 36 q6_K). Tensor mix verified identical in both my run and a prior session's run — deterministic.

## 6. Verification

### 6a. Metadata — exit 0
`llama-gguf tmp/gguf-build/gemma-3-270m-it-q6_k.gguf r` → GGUF V3, 34 KV pairs, 236 tensors; kv[0] `general.architecture = gemma3`, kv[30] `tokenizer.chat_template` PRESENT.
`llama-cli -m ...q6_k.gguf -p hi -n 1 -lv 4` banner: `arch = gemma3`, `n_layer = 18`, `n_ctx_train = 32768`, `model params = 268.10 M`, `file type = Q6_K`, `file size = 263.64 MiB (8.25 BPW)`, `general.name = Merged`, EOT token 106 `<end_of_turn>`, EOG = `<eos>`/`<end_of_turn>`/`</s>`, `general.file_type = 18` (Q6_K).

### 6b. Tokenizer — exit 0
`llama-tokenize -m ...q6_k.gguf -p "hello 馬"` →
```
2 -> '<bos>'
23391 -> 'hello'
141365 -> ' 馬'
```

### 6c. Generation via llama-server — see FAILURE below
`llama-server -m ...q6_k.gguf -t 2 --port 8080`, curl `POST /v1/chat/completions` with the census prompt, `max_tokens: 120, temperature: 0.0`, stop = `["<|im_end|>","</s>","<|eot_id|>","<end_of_turn>"]`. Server loads clean: `model loaded`, prompt eval 167 tokens, ~70 t/s generation.

**Result: the finetuned model echoes the user prompt instead of emitting the trained JSON; `finish_reason: length` at 120 tokens (no stop token ever emitted).** `--chat-template gemma` (task's suggested fix) changed the output but did not fix it (still `length`, no JSON). Details in the report; root cause is a llama.cpp tokenizer limitation, not the pipeline.

## 7. Deliverables
- `tmp/gguf-build/gemma-3-270m-it-q6_k.gguf` — 282,975,360 B (deliverable)
- `tmp/gguf-build/gemma-3-270m-it-f16.gguf` — 542,835,840 B (f16 reference)
- `tmp/gguf-build/merged/` — merged HF model (536 MB safetensors)
- `finetune/merge.py` — merge script
- `finetune/chat_request_census.json` / `tmp/gguf-build/chat_response_census.json` — verification artifacts

## Deviations
- Q6_K file is 270 MB, not the estimated ~220 MB: 90/236 tensors fall back to q8_0 (block-size mismatch), 109 norms stay f32. This is llama-quantize default behavior, deterministic across runs.
- No llama.cpp version substitution (tag b10223 exists and matches runtime).
- No Python packages were added; no OS packages installed.

## Fix: token_type patch (llama.cpp couldn't run the model)

Root cause (confirmed via google/gemma-3-12b-it-qat-q4_0-gguf/discussions/3 and
verified in the GGUF): <start_of_turn> (id 105) and <end_of_turn> (id 106) were
marked token_type=1 (NORMAL) instead of 3 (CONTROL). llama.cpp's tokenizer
BPE-splits NORMAL tokens; the intermediate merge pieces are UNK in gemma's
vocab, so it byte-split <start_of_turn> and the overfit finetune degenerated
(echoed the prompt, finish_reason=length).

Fix: in-place byte patch of the two int32 metadata entries (data_start +
id*4). patch_gguf_token_types.py. Result: gemma-3-270m-it-q6_k-patched.gguf.

Verified: llama-tokenize now yields single tokens 105/106; llama-server returns
finish_reason=stop with valid {"quality","simple_meaning","selected_meaning"}
JSON for the census prompt. cmp shows the patched file differs from the
original in exactly 2 bytes (no tensor changes).
