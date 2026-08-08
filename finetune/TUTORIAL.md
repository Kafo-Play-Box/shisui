# From Scraped Page to Serving Model

*The whole journey: a `.zim` file becomes a quantized GGUF running on
llama.cpp. Every program, every command, and why we chose each option.*

---

## The shape of the journey

```
.zim file
   |  zimdump: peel the archive open into raw HTML files
   v
HTML pages
   |  shisui scrape: strip the page furniture, keep the prose
   v
plain text
   |  shisui extract: pick one learnable word per sentence
   v
rows (word + phrase)
   |  shisui augment: ask Yomitan for IPA + dictionary meanings
   v
enriched rows
   |  shisui rewrite: an LLM writes the flashcard answer
   v
training rows (input -> answer)
   |  Colab + finetune_run.py: LoRA fine-tune of gemma-3-270m-it
   v
adapter (a few MB)
   |  merge.py: bake the LoRA deltas into the base weights
   v
merged HF model
   |  convert_hf_to_gguf.py: package as a GGUF (f16)
   v
f16 GGUF
   |  llama-quantize: shrink to Q6_K
   v
Q6_K GGUF
   |  patch_gguf_token_types.py: mend two metadata bytes
   v
patched GGUF
   |  llama-server: serve it; your flashcard app talks to it
   v
http://127.0.0.1:8080
```

Why this whole ladder? The end state is a small model that writes flashcard
definitions **on the user's machine, offline, for free** — no API key, no
per-card cost. The steps in between exist to get from raw scraped pages to a
model that has learned exactly one narrow skill.

---

## Stage 0 — unpeel the archive (`zimdump`)

A `.zim` file is a single-file archive of an offline wiki (Kiwix format). One
file, thousands of pages. `zimdump` (from the Arch package `zim-tools`) is
the reference tool for reading it.

```bash
zimdump dump --dir=/tmp/zim_out --ns=A wikipedia_en_100_nopic_2026-07.zim
```

What each option does:

- `dump` — extract the archive's contents to a directory (as opposed to
  `list`/`info`, which only inspect). We want the actual HTML pages.
- `--dir=/tmp/zim_out` — where the extracted files go.
- `--ns=A` — extract only the **article** namespace (`A`). The archive also
  holds images, CSS, JS, redirect stubs — all under other namespaces. We want
  the article text only, and `--ns=A` keeps the temp directory small
  (proportional to article count, not archive size — the thing that makes
  multi-gigabyte archives viable on a laptop).

Why a temp dir: the dump is disposable. We do not keep it; it is raw
material. `--ns=A` is the single most important option here — without it, a
big archive would dump hundreds of megabytes of pictures and scripts we never
use.

---

## Stage 1 — strip the furniture (`shisui scrape`)

The dumped HTML is a web page: navigation, table of contents, footer,
categories, edit links — all chrome. `scrape` parses the DOM and keeps only
the article prose, paragraph by paragraph.

```bash
./shisui scrape --force -o wiki100.plain.txt /tmp/zim_out/
```

Why this stage exists: the later stages feed the model sentences, and a model
learns from what it sees. Feed it "Menu Donate July 22, 2026", and it learns
nothing worth keeping. `scrape` drops, by DOM structure: `nav`, `header`,
`footer`, `aside`, `form`, `script`, `style`, the table of contents, category
links, citation footnotes, hidden elements, and `<pre>` code blocks. Inline
`<code>` stays — it is often a real word inside a sentence.

Options:
- `-o` — write to a file instead of stdout.
- `--force` — overwrite that file if it already exists (the tool refuses to
  clobber without it, so a re-run does not silently destroy work).

---

## Stage 2 — glean the words (`shisui extract`)

Now the text is clean prose, but the model needs one *target* per sentence:
the word worth learning. `extract` picks, for each sentence, the lemmatizable
word whose root is least frequent in a corpus — the rarest word is the one
most likely to be new to a learner.

```bash
./shisui extract --min-words=10 --force -o wiki100.extract.jsonl < wiki100.plain.txt
```

Options:
- `--min-words=10` — skip sentences shorter than 10 words. A flashcard needs
  enough context to infer meaning; a 5-word sentence is not enough.
- `--max-words=80` (default) — truncate very long sentences, centered on the
  target word, so the phrase stays focused.
- `--min-target-len=3` (default) — ignore target candidates shorter than 3
  letters (function words like "it" or "to" are not worth a card).

Why frequency: a word like "bank" appearing 100 times in the corpus is rarer
than "cat" at 5,400. The model should learn the words learners actually
stumble on, not the ones they already know.

---

## Stage 3 — enrich with the dictionary (`shisui augment`)

The model must ground its answer in real definitions, not invent them. For
each root word, `augment` asks a local **Yomitan** server — a running
browser-extension dictionary service — for the IPA reading and the dictionary
meanings.

```bash
./shisui augment --force -o wiki100.augmented.jsonl < wiki100.extract.jsonl
```

Options (defaults fine here):
- `--yomitan-url` — default `http://127.0.0.1:19633`, where Yomitan listens.
- `--yomitan-timeout=5s` — a slow dictionary is worse than no dictionary.
- `--concurrency=4` — a few parallel lookups; Yomitan serializes under load,
  so 4 beats 16 here (learned the hard way).

Rows with no dictionary meanings are dropped: a word Yomitan does not know
cannot be a grounded flashcard. This stage also attaches the **IPA** — and
this is where the "predict the IPA" story begins. The enriched row carries
the correct IPA; later, the model must learn to reproduce it from the word
alone.

---

## Stage 4 — write the teacher labels (`shisui rewrite`)

A model cannot be fine-tuned without an *expected answer*. `rewrite` asks an
LLM to produce, for each row, exactly what a good ESL flashcard should say:
a quality label, a simple definition, and the dictionary meaning it chose.
This is the teacher; the small model will learn to imitate it.

For a first experiment we train on a **20-row sample**, not the whole corpus —
enough to prove the pipeline, cheap to run. First take a sample:

```bash
shuf -n 20 --random-source=<(yes 42) tmp/wiki100.augmented.jsonl > tmp/wiki100.sample20.jsonl
```

(`shuf` picks 20 rows; `--random-source=<(yes 42)` makes the draw repeatable,
so a re-run gives the same sample. The sample is a stand-in — a real run uses
the whole set.)

Then rewrite those rows:

```bash
./shisui rewrite --api-url https://api.deepseek.com --model deepseek-v4-flash \
  --api-key "$SHISUI_API_KEY" --force -o tmp/wiki100.sample20.rewritten.jsonl \
  < tmp/wiki100.sample20.jsonl
```

Options:
- `--api-url` — the OpenAI-compatible endpoint. For DeepSeek, the base is
  `https://api.deepseek.com` (the tool appends `/chat/completions` itself).
- `--model deepseek-v4-flash` — the teacher model.
- `--api-key` — reads `SHISUI_API_KEY` from the environment if not given; the
  flag wins when both are present.

One hard-won detail: `deepseek-v4-flash` is *thinking-enabled* by default,
and it burns its entire token budget on reasoning before emitting anything —
every call came back empty. The tool therefore sends
`thinking: {"type": "disabled"}` for any `deepseek*` model, which returns
clean JSON. (Verified by an A/B: thinking on -> 4/20 rows failed; thinking
off -> 20/20 succeeded.)

For the IPA lesson: 20 rows were sampled for this experiment. A real run uses
thousands.

---

## Stage 5 — train the small model (Colab, `finetune_run.py`)

The fine-tune runs on a rented Colab GPU because a 270M model trains in
minutes on a T4 but would take hours on an ordinary laptop CPU.

```bash
colab new -s finetune --gpu T4
colab install -s finetune transformers peft accelerate datasets
colab upload -s finetune finetune/finetune_run.py
colab upload -s finetune tmp/wiki100.sample20.rewritten.jsonl
colab exec -s finetune -f finetune_run.py --timeout 600
colab download -s finetune /content/lora-adapter.zip ./lora-adapter.zip
colab stop -s finetune
```

What each Colab command does:
- `colab new --gpu T4` — rent a small GPU VM. `T4` is the smallest/cheapest
  GPU and is plenty for a 270M model.
- `colab install transformers peft accelerate datasets` — put the training
  libraries on the VM. `peft` provides LoRA; `transformers` provides the model
  and the `Trainer`.
- `colab upload` — copy the script and the data onto the VM. The first upload
  is the training script (path on *your* machine); the second is the data.
- `colab exec -f ... --timeout 600` — run the script remotely; `-f` reads the
  script *locally* and sends it (so edits are always current), and
  `--timeout 600` gives it 10 minutes (the default 30s dies on model
  download).
- `colab download` — bring the trained adapter home.
- `colab stop` — release the VM. It bills by the hour; leaving it running
  costs money for nothing.

### Why LoRA (and why these parameters)

LoRA freezes the whole base model and adds small trainable "adapters" to a
few weight matrices. Training 1.9M parameters instead of 270M means: less
memory, faster training, and the base model's knowledge stays intact — we are
teaching one narrow skill, not re-teaching language.

- `r=8` — the rank of the adapter matrices. Higher = more capacity (and more
  parameters); 8 is a sensible default for a small, narrow task. 20 rows do
  not need rank 32.
- `lora_alpha=16` — how strongly the adapters are allowed to nudge the base
  weights. A common rule of thumb is alpha about 2x the rank.
- `lora_dropout=0.05` — dropout on the adapter, a small amount of noise to
  discourage memorizing the 20 rows.
- `target_modules` — the matrices the adapters attach to: the attention
  projections (`q/k/v/o`) and the feed-forward gates (`gate/up/down`). These
  are where the model stores most of what it "knows", so they are where a
  narrow skill is best taught.
- `MAX_LEN=1024` — token budget per example; a flashcard fits comfortably.
- `EPOCHS=5` — five passes over the 20 rows. Few passes = the model barely
  learns; too many = it memorizes the 20 rows verbatim. 5 is a first guess
  for a tiny set.
- `LR=2e-4` — the learning rate, how big each training step's weight update
  is. 2e-4 is the peft-recommended default for LoRA; too big and training
  jumps over the optimum, too small and it crawls.
- `BATCH=4` — examples per training step; 4 fits the T4's memory for a 270M
  model.
- `bf16=True` — the model's native precision (the weights are stored as
  bfloat16; mixing precisions would waste memory or lose precision).

The one thing the *script* changed vs a stock LoRA run: **IPA**. The input
prompt deliberately omits the `IPA:` line, and the expected answer adds
`predicted_ipa`. If the IPA were in the prompt, the model would learn to copy
a line. By omitting it, the model must learn to *pronounce* the word — so a
card still gets its pronunciation even when the dictionary lookup has none.

---

## Stage 6 — bake and shrink (`merge.py`, `convert`, `llama-quantize`)

The adapter is a set of deltas; llama.cpp cannot load a LoRA directly. First
the deltas are baked into the base weights, then the result is packaged and
shrunk.

```bash
# venv with torch-cpu, transformers, peft (see BUILD.md)
tmp/venv/bin/python finetune/merge.py tmp/gguf-build/base tmp/lora-adapter tmp/gguf-build/merged

git clone --depth 1 --branch b10223 https://github.com/ggml-org/llama.cpp tmp/gguf-build/llama.cpp

tmp/venv/bin/python tmp/gguf-build/llama.cpp/convert_hf_to_gguf.py tmp/gguf-build/merged \
  --outfile tmp/gguf-build/gemma-3-270m-it-f16.gguf --outtype f16

llama-quantize tmp/gguf-build/gemma-3-270m-it-f16.gguf tmp/gguf-build/gemma-3-270m-it-q6_k.gguf Q6_K
```

Why each step:

- `merge.py` — `PeftModel(...).merge_and_unload()` adds the LoRA deltas to the
  base weights and discards the adapter scaffolding. The result is an ordinary
  HuggingFace model directory. We also copy `tokenizer.model` and
  `chat_template.jinja` from the base, because the adapter directory lacks the
  sentencepiece model the converter needs for gemma.
- `--branch b10223` — the clone is pinned to the *exact* build tag the
  installed `llama-cpp` package ships. The converter and the runtime must
  agree; a mismatched version is how subtle breakage sneaks in.
- `convert_hf_to_gguf.py ... --outtype f16` — turns the HF directory into a
  GGUF. `f16` (half precision) is chosen as the quantization *source*: it
  keeps enough mantissa bits that the next step, Q6_K, loses almost nothing.
  (The converter accepts `--outtype q8_0` etc., but not `q6_k` — K-quants are
  always a separate step.)
- `llama-quantize ... Q6_K` — the actual shrink. The type is a positional
  argument (case-insensitive). Q6_K is ~6.56 bits per weight, a good
  accuracy/size balance for a model that will live on a laptop. It produces a
  ~270 MB file. (Ninety of the 236 tensors fall back to q8_0 because gemma's
  matrix shapes do not fit Q6_K's 256-block structure — llama-quantize's
  documented behavior, not an error.)

---

## Stage 7 — mend the tokenizer (`patch_gguf_token_types.py`)

```bash
cp tmp/gguf-build/gemma-3-270m-it-q6_k.gguf tmp/gguf-build/gemma-3-270m-it-q6_k-patched.gguf
python finetune/patch_gguf_token_types.py tmp/gguf-build/gemma-3-270m-it-q6_k-patched.gguf
```

Why: llama.cpp's tokenizer byte-splits gemma's `<start_of_turn>` — the
token that opens every chat turn — because the GGUF metadata marks it a
*normal* token instead of a *control* token. The base model shrugs off the
mangled input; the fine-tuned model, which learned the single-token shape of
the task very hard, degenerates (it echoes the prompt forever). Google
confirmed the root cause and the fix: the metadata entry must say CONTROL.

The patch is two bytes in the GGUF's metadata array (a control token's id and
value). It is written in place — no tensor is touched; `cmp` shows exactly two
bytes changed. See `BUG-NOTES.md` for the full story and the source.

---

## Stage 8 — serve it (`llama-server`)

```bash
llama-server -m tmp/gguf-build/gemma-3-270m-it-q6_k-patched.gguf -t 4 --port 8080
```

- `-m` — the model file.
- `-t N` — how many CPU threads to use. Match `N` to your machine's thread
  count (4 is a common default for modern laptops); more threads than the CPU
  has just adds overhead.
- `--port 8080` — the HTTP API port (default is already 8080; kept explicit).

llama-server exposes an OpenAI-compatible API, so the flashcard app can talk
to it with ordinary HTTP:

```bash
curl http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d @finetune/chat_request_census.json
```

Why the request body carries `stop` tokens: llama-server relies entirely on
the request payload and the GGUF's chat template — unlike llama-cli, it
applies no interactive defaults. Without `stop`, the server keeps generating
past the end of the answer. The payload lists gemma's end-of-turn marker and
friends so generation stops cleanly.

---

## What the answer looks like

The fine-tuned model, given a word it never saw in training:

```json
{"quality": "good", "simple_meaning": "a count of people in a place",
 "selected_meaning": "a type of tax levied by feudal lords on peasants",
 "predicted_ipa": "ˈsen.səs"}
```

The JSON shape and the stopping behavior hold. The *judgment* — that "census"
here means the population count, not a feudal tax — is still weak, because 20
rows cannot teach it. That is the honest state of the art at this size and
data volume; it is the next thing to sharpen.

One truthful note on the `predicted_ipa` field: it appears in the training
format *now* (this tutorial, and the current `finetune_run.py`, both include
it), but the adapter saved earlier in this experiment was trained before that
change and emits the three original fields only. To get a model that predicts
IPA, retrain with the current script — the format, not the old weights, is
what carries the lesson.

---

## The whole ladder, one breath

- `zimdump` unpeels the archive.
- `shisui scrape` strips the page furniture.
- `shisui extract` picks one learnable word per sentence.
- `shisui augment` grounds each word in dictionary data.
- `shisui rewrite` writes the teacher labels.
- Colab + `finetune_run.py` teach a small model to imitate the teacher,
  including predicting the IPA.
- `merge.py`, `convert_hf_to_gguf.py`, `llama-quantize` bake, package, and
  shrink.
- `patch_gguf_token_types.py` mends the tokenizer metadata.
- `llama-server` serves the result to the flashcard app — offline, free, on
  the user's own machine.
