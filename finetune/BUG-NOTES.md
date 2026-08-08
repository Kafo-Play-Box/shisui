# The Two Missing Strokes

*A small story of a token that was split, and the two bytes that mended it.*

---

## The pot was cold

After the Q6_K quantization, the gleaned model would not speak through
`llama-server`. Given a word, a phrase, and its meanings, it did not answer
with a definition. It echoed the question back, over and over, until the
context ran out — `finish_reason: length`, never `stop`. The same model, in
transformers, answered cleanly. The same question, with the same weights,
through a different door, gave a different answer. The tea was not wrong; the
cup was.

## The finger pointed at the moon

The search tool, 指月 (zhiyue), points. A discussion on the Hugging Face
forums pointed too:

- **Source 1** — the report of the split, by mw55a:
  https://huggingface.co/google/gemma-3-12b-it-qat-q4_0-gguf/discussions/3
- **Source 2** — Google's own confirmation, by lkv (Google org), Sep 2 2025:
  > "the `<start_of_turn>` token (ID 105) being split into multiple tokens is
  > caused by an incorrect token_type in the GGUF metadata. In the QAT GGUF
  > files, it's currently marked as a normal token (token_type = 1) instead of
  > a control token (token_type = 3), which causes llama.cpp to treat it
  > incorrectly during tokenization."
- **Source 3** — the metadata diff, by stduhpf: the same discussion details
  the exact `NORMAL -> CONTROL` tokens (`<mask>`, `<start_of_turn>`,
  `<end_of_turn>`, `<start_of_image>`, `<end_of_image>`).

The moon, then: Gemma's tokenizer has a single token, id 105, for
`<start_of_turn>`. llama.cpp's tokenizer only treats tokens whose metadata says
`CONTROL` as indivisible. Tokens marked `NORMAL` are passed through the BPE
merge machinery — and the intermediate merge pieces (`<s`, `start_`, `start_of`,
`of_turn`, ...) are not in Gemma's vocabulary at all. So the merge cannot
reassemble the token, and the tokenizer falls back to byte-pieces:
`<`, `start`, `_`, `of`, `_`, `turn`, `>`. Seven pieces where one should stand.

The base model tolerates the mangled input. A twenty-row LoRA, which learned
the single-token shape of the task very hard, does not. That is why the
finetuned model degenerated while the base walked past the same gate.

## Confirmation before the fix

Verification, never half-claimed. First, prove the diagnosis on the actual
file:

```
sentencepiece (the real tokenizer):
  '<start_of_turn>' -> [105]          # one token

llama-tokenize (the GGUF path):
  236820 '<'   3041 'start'   236779 '_'   1340 'of'   236779 '_'   887 'turn'   236813 '>'
```

And read the metadata field:

```
GGUF tokenizer.ggml.token_type:
  <start_of_turn> (id 105): token_type = 1 (NORMAL)   <- wrong
  <end_of_turn>   (id 106): token_type = 1 (NORMAL)   <- wrong
  <bos>           (id   2): token_type = 3 (CONTROL)  <- right
  <eos>           (id   1): token_type = 3 (CONTROL)  <- right
```

The pattern is unmistakable. The two turn markers are the outliers.

## The fix

`token_type` is a metadata array in the GGUF. It is not a tensor. It lives in
the file's header region, laid out as:

```
field.offset
  -> [uint64 key_len][key bytes]     # "tokenizer.ggml.token_type"
  -> [uint32 vtype=9 (ARRAY)]
  -> [uint32 sub_type=5 (INT32)]
  -> [uint64 array_len]
  -> [array data ...]                # 4 bytes per element, id * 4 to reach it
```

So element `id` sits at `offset + 8 + key_len + 4 + 4 + 8 + id*4`. The whole
fix is writing the value `3` into the two slots for ids 105 and 106. Eight
bytes. No tensor is touched; the file keeps its size and every other byte.

Two paths were weighed. Rewriting the GGUF through `GGUFWriter` would mean
re-emitting 236 tensors (some of them Q8_0 fallbacks), rebuilding the header,
and risking any mismatch in the process. The mended tea should be the same
tea. So the patch is surgical: open the file, seek to the two offsets, write
`3`, close. A single script does it, reading the token names first so it never
patches the wrong slot.

The script lives here, beside this note:

- **`patch_gguf_token_types.py`** — the mender itself.

Run it against a copy of the GGUF:

```
cp gemma-3-270m-it-q6_k.gguf gemma-3-270m-it-q6_k-patched.gguf
python patch_gguf_token_types.py gemma-3-270m-it-q6_k-patched.gguf
```

## The proof

- `cmp` on the original and patched files: **exactly two bytes differ**, at
  the offsets of tokens 105 and 106. Nothing else moved.
- `llama-tokenize` now yields `105 '<start_of_turn>'` and `106 '<end_of_turn>'`
  as single tokens.
- `llama-server` on the patched GGUF, asked about the word *census*, returns
  `finish_reason: stop` and a well-formed answer:

  ```json
  {"quality": "good", "simple_meaning": "a count of people in a place",
   "selected_meaning": "a type of tax levied by feudal lords on peasants"}
  ```

  The form holds. (The chosen sense is not the right one — the tax, not the
  population count. That is the twenty-row weakness of meaning, not the
  tokenizer. The format and the stopping are the mended things.)

## A note for the road

Unsloth's GGUFs already carry `token_type = 3` for these tokens, so a future
retrain through their converter walks past this gate unbothered. The patch
belongs to the models we already have. It is small, it is understood, and it
stays in the record so the next gleaner does not relight this kettle.
