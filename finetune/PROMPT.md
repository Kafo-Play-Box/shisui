# Prompting the Fine-Tuned Model

*The exact prompt and answer format the fine-tuned flashcard model was trained
on. Read this first when writing a wrapper around the model (an app, a server
route, a curl, a test).*

---

## One paragraph

The fine-tuned model is `google/gemma-3-270m-it` + a LoRA adapter. It is an
*instruction-follower by imitation*, not by rules: it produces whatever shape
it was trained on, so the wrapper MUST send the prompt below **verbatim in
format**, and MUST expect the answer **in the exact key order shown**. The
model was trained with NO system prompt and NO IPA in the input — the wrapper
must not add either.

## The input (user turn)

```
Word: <target_word>
Root: <root_word>
Phrase: "<phrase>"
Dictionary meanings:
- <meaning 1>
- <meaning 2>
...
```

Rules:

- **No `IPA:` line.** The model learned to *predict* pronunciation; giving it
  the IPA in the input teaches nothing and shifts its behavior.
- **No system message.** The trained examples are a single user turn only.
- Each meaning on its own line, prefixed with `- ` (dash space).
- The phrase is wrapped in double quotes.
- An empty `Output:` line is NOT required — `make_user_prompt` does not emit
  one. Keep the prompt exactly as the lines above.

## The answer (model output)

The model was trained to emit a single JSON object with **this key order**:

```json
{"selected_meaning": "<verbatim dictionary meaning>",
 "simple_meaning": "<simple English definition>",
 "quality": "good",
 "predicted_ipa": "<IPA reading>"}
```

Key order is significant (selected_meaning FIRST), because the model emits
exactly the order it learned:

| Key | Meaning | Example |
|-----|---------|---------|
| `selected_meaning` | the dictionary meaning, copied verbatim from the input list, that fits the phrase | `"in the time following (an event or another period of time)"` |
| `simple_meaning` | the same idea rewritten in simple English (A2-B1), under 15 words | `"after the war means in the time following the war"` |
| `quality` | `"good"` \| `"medium"` \| `"bad"` — how well the phrase matches one meaning | `"good"` |
| `predicted_ipa` | the IPA the model predicts from the word alone | `"ˈæf.tɚ"` |

Notes:

- `selected_meaning` may be `""` and `quality` `"bad"` when no meaning fits.
- `predicted_ipa` may be `""` for words the model cannot pronounce.
- The Go/rewrite-side parser is order-insensitive, but the *model* is not.
  A wrapper that parses JSON is fine; a wrapper that appends to the raw text
  or diffs it should expect this exact order.

## Full example (round trip)

User turn:

```
Word: after
Root: after
Phrase: "World War II caused a pause in palaeontological research; after the war, research attention was also diverted increasingly to fossil mammals rather than dinosaurs, which were seen as sluggish and cold-blooded."
Dictionary meanings:
- following in time, place, or order
- to be looking for someone or something or trying to find or get him, her, or it
- used to say politely that someone can go in front of you or serve themselves with food before you
- typical of or similar to the style of
- used when giving someone or something the same name as another person or thing
- later than someone or something else
- at a time that is later than another event
- coming after
- as a result of; because
- despite
- wanting to find or have
- in the time following (an event or another period of time)
```

Expected answer:

```json
{"selected_meaning": "in the time following (an event or another period of time)", "simple_meaning": "after the war means in the time following the war", "quality": "good", "predicted_ipa": "ˈæf.tɚ"}
```

## Serving it

### transformers (in-process)

Load base + adapter, then send the user turn through the model's chat
template:

```python
messages = [{"role": "user", "content": user_text}]
prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(prompt, return_tensors="pt")
# model.generate(..., max_new_tokens=120, do_sample=False)
```

Use greedy decoding (`do_sample=False`) for deterministic output.

### llama.cpp / llama-server (GGUF)

The GGUF carries the chat template, so send the user turn as a normal chat
message and set stop tokens explicitly — the server applies no interactive
defaults:

```bash
curl http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "messages": [{"role": "user", "content": "<user turn above>"}],
    "max_tokens": 120,
    "temperature": 0.0,
    "stop": ["<|im_end|>", "</s>", "<|eot_id|>", "<end_of_turn>"]
  }'
```

## Where this lives in code

| Artifact | Location |
|----------|----------|
| User prompt builder | `finetune_run.py:make_user_prompt()` |
| Answer builder (key order) | `finetune_run.py:make_answer()` |
| Expected-answer docs for wrappers | this file |
| Rewrite-stage (teacher) system prompt | `internal/rewrite/prompts/system.txt` |
| Rewrite-stage (teacher) user template | `internal/rewrite/prompts/user.txt` |

## Why it looks like this

- **No IPA in input** — the model was trained to predict `predicted_ipa` from
  the word alone, so it is useful even when the dictionary has no reading.
- **No system prompt** — the fine-tune examples are single user turns; the
  model never learned to obey a system message.
- **`selected_meaning` first** — the model commits to *which meaning fits*
  before writing the definition, mirroring the teacher prompt's order.
- **Meaning list injected in the prompt** — grounding the answer in provided
  meanings prevents hallucinated definitions.

## Known limits

- 20-row training teaches the *shape* reliably; the *judgment* of which
  meaning fits is still weak. Expect wrong `selected_meaning` picks on
  ambiguous words.
- The currently saved adapter predates the IPA and key-order changes; a
  retrain with the current `finetune_run.py` is needed for `predicted_ipa`
  and the new key order to appear in output.
