#!/usr/bin/env python3
"""Full prompt + full answer, raw vs finetuned, for every training row.

Loads the base model once, generates an answer for every row, then loads the
LoRA adapter and does the same, writing the FULL user prompt and the FULL raw
model output for both into compare_output.txt alongside the expected answer.

Run:
  python compare_all.py                  # writes compare_output.txt
  python compare_all.py 5                # only the first 5 rows (faster)
  python compare_all.py 5 --data-dir P   # data artifacts live elsewhere
"""

import json
import os
import sys

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA = os.path.abspath(os.path.join(HERE, "..", "tmp"))
MODEL_ID = "google/gemma-3-270m-it"
MAX_NEW = 120


def read_hf_token(data_dir):
    if os.environ.get("HF_TOKEN"):
        return os.environ["HF_TOKEN"]
    tok = os.path.join(data_dir, "hf_token.txt")
    if os.path.exists(tok):
        return open(tok).read().strip()
    raise SystemExit("no HF_TOKEN set and no hf_token.txt found")


def build_prompt(row):
    meanings = "\n".join(f"- {m}" for m in row["meanings"])
    return (
        f"Word: {row['target_word']}\n"
        f"Root: {row['root_word']}\n"
        f'Phrase: "{row["phrase"]}"\n'
        f"IPA: {row['ipa']}\n"
        f"Dictionary meanings:\n{meanings}\n"
    )


def answer(model, tokenizer, user_text, device):
    messages = [{"role": "user", "content": user_text}]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    gen = out[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(gen, skip_special_tokens=True).strip()


def main():
    import argparse
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("limit", nargs="?", type=int, default=None,
                   help="only the first N rows (faster)")
    p.add_argument("--data-dir", default=DEFAULT_DATA,
                   help="dir with lora-adapter/, training jsonl, hf_token.txt")
    p.add_argument("--out", default=None, help="output file path")
    args = p.parse_args()

    limit = args.limit
    data_dir = args.data_dir
    adapter = os.path.join(data_dir, "lora-adapter")
    data = os.path.join(data_dir, "wiki100.sample20.rewritten.jsonl")
    out_path = args.out or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "compare_output.txt"
    )

    token = read_hf_token(data_dir)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    rows = [json.loads(l) for l in open(data)]
    if limit:
        rows = rows[:limit]

    print(f"device: {device}, rows: {len(rows)}")
    print(f"loading base model {MODEL_ID} ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, token=token, dtype=torch.bfloat16
    ).to(device)
    model.eval()

    lines = []
    lines.append("=" * 78)
    lines.append(f"RAW vs FINETUNED  |  {MODEL_ID}  |  device: {device}")
    lines.append("=" * 78)

    prompts = [build_prompt(r) for r in rows]

    print("raw answers ...")
    raw_answers = []
    for i, (row, prompt) in enumerate(zip(rows, prompts)):
        raw_answers.append(answer(model, tokenizer, prompt, device))
        print(f"  [{i+1}/{len(rows)}] raw done for {row['target_word']!r}")

    print("loading adapter ...")
    model = PeftModel.from_pretrained(model, adapter)
    model.eval()

    print("finetuned answers ...")
    fin_answers = []
    for i, (row, prompt) in enumerate(zip(rows, prompts)):
        fin_answers.append(answer(model, tokenizer, prompt, device))
        print(f"  [{i+1}/{len(rows)}] finetuned done for {row['target_word']!r}")

    for i, (row, prompt) in enumerate(zip(rows, prompts)):
        lines.append("\n" + "#" * 78)
        lines.append(f"# ROW {i+1}/{len(rows)}  word={row['target_word']!r}")
        lines.append("#" * 78)
        lines.append("\n--- FULL PROMPT ---")
        lines.append(prompt)
        lines.append("\n--- RAW MODEL ANSWER ---")
        lines.append(raw_answers[i])
        lines.append("\n--- FINETUNED MODEL ANSWER ---")
        lines.append(fin_answers[i])

    lines.append("\n" + "#" * 78)
    lines.append("# EXPECTED (training data) — for reference")
    lines.append("#" * 78)
    for i, row in enumerate(rows):
        lines.append(
            f"[{i+1}] {row['target_word']!r}  quality={row['quality']!r}"
        )
        lines.append(f"    simple_meaning: {row['simple_meaning']}")
        lines.append(f"    selected_meaning: {row['selected_meaning']}")

    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    sys.exit(main())
