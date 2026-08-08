#!/usr/bin/env python3
"""Compare raw vs LoRA-finetuned gemma-3-270m-it on a few flashcard prompts.

Loads the base model once, answers 3 prompts with the RAW model, then loads
the LoRA adapter on top and answers the SAME prompts again, printing both.
The base model (536MB) downloads from HF on first run (gated: needs a token).

The adapter, the training rows, and the HF token are data artifacts that live
in the gitignored tmp/ dir; point --data-dir there (the default is
<repo>/shisui/tmp). The token is read from hf_token.txt in that dir, or from
the HF_TOKEN env var.

Run:
  python compare_raw_vs_finetuned.py                 # default data dir
  python compare_raw_vs_finetuned.py --data-dir PATH
"""

import argparse
import json
import os
import sys

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "google/gemma-3-270m-it"
MAX_NEW = 120          # max tokens the model may generate per answer
ROWS = 3               # how many rows to compare


def parse_args():
    default_data = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "tmp")
    )
    p = argparse.ArgumentParser()
    p.add_argument(
        "--data-dir", default=default_data,
        help="dir holding lora-adapter/, the training jsonl, and hf_token.txt",
    )
    return p.parse_args()


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
            do_sample=False,          # greedy: deterministic output
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    gen = out[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(gen, skip_special_tokens=True).strip()


def main():
    args = parse_args()
    adapter = os.path.join(args.data_dir, "lora-adapter")
    data = os.path.join(args.data_dir, "wiki100.sample20.rewritten.jsonl")
    token = read_hf_token(args.data_dir)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    rows = [json.loads(l) for l in open(data)][:ROWS]

    print(f"device: {device}")
    print(f"loading base model {MODEL_ID} ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, token=token, dtype=torch.bfloat16
    ).to(device)
    model.eval()

    prompts = [build_prompt(r) for r in rows]

    print("\n=== RAW MODEL (no adapter) ===")
    raw_answers = []
    for i, (p, row) in enumerate(zip(prompts, rows)):
        a = answer(model, tokenizer, p, device)
        raw_answers.append(a)
        print(f"\n[{i+1}] word={row['target_word']!r}")
        print(f"    raw: {a}")

    print("\n=== FINETUNED MODEL (base + LoRA adapter) ===")
    model = PeftModel.from_pretrained(model, adapter)
    model.eval()
    finetuned_answers = []
    for i, (p, row) in enumerate(zip(prompts, rows)):
        a = answer(model, tokenizer, p, device)
        finetuned_answers.append(a)
        print(f"\n[{i+1}] word={row['target_word']!r}")
        print(f"    finetuned: {a}")

    print("\n=== EXPECTED (from training data) ===")
    for i, row in enumerate(rows):
        print(f"[{i+1}] quality={row['quality']!r}")
        print(f"    simple_meaning: {row['simple_meaning']}")
        print(f"    selected_meaning: {row['selected_meaning']}")


if __name__ == "__main__":
    sys.exit(main())
