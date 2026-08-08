#!/usr/bin/env python3
"""Test the finetuned model on a word it never saw in training.

The adapter lives in the gitignored tmp/ dir; point --data-dir there (the
default is <repo>/shisui/tmp). Run:
  python test_unseen_word.py
  python test_unseen_word.py --data-dir PATH
"""
import argparse
import os

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "google/gemma-3-270m-it"
MAX_NEW = 120

p = argparse.ArgumentParser()
p.add_argument(
    "--data-dir",
    default=os.path.abspath(os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "tmp")),
    help="dir holding the lora-adapter/",
)
args = p.parse_args()
ADAPTER = os.path.join(args.data_dir, "lora-adapter")

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, dtype=torch.bfloat16
).to("cpu")
model = PeftModel.from_pretrained(model, ADAPTER)
model.eval()

row = {
    "target_word": "census",
    "root_word": "census",
    "phrase": "At the time of the 2000 US census, 54.8% of African Americans lived in the South.",
    "ipa": "ˈsen.səs",
    "meanings": [
        "an official count or enumeration of members of a population (not necessarily human), usually residents or citizens in a particular region, often done at regular intervals",
        "count, tally",
        "a type of tax levied by feudal lords on peasants",
        "(cellular automata) a count of the number of individual patterns within a larger pattern, most often the ash of a soup or a methuselah",
        "to conduct a census on",
        "to collect a census",
    ],
}

meanings = "\n".join(f"- {m}" for m in row["meanings"])
user = (
    f"Word: {row['target_word']}\n"
    f"Root: {row['root_word']}\n"
    f'Phrase: "{row["phrase"]}"\n'
    f"IPA: {row['ipa']}\n"
    f"Dictionary meanings:\n{meanings}\n"
)

messages = [{"role": "user", "content": user}]
prompt = tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
inputs = tokenizer(prompt, return_tensors="pt")
with torch.no_grad():
    out = model.generate(
        **inputs,
        max_new_tokens=MAX_NEW,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
gen = out[0][inputs["input_ids"].shape[1]:]
print("=== PROMPT ===")
print(user)
print("=== FINETUNED ANSWER (word: census, UNSEEN) ===")
print(tokenizer.decode(gen, skip_special_tokens=True).strip())
