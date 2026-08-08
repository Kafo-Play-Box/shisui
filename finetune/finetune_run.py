#!/usr/bin/env python3
"""LoRA fine-tune of google/gemma-3-270m on shisui flashcard rows.

What this does, in plain words:
  1. Loads the small Gemma model (already good at language) and the tokenizer.
  2. Reads the 20 training rows: each row has an "input" (word + phrase +
     meanings) and the "answer" the model should learn to produce (quality,
     simple_meaning, selected_meaning) — the exact output the rewrite stage
     generated with DeepSeek.
  3. Attaches LoRA: tiny trainable adapters on a few weight matrices. The
     original model weights stay frozen, so we train a handful of parameters
     instead of all 270M. That is the LoRA trick — cheap and fast.
  4. Trains with the standard Transformers Trainer, computing loss only on
     the answer part (the prompt is masked), so the model learns to write the
     answer, not to repeat the question.
  5. Saves ONLY the adapter (a few MB), not the whole model.

Run it on a Colab VM:  upload this script + the data + your HF token, then
`python3 finetune_run.py`. The adapter lands in /content/lora-adapter.
"""

import json
import os

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

DATA = "/content/wiki100.sample20.rewritten.jsonl"   # the 20 training rows
# -it = instruction-tuned variant: already knows chat format + how to follow
# instructions. The plain model (no -it) is a pretrained base and would need
# far more data to learn basic chat before your task.
MODEL_ID = "google/gemma-3-270m-it"
OUT_DIR = "/content/lora-adapter"                    # where the adapter goes
MAX_LEN = 1024                                       # token budget per example
EPOCHS = 5                                           # passes over the 20 rows
LR = 2e-4                                            # learning rate
BATCH = 4                                            # examples per training step

# Set in main() from the loaded tokenizer; used by pad_collator.
PAD_ID = 0

# colab upload may place files at /content or at the VM root; try both.
DATA_CANDIDATES = [
    DATA,
    "/wiki100.sample20.rewritten.jsonl",
    "wiki100.sample20.rewritten.jsonl",
]
TOKEN_CANDIDATES = ["/content/hf_token.txt", "/hf_token.txt", "hf_token.txt"]


def first_existing(candidates):
    for p in candidates:
        if os.path.exists(p):
            return p
    raise SystemExit(f"none of these files exist: {candidates}")


def load_rows(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def read_hf_token():
    """Gemma is license-gated. Token from env, else from an uploaded file."""
    if os.environ.get("HF_TOKEN"):
        return os.environ["HF_TOKEN"]
    tok_file = first_existing(TOKEN_CANDIDATES)
    return open(tok_file).read().strip()


def make_user_prompt(row):
    """Mirror the shisui rewrite prompt: word, phrase, IPA, meanings."""
    meanings = "\n".join(f"- {m}" for m in row["meanings"])
    return (
        f"Word: {row['target_word']}\n"
        f"Root: {row['root_word']}\n"
        f'Phrase: "{row["phrase"]}"\n'
        f"IPA: {row['ipa']}\n"
        f"Dictionary meanings:\n{meanings}\n"
    )


def make_answer(row):
    """The exact JSON the model must learn to write."""
    return (
        f'{{"quality": "{row["quality"]}", '
        f'"simple_meaning": "{row["simple_meaning"]}", '
        f'"selected_meaning": "{row["selected_meaning"]}"}}'
    )


def tokenize_example(example, tokenizer):
    """Turn one row into input_ids + labels, masking the prompt so the model
    only learns from the answer. The -it model ships a chat_template, so use
    it (canonical format) rather than hand-writing Gemma's turn markers."""
    user_text = tokenizer.apply_chat_template(
        [{"role": "user", "content": example["user"]}],
        tokenize=False,
        add_generation_prompt=True,
    )
    full = user_text + example["answer"] + tokenizer.eos_token
    enc = tokenizer(full, truncation=True, max_length=MAX_LEN)
    prompt_len = len(tokenizer(user_text, truncation=True, max_length=MAX_LEN)["input_ids"])
    labels = [-100] * prompt_len + enc["input_ids"][prompt_len:]
    return {
        "input_ids": enc["input_ids"],
        "attention_mask": enc["attention_mask"],
        "labels": labels,
    }


def pad_collator(batch):
    """Pad a list of tokenized examples to the longest in the batch. input_ids
    and attention_mask pad with the tokenizer's pad token; labels pad with -100
    (the ignore index, so padding tokens contribute no loss)."""
    max_len = max(len(ex["input_ids"]) for ex in batch)
    input_ids, attention_mask, labels = [], [], []
    for ex in batch:
        pad = max_len - len(ex["input_ids"])
        input_ids.append(ex["input_ids"] + [PAD_ID] * pad)
        attention_mask.append(ex["attention_mask"] + [0] * pad)
        labels.append(ex["labels"] + [-100] * pad)
    return {
        "input_ids": torch.tensor(input_ids),
        "attention_mask": torch.tensor(attention_mask),
        "labels": torch.tensor(labels),
    }


def main():
    token = read_hf_token()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    global PAD_ID
    PAD_ID = tokenizer.pad_token_id

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        token=token,
        dtype=torch.bfloat16,         # match the model's native precision
    )

    # LoRA config: r=8 ranks, applied to the attention + feed-forward matrices.
    lora = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()   # proves only the adapters train

    rows = load_rows(first_existing(DATA_CANDIDATES))
    print(f"loaded {len(rows)} training rows")
    examples = [
        {"user": make_user_prompt(r), "answer": make_answer(r)}
        for r in rows
    ]
    ds = Dataset.from_list(examples).map(
        lambda ex: tokenize_example(ex, tokenizer),
        remove_columns=["user", "answer"],
    )

    args = TrainingArguments(
        output_dir="/content/train-out",
        per_device_train_batch_size=BATCH,
        learning_rate=LR,
        num_train_epochs=EPOCHS,
        logging_steps=1,
        save_strategy="epoch",
        bf16=True,                 # fast GPU math
        report_to=[],              # no wandb/tensorboard noise
        seed=42,
    )
    collator = pad_collator

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=ds,
        data_collator=collator,
    )
    trainer.train()

    model.save_pretrained(OUT_DIR)     # adapter only, not the full model
    tokenizer.save_pretrained(OUT_DIR)
    print(f"DONE. Adapter saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
