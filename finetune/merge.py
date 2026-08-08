#!/usr/bin/env python
"""Merge the LoRA adapter into the base gemma-3-270m-it and save the merged model.

Usage: merge.py <base_dir> <adapter_dir> <out_dir>
"""
import shutil
import sys
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

def main() -> None:
    base_dir, adapter_dir, out_dir = (Path(p) for p in sys.argv[1:4])

    model = AutoModelForCausalLM.from_pretrained(
        base_dir, dtype=torch.bfloat16, device_map="cpu"
    )
    model = PeftModel.from_pretrained(model, adapter_dir)
    model = model.merge_and_unload()

    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir, safe_serialization=True)
    tokenizer = AutoTokenizer.from_pretrained(base_dir)
    tokenizer.save_pretrained(out_dir)

    # The adapter dir lacks tokenizer.model; the converter needs it for gemma3.
    for name in ("tokenizer.model", "chat_template.jinja"):
        src = base_dir / name
        if src.is_file() and not (out_dir / name).exists():
            shutil.copy2(src, out_dir / name)
            print(f"copied {name}")

    files = sorted(p.name for p in out_dir.iterdir())
    print(f"MERGED_DIR: {out_dir}")
    print(f"FILES: {', '.join(files)}")
    assert (out_dir / "model.safetensors").is_file(), "model.safetensors missing"
    assert (out_dir / "config.json").is_file(), "config.json missing"
    assert (out_dir / "tokenizer.model").is_file(), "tokenizer.model missing"
    assert (out_dir / "chat_template.jinja").is_file(), "chat_template.jinja missing"
    assert (out_dir / "tokenizer_config.json").is_file(), "tokenizer_config.json missing"

if __name__ == "__main__":
    main()
