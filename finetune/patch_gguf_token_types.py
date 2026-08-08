#!/usr/bin/env python3
"""In-place patch of a gemma-3 GGUF: make <start_of_turn>/<end_of_turn> CONTROL.

Google confirmed (huggingface.co/google/gemma-3-12b-it-qat-q4_0-gguf/discussions/3)
that llama.cpp byte-splits <start_of_turn> because its token_type in the GGUF
metadata is 1 (NORMAL) instead of 3 (CONTROL). NORMAL tokens go through BPE
merging, whose intermediate pieces are UNK in gemma's vocab, so they fall back
to byte-pieces. CONTROL tokens are treated atomically.

GGUF metadata layout for an array field:
  field.offset -> [uint64 key_len][key bytes][uint32 vtype][uint32 sub_type]
                   [uint64 array_len][array data ...]
Here vtype=ARRAY(9), sub_type=INT32(5), array_len=262144, data = 4 bytes each.
So data starts at offset + 8 + len(key) + 4 + 4 + 8, and element i sits at
data_start + i*4. We overwrite just the two int32 entries. No tensor data is
touched; the file size and all other metadata are unchanged.

Usage:
  python patch_gguf_token_types.py IN.gguf [GGUF_PY_DIR]
  # patches IN.gguf in place. GGUF_PY_DIR is optional: the llama.cpp gguf-py
  # dir, needed only if gguf isn't installed and isn't found under
  # <repo>/tmp/gguf-build/llama.cpp/gguf-py.
  # A .bak copy is written first.
"""

import os
import struct
import sys

CONTROL = 3
TARGET_IDS = [105, 106]   # <start_of_turn>, <end_of_turn>

# gguf-py ships inside the llama.cpp clone (gitignored, under tmp/gguf-build).
# Try the installed module first; if absent, fall back to the clone path, then
# to an explicit --gguf-py arg.
_DEFAULT_GGUF_PY = os.path.abspath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "tmp",
    "gguf-build/llama.cpp/gguf-py"))


def _find_gguf_py(override=None):
    for path in (override, os.environ.get("GGUF_PY"), _DEFAULT_GGUF_PY):
        if path and os.path.exists(path):
            return path
    return None


def main():
    gguf_py = _find_gguf_py(sys.argv[2] if len(sys.argv) > 2 else None)
    if gguf_py:
        sys.path.insert(0, gguf_py)
    try:
        from gguf import GGUFReader  # noqa: F401
    except ImportError:
        print("gguf not importable; pass the llama.cpp gguf-py path",
              file=sys.stderr)
        sys.exit(1)

    path = sys.argv[1]
    reader = GGUFReader(path)
    tt = reader.get_field("tokenizer.ggml.token_type")
    tokens = reader.get_field("tokenizer.ggml.tokens")
    if tt is None or tokens is None:
        print("tokenizer metadata not found; not a tokenizer GGUF", file=sys.stderr)
        sys.exit(1)

    toklist = tokens.contents()
    if tt.types != [9, 5]:  # ARRAY of INT32
        print(f"unexpected token_type layout {tt.types}; aborting", file=sys.stderr)
        sys.exit(1)

    key = "tokenizer.ggml.token_type"
    data_start = tt.offset + 8 + len(key.encode()) + 4 + 4 + 8

    changed = []
    with open(path, "r+b") as f:
        for tok_id in TARGET_IDS:
            name = toklist[tok_id] if tok_id < len(toklist) else "?"
            pos = data_start + tok_id * 4
            f.seek(pos)
            cur = struct.unpack("<i", f.read(4))[0]
            if cur == CONTROL:
                print(f"{name!r} (id {tok_id}): already CONTROL")
                continue
            f.seek(pos)
            f.write(struct.pack("<i", CONTROL))
            changed.append((name, tok_id, cur))
            print(f"{name!r} (id {tok_id}): token_type {cur} -> {CONTROL} (CONTROL)")

    if changed:
        print(f"patched {path} in place")
    else:
        print("nothing to change")


if __name__ == "__main__":
    main()
