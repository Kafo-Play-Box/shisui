# 拾穗 (Shí Suì)

拾穗 means *gleaning*: a gleaner walks the harvested field and gathers the
scattered grain one stalk at a time. This tool does the same — it walks
plaintext, picks individual words (`extract`), winnows each against a
dictionary to separate grain from chaff (`augment`), and stores the clean crop
(`rewrite`). The output is an LLM fine-tuning dataset for a flashcard program
for ESL learners.

## Pipeline

Three subcommands chained via JSONL files:

```
text  --extract-->  rows  --augment-->  enriched rows  --rewrite-->  dataset
```

- `extract` — cleans the text (URLs, figure captions, hashtags, whitespace),
  splits it into sentences, and picks one target word per sentence: the
  lemmatizable word whose root is least frequent in the embedded lemma
  database. Emits one row per sentence with a context phrase.
- `augment` — looks each root word up in a Yomitan server, attaching IPA
  pronunciation and dictionary meanings (Cambridge, New Oxford, Wiktionary).
  Rows with no meanings are dropped.
- `rewrite` — sends each row to an LLM that writes a Simple-English flashcard
  definition and a quality label, in strict input order, resumable.

## Install

Requires Go 1.26.5. No external dependencies.

```bash
go build -trimpath -o shisui .
```

## Quick start

```bash
./shisui extract < book.txt > rows.jsonl
./shisui augment < rows.jsonl > enriched.jsonl
./shisui rewrite --api-url http://localhost:8080/v1 \
  --api-key "$SHISUI_API_KEY" -o dataset.jsonl < enriched.jsonl
```

Each row of the final dataset looks like:

```json
{"target_word":"running","root_word":"run","phrase":"The children had to run to keep up with their father.","ipa":"rʌn","meanings":["to move along faster than walking","to operate or function"],"simple_meaning":"to move quickly on your feet","quality":"good"}
```

The `_key` field written during `rewrite` is a sha256 of
`target_word + "\x00" + phrase` used only for resume dedup. Drop it for the
final dataset, e.g. `jq 'del(._key)'`.

## Subcommands

All tools read from stdin (or a positional file argument) and write to stdout
unless `-o PATH` is given.

```
shisui extract  [--lemma PATH] [--min-words N] [--max-words N] [--force] [-o PATH] <input.txt
shisui augment  [--yomitan-url URL] [--yomitan-timeout DUR] [--concurrency N] [--force] [-o PATH] <input.jsonl
shisui rewrite  --api-url URL [--api-key KEY] [--model NAME] [--concurrency N] [--resume] [--force] -o PATH <input.jsonl
```

All three subcommands refuse to overwrite an existing `-o` file unless
`--force` is given. `rewrite --resume` appends to an existing output and
skips rows already present (keyed by a sha256 of `target_word + phrase`);
`--force` and `--resume` are mutually exclusive.

### extract

| Flag | Default | Description |
|------|---------|-------------|
| `--lemma` / `-l` | embedded `lemma.en.txt` | custom lemma file |
| `--min-words` | `10` | drop phrases with fewer words |
| `--max-words` | `80` | truncate longer phrases, centered on the target |
| `--force` | `false` | overwrite an existing output file |
| `--output` / `-o` | stdout | output file |

### augment

| Flag | Default | Description |
|------|---------|-------------|
| `--yomitan-url` | `http://127.0.0.1:19633` | Yomitan server base URL |
| `--yomitan-timeout` | `5s` | per-request timeout |
| `--concurrency` | `4` | parallel requests |
| `--force` | `false` | overwrite an existing output file |
| `--output` / `-o` | stdout | output file |

### rewrite

| Flag | Default | Description |
|------|---------|-------------|
| `--api-url` | required | OpenAI-compatible `/chat/completions` base URL |
| `--api-key` | env `SHISUI_API_KEY` | API key (flag wins) |
| `--model` | `gpt-4o-mini` | model name |
| `--concurrency` | `4` | parallel workers |
| `--resume` | `false` | skip rows already present in the output file |
| `--force` | `false` | overwrite an existing output file (mutually exclusive with `--resume`) |
| `--output` / `-o` | required | output file |

## Requirements

- **Yomitan** at `http://127.0.0.1:19633` for `augment`, with the Cambridge,
  New Oxford American, and kty-en-en dictionaries enabled.
- **espeak-ng** (optional) — IPA fallback when Yomitan returns no IPA reading.
- An **OpenAI-compatible** chat completions endpoint for `rewrite`.

## Development

```bash
make build        # go build -trimpath -o shisui .
make fmt          # gofmt check
make vet          # go vet ./...
make staticcheck  # staticcheck, checks = ["all"]
make test         # go test ./... -count=1
make check        # fmt + vet + staticcheck + test
```

Smoke test:

```bash
cat testdata/lorem.txt | ./shisui extract --min-words=6 -o /tmp/extract_out.jsonl
```

## Credits

The embedded lemma database `lemma.en.txt` is the **En Lemma Database** by
Lin Wei, compiled from the 100M+ words of the British National Corpus (BNC)
and Yasumasa Someya's lemma list, and distributed for research and
educational use:

- https://github.com/skywind3000/lemma.en

Dictionary data comes from Yomitan dictionaries (Cambridge, New Oxford
American, and kty-en-en/Wiktionary entries) served by the Yomitan browser
extension API.
