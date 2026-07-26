# Publishing an index

A leann.cpp index is small because the dense document vectors are thrown away.
That is what makes it shareable: a corpus whose dense index would be hundreds
of gigabytes fits in a repository somebody can actually download.

This is the end-to-end path from a directory of text to a published artifact.
The CLI contract for `pull` and `verify` and the `LEANNMF1` grammar are in
[CLI.md](CLI.md); the format itself is in [../DESIGN.md](../DESIGN.md).

## 1. Chunk

`leann build --docs FILE` reads one document per LF-delimited line with no
escaping, so chunks must not contain newlines.

```bash
python3 scripts/chunk_corpus.py \
  --input path/to/corpus --output corpus.txt --max-bytes 1200
```

The budget is in bytes, not tokens, deliberately: counting tokens would mean
loading a GGUF vocabulary and pinning the chunker to one model. The byte
budget instead gives a guarantee that holds for every tokenizer — a chunk of
N bytes can never produce more than N tokens, because no token is shorter than
one byte. So `--ctx >= max_chunk_bytes` always builds. The script prints
`safe_ctx_tokens` for exactly this.

That bound is worst-case, not typical. English prose runs near 4 bytes per
token, so a much smaller `--ctx` is usually fine — but the embedder **throws**
on an over-length document rather than truncating, so a corpus that has not
been measured should use the safe number. A `--document-prefix` is prepended
before tokenization and eats into the same budget.

## 2. Build with a descriptor

```bash
leann build --docs corpus.txt --index corpus \
  --embedder llama --model work/models/nomic-embed-text-v1.5.Q4_K_M.gguf \
  --ctx 1280 \
  --model-source 'hf:nomic-ai/nomic-embed-text-v1.5-GGUF/nomic-embed-text-v1.5.Q4_K_M.gguf' \
  --document-prefix 'search_document: ' \
  --query-prefix 'search_query: ' \
  --card license=apache-2.0 \
  --card corpus='what this corpus is' \
  --card chunker='scripts/chunk_corpus.py --max-bytes 1200'
```

`--model-source` is the string a downloader needs to find the weights;
everything else about the model — its SHA-256, size, pooling mode, context
budget — is read from the file itself and recorded automatically.

The prefixes matter for instruction-tuned embedding models. nomic-embed
expects `search_document: ` on stored text and `search_query: ` on queries,
and getting that wrong degrades recall without erroring. Recording them in the
index means a downloader cannot get it wrong: `Index` applies them.

If the corpus text **already** carries prefixes — as
`scripts/prepare_beir_nq_scale.py` produces for the benchmark harness — leave
both options empty. Prefixing twice is silent, not an error.

## 3. Pack

```bash
python3 scripts/publish_hf_index.py pack \
  --index corpus --output upload --repo OWNER/NAME --leann ./leann \
  --license apache-2.0 --tag documentation --language en
```

This writes an upload-ready directory: the two artifacts (hard-linked, not
copied, so a multi-gigabyte index does not double on disk), a `leann.manifest`
in `LEANNMF1` format, a `.gitattributes` marking `*.leann` and `*.docs` as LFS,
and a README model card. It shells out to the `leann` binary for `stats
--format json`, so the card can never disagree with the artifact.

`pack` opens no socket and its output is byte-identical across runs.

## 4. Push

```bash
HF_TOKEN=... python3 scripts/publish_hf_index.py push \
  --directory upload --repo OWNER/NAME --create-repo --yes
```

`push` is gated twice: an explicit `--yes` and a token from `--token-file` or
`HF_TOKEN`, both checked before anything opens a socket. It uploads only the
files `pack` produced, never whatever else is sitting in the directory.

`push` uses plain HTTPS through the standard library — whoami, optional repo
create, preupload, the git-lfs batch endpoint, then a commit. It has **not**
been exercised against the live API from this repository, because there is no
token here. If it fails, `hf upload` from `huggingface_hub` does the same job
against the same directory.

## 5. Consume

```bash
leann pull hf:OWNER/NAME
# run the printed curl line to fetch leann.manifest, then:
leann pull hf:OWNER/NAME --manifest leann.manifest
# run the printed curl lines, then:
leann verify --index corpus --manifest leann.manifest
leann search --index corpus --query 'a question' \
  --embedder llama --model <the GGUF the card names> --ctx 1280
```

## What a published index does not establish

- The manifest is unsigned. There is no trust root and no revocation.
- A matching digest proves the bytes are the ones the publisher recorded. It
  says nothing about whether the publisher is trustworthy, and nothing about
  retrieval quality.
- The recorded model SHA-256 is metadata. Nothing checks that the model in use
  hashes to it; the fingerprint guardrail is a separate, weaker check.
- A digest pins one build, not one corpus. Embeddings are not bit-identical
  across backends or batch shapes, so the same chunks rebuilt on other
  hardware, or with a different `--batch-tokens` or `--parallel`, produce a
  valid index with different bytes. Two publishers of the same corpus will not
  agree on digests, and that is expected rather than a sign of tampering.

## Corpus recipes

The seed index measured in [../VALIDATION.md](../VALIDATION.md) is the
llama.cpp repository's own markdown. Larger corpora follow the same four
steps; the only thing that changes is preparation and the wall-clock.

**Technical documentation** (what the seed index does). Point
`chunk_corpus.py` at a checkout. 223 files and 1.2 MB of text produced 1304
chunks and a 133 KB index in 24.9 s on an Apple M4 Pro.

**arXiv abstracts.** The Kaggle arXiv metadata dump is one JSON object per
line; extract `abstract` into a line-oriented file with a short script, then
chunk with `--max-bytes 1200`. Abstracts are already near chunk size, so the
chunker mostly passes them through. At roughly 18k tokens/s measured on Metal,
1M abstracts of ~200 tokens each is on the order of three hours of embedding.

**Wikipedia.** Use a `wikiextractor` dump or the Hugging Face `wikimedia/
wikipedia` parquet exports, write plain text per article, then chunk. This is
the case the storage argument is really about, and also the one that is not a
laptop-afternoon job: a full language edition is millions of chunks and, at
the measured throughput, days of embedding on one machine. Build it where the
GPU is, not where you happen to be sitting.

Note that build memory is still `O(N * dimension)` — all dense embeddings are
held in RAM during the build even though none are written. That, not disk, is
the binding constraint on how large a corpus one machine can index.
