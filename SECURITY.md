# Security policy

## Status of this project

`leann.cpp` is a **research spike**, not a hardened production system.
`README.md`'s "Honest spike boundaries" section is the authoritative statement
of what it does not do. Please calibrate expectations accordingly: there is no
security release train, no backported fixes, and no supported-version matrix.

Only the current `main` is supported.

## Reporting a vulnerability

Please report privately, not as a public issue:

**[Open a private security advisory](https://github.com/toreleon/leann.cpp/security/advisories/new)**

Include what you would put in a bug report — the exact commands, the artifact
or input that triggers it, and the platform and compiler. A reproducer using
the deterministic `--embedder hash` backend is far easier to act on than one
requiring a specific GGUF model.

There is no bounty, and no guaranteed response time. This is a spike maintained
by one person.

## What is most likely to be a real finding

The genuinely interesting attack surface is **parsing untrusted artifacts**.
The formats carry declared offsets, lengths, and counts that a malicious file
can lie about:

- `.leann` compact index and `.docs` document store — see `DESIGN.md` for the
  layouts.
- `LEANNBC2` streamed embedding caches.
- `LEANN_GT1` ground-truth files.

A crafted file that causes an out-of-bounds read, an integer overflow feeding
an allocation, or an unbounded allocation is in scope and worth reporting. The
code defends against this deliberately — index load verifies SHA-256 over the
whole compact index, document chunks are CRC32C-checked on read, and non-finite
values are rejected before they can reach an integer conversion or a sort —
but those defences have not been fuzzed, and absence of a known bug is not
evidence of absence.

Also in scope:

- Path handling in artifact publication, including the `.lock`, `.tmp.*`, and
  `.bak.*` files, where a symlink or a race could matter.
- Anything that lets a document store's contents escape into a context that
  treats it as trusted.

## What is not in scope

- **Checksums are integrity, not authenticity.** SHA-256 and CRC32C detect
  corruption and accidental mismatch. They are not signatures — anyone who can
  rewrite an artifact can rewrite its digest. "I modified the index and the
  digest still matched after I recomputed it" is the designed behaviour.
- **The embedder fingerprint is a configuration guardrail, not a cryptographic
  identity.** It catches common build/search mismatches. It does not identify
  the GGUF weights, the llama.cpp build, or the compute backend, and it is not
  intended to.
- **Publication is fail-closed, not power-loss atomic.** Losing power mid-build
  can leave a `.tmp.*` or `.bak.*` behind; `leann doctor` exists to report and
  clean that. This is documented, not a vulnerability.
- Denial of service from a deliberately enormous corpus you supplied yourself.
- Vulnerabilities in hnswlib or llama.cpp — please report those upstream. Both
  are pinned by commit in `CMakeLists.txt`.
