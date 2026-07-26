<!--
Thanks for contributing. CONTRIBUTING.md has the build paths, test
conventions, and the invariants that will get a PR sent back.

Delete any section that genuinely does not apply.
-->

## What and why

<!-- What changes, and what problem it solves. -->

## Verified

<!--
List what you actually ran, with results. Be specific — "tests pass" is
less useful than the command and its outcome.
-->

- [ ] `ctest --test-dir build --output-on-failure` — all suites pass
- [ ] Sanitizers, if the library or search path changed
      (`core-safety-test persistence-test` under ASan/UBSan)
- [ ] `python3 -m unittest discover -s tests -p 'test_*.py'`, if `scripts/`
      changed
- [ ] No new lines over 80 columns

## Not verified

<!--
Just as important. Which platform, toolchain, or leg did you not exercise?
"Builds on macOS, not tested on Linux" is a useful sentence; silence is not.
-->

## Invariants

<!--
Tick anything this PR touches. Each one is defended by a test, and changing
one deliberately means updating DESIGN.md and VALIDATION.md in this PR.
-->

- [ ] On-disk format (magic or version) — `DESIGN.md` tables updated
- [ ] CLI text output — confirmed byte-identical, or the change is intended
      and documented
- [ ] An exception message — note it, a test likely matches its substring
- [ ] A new CLI option — has a `command_specs` entry for every command that
      reads it
- [ ] A new test suite — registered in **both** `CMakeLists.txt` and the
      `Makefile`
- [ ] `include/leann/leann.h` — still valid C11, `docs/C_API.md` updated
- [ ] None of the above

## Claims

<!--
If this PR states a performance, recall, or storage number anywhere — in
code comments, docs, or this description — point at the measurement behind
it. Unmeasured claims will be asked to come out.
-->

- [ ] This PR makes no performance, recall, or storage claim
- [ ] It does, and the measurement is recorded in `VALIDATION.md`
