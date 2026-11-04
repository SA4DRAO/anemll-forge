# Report: disk tier for Forge's prompt snapshots

Date: 2026-11-02, results updated 2026-11-03. Machine: M6 Mac mini, 32 GB, macOS 27. Build: 0.2.1 at 128K (kv8 cache, DFlash2 drafter, all3).
Branch: `feat/prompt-disk-cache` on SA4DRAO/anemll-forge (five commits on top of upstream `main` 703131e). PR within the fork only: SA4DRAO/anemll-forge#1. Upstream: issue Anemll/anemll-forge#18, no upstream PR.

## Summary
A long prompt that Forge has already read can be restored from the internal SSD in 0.9 to 1.8 seconds instead of being
prefilled again (5 to 13 minutes at 64K to 126K tokens here; the first token comes 1.6 to 3.1 s after the request). The restored state gives identical output.
The feature is off unless `PROMPT_DISK_DIR` is set.

## Problem
Forge keeps two in-memory snapshots per prompt ("system" and "turn"). They hold the Gated DeltaNet states, the position and
the drafter's slot map. The KV rows are not copied: they stay in one live cache and are hidden by position. So:
- a restart loses every snapshot;
- a second conversation overwrites the live KV rows, and going back to the first one prefills it again.

## Design
On the turn snapshot (just before the last assistant header), also write the full prompt state to disk. Before choosing a
prefill path, look for a saved prompt that is a prefix of the new one and is at least 2048 tokens longer than what memory
offers. If found, read it back instead of prefilling.

Files per entry (folder named `<time>-<tokens>-<hash>`):
| file | content | size |
|---|---|---|
| `kv.bin` | KV rows [0, pos) of every attention layer, int8 + scales | about 33 KB per token |
| `deltanet.bin` | DeltaNet / conv states of the snapshot | 101 MB, fixed |
| `ring.bin` | drafter K/V ring | 42 MB, fixed |
| `draft.npz` | drafter slot positions and queued feature rows | 0.4 MB |
| `ids.npy` | the prompt's token ids | 4 bytes per token |
| `meta.json` | position, pending count, fingerprint | small |

Rules:
- saved only when at least 8192 tokens are new compared with what disk already holds for that prompt;
- written to `<name>.tmp` with fsync, then renamed;
- an entry is ignored unless its fingerprint matches: SHA-256 of the build's `manifest.json`, KV dtype, key layout, SHA-256 of the drafter's `main.hash`, ctx;
- a newer entry that extends an older one deletes it; total size is capped by `PROMPT_DISK_GB` (default 16), least recently used first;
- reads use `F_NOCACHE`, so a restore does not fill the page cache;
- a failed save or load is logged; a failed load resets the model and prefills normally.

Environment: `PROMPT_DISK_DIR` (enables it), `PROMPT_DISK_GB`, `PROMPT_DISK_MIN_GAIN` (2048), `PROMPT_DISK_SAVE_GAP` (8192).

Code: `scripts/prompt_disk_cache.py` (new), 3 hooks in `scripts/qwen38_server.py` (create, look up before prefill, save after the turn snapshot),
`scripts/bench_prompt_disk.py` (benchmark).

## Measurements (internal SSD, restore in a new process, files re-cloned with `cp -c` so reads miss the page cache)
Rerun on 2026-11-03 with the reviewed code (draft state as .npz, stale-snapshot fix). One sample per size. The build's context is 128K
(131,072 tokens); the largest prompt is 126,026 tokens so the follow-up (about 20 tokens) and 64 generated tokens fit.
| prompt | cold prefill | save | restore | first token after restore | same as in-memory restore |
|---|---|---|---|---|---|
| 8,026 tok | 28 s | 0.06 s (0.4 GB) | 0.14 s | 0.63 s | text identical, logits diff 0.0 |
| 64,026 tok | 300 s | 0.74 s (2.26 GB) | 0.91 s | 1.58 s (0.50 s from memory) | text identical, logits diff 0.0 |
| 126,026 tok | 785 s | 1.58 s (4.30 GB) | 1.81 s | 3.10 s (0.52 s from memory) | text identical, logits diff 0.0 |

Restore speed is about 2.4 to 2.5 GB/s at 64K and 126K (about 2.7 GB/s on 2026-11-02). The first token after a disk restore comes
0.3 s (8K), 1.1 s (64K) and 2.6 s (126K) later than after an in-memory restore (0.34, 0.50 and 0.52 s). The extra time varies between
runs: the first token at 126K was 2.36 s on 2026-11-02 and 3.10 s on 2026-11-03. The cause is not found.

The 2026-11-02 run (earlier code): cold prefill 33 / 302 / 785 s, save 0.17 / 0.67 / 1.42 s, restore 0.16 / 0.89 / 1.61 s, first token
after the disk restore 0.61 / 1.50 / 2.36 s, logits diff 0.0 at all three sizes.

Edge cases on the real model (`scripts/check_prompt_disk.py`, 6K-token prompts, 15 checks, all pass):
- alternating A, B, A: A's follow-up is restored from disk with the same text and logits (diff 0.0) as the in-memory restore;
- stale snapshots: after a disk restore the in-memory snapshots are dropped. With the old behaviour (reproduced by the check) a
  prompt with an old system prefix restored the old "system" snapshot onto the other conversation's KV rows and gave different output;
- a truncated `kv.bin`: the load fails, the request is prefilled cold and succeeds, the entry is removed;
- a cache folder that cannot be written: the request succeeds, no `.tmp` folder is left;
- the size cap: entries beyond it are evicted, folders on disk match the index.
Host tests: `tests/test_prompt_disk_cache.py` (10 tests). The full repo suite has 3 errors that need `sklearn`, which this venv lacks (quantization tests, unrelated).

How "same output" is judged: the follow-up prompt is run (a) in the process that saved, from the in-memory snapshot, and (b) in a
new process from disk. Logits after the prefill and 64 greedy tokens are compared. A cold prefill of the follow-up is NOT a valid
reference: prefilling in different blocks changes fp16 rounding (logits differ by about 3.4, same top token).

## Limits and open questions
- One machine, one build. Not tried: long idle periods, two requests at once (the engine lock serialises them), a different Mac.
- The 0.3 to 2.6 s extra first-token latency after a disk restore (compared with memory) is not explained; a guess is the first ANE calls after reading 2 to 4 GB.
- Privacy: entries hold the conversation (token ids, and KV rows that can be inverted to text with effort). Off by default; there is no per-client
  switch yet, so it covers every client of the server.
- Every save rewrites the whole prompt. Appending only new rows would cut save cost for long conversations.
- Only the "turn" snapshot is saved, not the "system" one.
- Not covered: the older non-batched model path (`self.v2` false); the server ignores `PROMPT_DISK_DIR` there and logs it.
- The external SSD was not used or measured. It runs at 35 to 39 MB/s (USB 2.0 speed), so a 64K restore from it would take about a minute.

## Before this goes upstream
- Done: tests, docs/SERVER.md section, header comment, edge cases on the real model, CONTRIBUTING.md read, issue Anemll/anemll-forge#18 opened.
- Measure through the HTTP server with several samples and memory peaks (CONTRIBUTING.md asks for sample count and memory).
- Run the benchmark on another Mac if possible.

## Other findings from this work
- Drafter on the GPU instead of the ANE: 16 to 24% slower (`feat/gpu-drafter-bench`, results in `docs/results/drafter_placement/`).
- Cancel when the client disconnects: `feat/cancel-on-disconnect` (pushed to the fork).
- Benchmark traps: `F_NOCACHE` still returns cached pages, so a read of a file just written or just read measures RAM; a fresh APFS clone (`cp -c`) forces a real read.
  An earlier "SSD at 790 MB/s" figure on this machine was this mistake.
- Post-reboot stall (Oct 30): every request hung 100 s and failed with ANE "Request cancelled"; kernel log said
  "schedule: not able to select correct variant for request". Cleared after a tiny Core ML model ran on the ANE. A boot on Nov 1 did not show it
  and no warm-up was needed. Cause unknown. Chunks that loaded fine at 18:09 on Nov 1 failed with nilError at 18:13 and recompiled while internal disk had 11 to 15 GB free; low disk may evict compiled programs (unconfirmed).
- Betelgeuse comparison (llama.cpp fork, MoE experts paged VRAM / host RAM / NVMe) and a macOS port estimate were discussed in the session, not written up here.
