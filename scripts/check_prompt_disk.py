"""Edge cases of the disk prompt tier on the real model (qwen38_server.Engine), in one process.

  check_prompt_disk.py N DIR -- <server flags>

  alternating   A, B, then A's follow-up again: served from disk, same text and logits as the in-memory restore
  stale_system  a disk restore must drop the in-memory snapshots: restoring an old "system" snapshot onto the KV rows
                the load wrote gives wrong output (the check also reproduces that with the old behaviour)
  corrupt       a truncated kv.bin: the load fails, the request is prefilled cold and succeeds, the entry is dropped
  save_failure  a cache folder that cannot be written: the request succeeds, no partial folder is left
  budget        entries beyond PROMPT_DISK_GB are evicted, least recently used first

Prints one PASS / FAIL line per check and exits 1 on any failure."""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bench_prompt_disk import prompts, run  # noqa: E402

results = []


def check(name, ok, detail=""):
    results.append((name, bool(ok), detail))
    print(("PASS " if ok else "FAIL ") + name + (f"  [{detail}]" if detail else ""), flush=True)


def main():
    n, out = int(sys.argv[1]), Path(sys.argv[2])
    flags = sys.argv[sys.argv.index("--") + 1:]
    os.environ["PROMPT_DISK_DIR"] = str(out / "cache")
    os.environ.setdefault("PROMPT_DISK_SAVE_GAP", "1024")
    import qwen38_server as S
    e = S.Engine(S.parse(flags))
    loads = []
    load = e.disk.load
    e.disk.load = lambda eng, name: (loads.append(name), load(eng, name))[1]

    A1, A2, _ = prompts(e, n, " A", 0)
    B1, B2, _ = prompts(e, n, " B", 40000)
    C1, C2, _ = prompts(e, n, " C", 80000)
    # same system prompt as C, different document: the prompt that finds C's "system" snapshot
    D1, _, _ = prompts(e, n, " C", 120000)

    # ---- alternating conversations ----
    run(e, A1)
    ref, ref_logits = run(e, A2)                       # in-memory restore of A's turn snapshot
    check("alternating: A's follow-up is served from memory", ref["reused"] > n and not loads, f"reused {ref['reused']}")
    run(e, B1)
    got, got_logits = run(e, A2)                       # B overwrote the live KV rows: A comes from disk
    check("alternating: A's follow-up after B is restored from disk", len(loads) == 1 and got["reused"] == ref["reused"],
          f"loads {len(loads)}, reused {got['reused']}")
    check("alternating: same text as the in-memory restore", got["text"] == ref["text"])
    diff = float(np.abs(got_logits.astype(np.float32) - ref_logits.astype(np.float32)).max())
    check("alternating: same logits as the in-memory restore", diff == 0.0, f"max diff {diff}")

    # ---- stale in-memory snapshots after a disk restore ----
    loads.clear()
    run(e, C1)
    stale = dict(e.snaps)                              # "system" and "turn" of C
    run(e, A2)                                         # C overwrote the KV rows: A comes from disk again
    check("stale_system: A restored from disk after C", len(loads) == 1)
    check("stale_system: in-memory snapshots dropped by the restore", set(e.snaps) == {"turn"}, str(sorted(e.snaps)))
    cold, _ = run(e, D1)                               # starts with C's system prompt
    check("stale_system: a prompt with C's system prefix is prefilled cold, not restored onto A's KV",
          cold["reused"] == 0, f"reused {cold['reused']}")
    run(e, A2)                                         # back to A's KV
    e.snaps = {**stale, "turn": e.snaps["turn"]}       # what the code did before the fix: only "turn" replaced
    lookup, e.disk.lookup = e.disk.lookup, lambda *a, **k: None   # D1 was saved by the cold run: keep disk out of it
    bad, _ = run(e, D1)
    e.disk.lookup = lookup
    check("stale_system: repro of the old behaviour restores C's system snapshot", bad["reused"] > 0,
          f"reused {bad['reused']}")
    check("stale_system: and its output differs from the cold prefill", bad["text"] != cold["text"])
    e.snaps = {}

    # ---- a corrupt entry ----
    run(e, B1)                                         # so memory does not offer A
    entry = e.disk.lookup(A2, 0)
    kv = e.disk.root / entry / "kv.bin"
    kv.write_bytes(kv.read_bytes()[: kv.stat().st_size // 2])
    loads.clear()
    res, _ = run(e, A2)
    check("corrupt: the request still succeeds", bool(res["text"].strip()) and res["reused"] == 0,
          f"reused {res['reused']}")
    check("corrupt: the bad entry was dropped", entry not in e.disk.entries and not (e.disk.root / entry).exists())

    # ---- a save that fails ----
    E1, _, _ = prompts(e, n, " E", 20000)
    before = set(os.listdir(e.disk.root))
    os.chmod(e.disk.root, 0o500)
    try:
        res, _ = run(e, E1)
    finally:
        os.chmod(e.disk.root, 0o700)
    check("save_failure: the request succeeds", bool(res["text"].strip()))
    check("save_failure: no partial folder is left", not list(e.disk.root.glob("*.tmp")))

    # ---- budget ----
    sizes = [v["bytes"] for v in e.disk.entries.values()]
    e.disk.budget = int(max(sizes) * 1.5)
    for i, skip in enumerate((30000, 50000, 70000)):
        run(e, prompts(e, n, f" F{i}", skip)[0])
    total = e.disk.total()
    check("budget: total stays within the limit", total <= e.disk.budget or len(e.disk.entries) == 1,
          f"{total / 1e9:.2f} of {e.disk.budget / 1e9:.2f} GB, {len(e.disk.entries)} entries")
    on_disk = sorted(p.name for p in e.disk.root.iterdir())
    check("budget: folders on disk match the index", on_disk == sorted(e.disk.entries), f"{len(on_disk)} folders")

    bad = [r for r in results if not r[1]]
    (out / "check.json").write_text(json.dumps(results, indent=1))
    print(f"{len(results) - len(bad)} passed, {len(bad)} failed", flush=True)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
