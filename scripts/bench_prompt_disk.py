"""Benchmark the server's disk prompt tier: cold prefill + save in one process, restore from disk in a new one.

  bench_prompt_disk.py cold N DIR   prefill an N-token prompt (the turn snapshot is saved to DIR), then a follow-up
                                    prompt served from the in-memory snapshot (the reference)
  bench_prompt_disk.py warm N DIR   new process: the same follow-up prompt, served from disk; compares with the reference

Both use qwen38_server.Engine with PROMPT_DISK_DIR=DIR, so this exercises the code the server runs. The entry is
re-cloned (`cp -c`) before the restore: a fresh APFS clone is not in the page cache, so the read is a real disk read.
Pass the server's own flags after `--` (build, bundle, draft, ctx)."""
import glob
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
GEN = 64


def prompts(e, n, tag="", skip=0):
    """tag changes the system prompt, skip rotates the document: two tags give two unrelated conversations."""
    tok = lambda s: e.tok.encode(s, add_special_tokens=False)  # noqa: E731
    files = [REPO / "README.md"] + sorted(Path(p) for p in glob.glob(str(REPO / "docs/**/*.md"), recursive=True))
    base = tok("\n\n".join(f.read_text(errors="replace") for f in files))
    base = base[skip:] + base[:skip]
    body, k = [], 0
    while len(body) < n:   # tile the repo docs, each copy introduced by its own heading, up to n tokens
        body += tok(f"\n\n## Part {k}\n\n") + [t for t in base if t != e.im_start]
        k += 1
    body = body[:n]
    head = tok(f"<|im_start|>system\nYou are a careful assistant{tag}.<|im_end|>\n<|im_start|>user\n")
    ask = tok("\nSummarize the above in one sentence.<|im_end|>\n<|im_start|>assistant\n")
    first = head + body + ask
    cut = max(i for i, t in enumerate(first) if t == e.im_start)   # the assistant header: the turn snapshot's cut
    second = first[:cut] + tok("<|im_start|>assistant\nOK.<|im_end|>\n<|im_start|>user\nWhat next?<|im_end|>\n"
                               "<|im_start|>assistant\n")
    return first, second, cut


def run(e, ids):
    box, orig = {}, e.prefill

    def prefill(*a, **k):
        t = time.perf_counter()
        logits, reused = orig(*a, **k)
        box.update(logits=np.array(logits), reused=reused, prefill_s=time.perf_counter() - t)
        return logits, reused

    e.prefill = prefill
    t, first = time.perf_counter(), []
    text, finish, n, reused, dt = e.generate(ids, GEN, 0.0, 1.0, 0, [], None,
                                             lambda s: first.append(time.perf_counter() - t) if not first else None,
                                             presence=0.0, dry=None, think_budget=0, min_p=0.0)
    e.prefill = orig
    return {"text": text, "n": n, "reused": reused, "total_s": time.perf_counter() - t,
            "prefill_s": box["prefill_s"], "first_text_s": first[0] if first else None}, box["logits"]


def main():
    mode, n, out = sys.argv[1], int(sys.argv[2]), Path(sys.argv[3])
    flags = sys.argv[sys.argv.index("--") + 1:]
    if mode == "cold" and (out / "cache").exists() and any((out / "cache").iterdir()):
        sys.exit(f"{out / 'cache'} already holds entries: a cold run there would restore instead of prefilling")
    os.environ["PROMPT_DISK_DIR"] = str(out / "cache")
    import qwen38_server as S
    t = time.perf_counter()
    e = S.Engine(S.parse(flags))
    load_s = time.perf_counter() - t
    first, second, cut = prompts(e, n)
    rep = {"prompt_tokens": len(first), "cut": cut, "follow_up_tokens": len(second), "engine_load_s": load_s}
    if mode == "cold":
        saves, orig = [], e.disk._save

        def timed(*a, **k):
            t0 = time.perf_counter()
            orig(*a, **k)
            saves.append(time.perf_counter() - t0)

        e.disk._save = timed
        cold, _ = run(e, first)
        mem, logits = run(e, second)
        np.save(out / "ref_logits.npy", logits)
        rep.update(cold=cold, save_s=saves, in_memory_follow_up=mem)
        (out / "cold.json").write_text(json.dumps(rep, indent=1))
    else:
        for d in (out / "cache").iterdir():
            for f in d.glob("*.bin"):   # fresh clones: not in the page cache
                f.rename(f.with_name(f.name + ".old"))
                subprocess.run(["cp", "-c", f.with_name(f.name + ".old"), f], check=True)
                f.with_name(f.name + ".old").unlink()
        disk, logits = run(e, second)
        ref = json.loads((out / "cold.json").read_text())
        rl = np.load(out / "ref_logits.npy").astype(np.float32)
        rep.update(disk_follow_up=disk, same_text=disk["text"] == ref["in_memory_follow_up"]["text"],
                   logits_max_abs_diff=float(np.abs(logits.astype(np.float32) - rl).max()),
                   same_top_token=int(logits.argmax()) == int(rl.argmax()))
        (out / "warm.json").write_text(json.dumps(rep, indent=1))
    print("REPORT " + json.dumps({k: v for k, v in rep.items() if k not in ("cold", "in_memory_follow_up", "disk_follow_up")}), flush=True)
    for k in ("cold", "in_memory_follow_up", "disk_follow_up"):
        if k in rep:
            print(f"{k.upper()} " + json.dumps({a: b for a, b in rep[k].items() if a != "text"}), flush=True)
    if mode == "cold":
        print("SAVE_S", rep["save_s"], flush=True)


if __name__ == "__main__":
    main()
