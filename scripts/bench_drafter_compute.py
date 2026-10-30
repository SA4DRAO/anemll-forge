"""Decode and prefill benchmark for one drafter placement (ANE / GPU / CPU) on the release 0.2.1 build.

Follows the release notes' decode matrix: one cold prompt for prefill, then decode tok/s per content type (code, prose,
explain, JSON, thinking) with greedy sampling, a 256-token cap and the median of three repeats, using the server's own
/health timers. The drafter placement is COREAI_DRAFTER_COMPUTE; everything else stays fixed. The server is started and
stopped through scripts/qwen38_server.sh on its own port and log, so a running production server is not touched.

    python scripts/bench_drafter_compute.py --compute gpu --out results/drafter_gpu.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import subprocess
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CASES = {
    "code": ("Write a Python class implementing an LRU cache with get and put, followed by unit tests.", False),
    "prose": ("Write a short story of about 400 words about a lighthouse keeper who finds a message in a bottle.", False),
    "explain": ("Explain how TCP congestion control works, step by step, for someone new to networking.", False),
    "json": ("Return only a JSON array of 12 objects with fields id, name, email and role describing fictional "
             "employees.", False),
    "thinking": ("How many positive integers below 100 are divisible by 3 or by 5 but not by both? Think carefully.", True),
}
FILLER = ("The release notes describe measured latency, memory use and numerical error for each context size. "
          "A careful test repeats every run and records the median. ")


def call(port, path, payload=None, timeout=3600):
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", data=None if payload is None else json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def chat(port, content, thinking, max_tokens):
    return call(port, "/v1/chat/completions", {"model": "qwen38-27b-ane", "temperature": 0, "max_tokens": max_tokens,
                "stream": False, "chat_template_kwargs": {"enable_thinking": thinking},
                "messages": [{"role": "user", "content": content}]})


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--compute", required=True, choices=("ane", "gpu", "cpu"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--port", type=int, default=8788)
    ap.add_argument("--bundle", type=Path, default=Path.home() / "Models/anemll-forge-qwen3.8-27B-192K")
    ap.add_argument("--build", type=Path, required=True, help="capped build directory under ~/.anemll-forge/builds")
    ap.add_argument("--ctx", default="128K")
    ap.add_argument("--prefill-repeats", type=int, default=130, help="filler repeats; about 60 tokens each")
    ap.add_argument("--reps", type=int, default=3)
    a = ap.parse_args()
    if a.out.exists():
        raise SystemExit(f"{a.out} exists; keeping the earlier record")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    state = a.out.with_suffix(".state")
    state.mkdir(exist_ok=True)
    drafter = a.bundle / "drafter/all3"
    env = {**os.environ, "COREAI_DRAFTER_COMPUTE": a.compute, "DRAFT_GAP_MS": "3", "PI_SYNC": "0", "PORT": str(a.port),
           "MODEL": str(a.bundle / "model"), "BUILD": str(a.build), "CTX": a.ctx,
           "DRAFT": str(drafter / "dflash2_lut4_gptq_nohead.aimodel"), "DRAFTER": str(drafter),
           "LOG": str(state / "server.log"), "PIDFILE": str(state / "server.pid")}
    sh = ["bash", str(ROOT / "scripts/qwen38_server.sh")]
    rec = {"compute": a.compute, "ctx": a.ctx, "build": str(a.build), "draft_gap_ms": 3, "max_tokens": 256,
           "temperature": 0, "started": time.strftime("%F %T"), "cases": {}}
    t0 = time.time()
    try:
        subprocess.run(sh + ["start"], env=env, check=True, timeout=3 * 3600)
        rec["load_wall_s"] = time.time() - t0
        h = call(a.port, "/health")
        rec["model_build"] = h.get("model_build")
        chat(a.port, "Reply with OK.", False, 4)  # warm-up
        long = FILLER * a.prefill_repeats
        out = chat(a.port, long + "\nSummarize the above in one sentence.", False, 8)
        h = call(a.port, "/health")
        rec["prefill"] = {"prompt_tokens": out["usage"]["prompt_tokens"], **{k: h["prefill"][k] for k in ("tokens", "seconds", "tokens_per_second")}}
        print(f"[{a.compute}] prefill {rec['prefill']['tokens']} tok at {rec['prefill']['tokens_per_second']:.0f} tok/s", flush=True)
        for name, (prompt, thinking) in CASES.items():
            reps = []
            for rep in range(a.reps + 1):  # the first repeat also warms the prompt cache
                out = chat(a.port, prompt, thinking, 256)
                d = call(a.port, "/health")["decode"]
                text = out["choices"][0]["message"].get("content") or ""
                reps.append({"tps": d["tokens_per_second"], "tokens": d["tokens"], "per_call": d["tokens_per_call"],
                             "accept": d["draft_accept_percent"], "phase_s": d["phase_seconds"], "cycles": d["cycles"],
                             "sha": hashlib.sha256((text + json.dumps(out["choices"][0]["message"].get("reasoning_content"))).encode()).hexdigest()[:16]})
            use = reps[1:]
            rec["cases"][name] = {"tps_median": statistics.median(r["tps"] for r in use), "tps": [r["tps"] for r in use],
                                  "per_call": statistics.median(r["per_call"] for r in use),
                                  "accept": statistics.median(r["accept"] for r in use),
                                  "ms_draft_per_cycle": 1000 * statistics.median(r["phase_s"]["draft"] / r["cycles"] for r in use),
                                  "ms_verify_per_cycle": 1000 * statistics.median(r["phase_s"]["verify"] / r["cycles"] for r in use),
                                  "sha": [r["sha"] for r in reps]}
            c = rec["cases"][name]
            print(f"[{a.compute}] {name:9s} {c['tps_median']:5.1f} tok/s  draft {c['ms_draft_per_cycle']:.1f} ms  verify "
                  f"{c['ms_verify_per_cycle']:.1f} ms  accept {c['accept']:.0f}%", flush=True)
        rec["status"] = "ok"
    except BaseException as e:
        rec.update(status="error", error=f"{type(e).__name__}: {e}")
        raise
    finally:
        subprocess.run(sh + ["stop"], env=env, timeout=120)
        rec["elapsed_s"] = time.time() - t0
        a.out.write_text(json.dumps(rec, indent=1) + "\n")


if __name__ == "__main__":
    main()
