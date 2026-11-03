"""Disk tier for the server's prompt snapshots (the "victim cache").

The in-memory "system" / "turn" snapshots hold only the DeltaNet states; the KV rows stay in the one live cache, so a
restart or a different conversation loses them. This tier writes the full prompt state at a turn cut and reads it back
into a fresh process instead of prefilling again:

  kv.bin        KV rows [0, pos) of every attention layer (about 33 KB per token)
  deltanet.bin  the snapshot's DeltaNet / conv states (fixed size)
  ring.bin      the drafter's K / V ring
  draft.npz     drafter slot positions and queued feature rows
  ids.npy       the prompt's token ids (an entry serves a prompt that starts with them)

Off unless PROMPT_DISK_DIR is set: the files hold the conversation (token ids and its KV rows). Reads use F_NOCACHE,
so a restore does not fill the page cache with a conversation's KV."""
import fcntl
import hashlib
import json
import os
import re
import sys
import shutil
import threading
import time
from pathlib import Path

import numpy as np

F_NOCACHE, F_FULLFSYNC = 48, 51   # macOS fcntl commands
DARWIN = sys.platform == "darwin"
FORMAT = 1
ENTRY = re.compile(r"^\d+-\d+-[0-9a-f]{8}$")   # folder names this cache writes


def file_digest(*paths, head=None):
    """SHA-256 of the files (with `head`, of each file's size and first `head` bytes), for the entry fingerprint."""
    h = hashlib.sha256()
    for p in paths:
        p = Path(p)
        h.update(p.name.encode())
        try:
            with open(p, "rb") as f:
                if head:
                    h.update(str(os.fstat(f.fileno()).st_size).encode())
                h.update(f.read(head) if head else f.read())
        except OSError:
            h.update(b"missing")
    return h.hexdigest()[:16]


def build_digest(d):
    """A compiled build or drafter package: its manifest and every package's main.hash (the compiled program's
    content hash, so it changes with the weights; manifest names alone do not)."""
    d = Path(d)
    files = ([d / "manifest.json"] if (d / "manifest.json").exists() else []) + sorted(d.glob("**/main.hash"))
    return file_digest(*files) if files else file_digest(d)


def _sync(f):
    f.flush()
    os.fsync(f.fileno())
    if DARWIN:   # fsync on macOS does not flush the drive's cache
        try:
            fcntl.fcntl(f.fileno(), F_FULLFSYNC)
        except OSError:
            pass


def _sync_dir(path):
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _write_bytes(path, data):
    with open(path, "wb") as f:
        f.write(data)
        _sync(f)


def _write(path, arrays):
    n = 0
    with open(path, "wb") as f:
        for a in arrays:
            a = np.ascontiguousarray(a)
            f.write(memoryview(a).cast("B"))
            n += a.nbytes
        _sync(f)
    return n


def _read_into(path, arrays):
    """Direct reads into the arrays (live IOSurface buffers for KV and ring)."""
    n = 0
    with open(path, "rb", buffering=0) as f:
        if DARWIN:
            fcntl.fcntl(f.fileno(), F_NOCACHE, 1)
        for a in arrays:
            b = bytearray(a.nbytes)
            mv, got = memoryview(b), 0
            while got < a.nbytes:
                r = f.readinto(mv[got:])
                if not r:
                    raise EOFError(path)
                got += r
            a[...] = np.frombuffer(b, a.dtype).reshape(a.shape)
            n += got
    return n


class PromptDiskCache:
    def __init__(self, root, budget_gb, fingerprint, min_gain=2048, save_gap=8192, log=print):
        self.root, self.budget = Path(root).expanduser(), int(budget_gb * 1e9)
        self.fp, self.min_gain, self.save_gap, self.log = fingerprint, min_gain, save_gap, log
        new = not self.root.exists()
        self.root.mkdir(parents=True, exist_ok=True)
        if new:
            os.chmod(self.root, 0o700)   # a folder we made holds conversations; one the user chose keeps its mode
        self.lock = threading.Lock()
        self.entries = {}   # name -> {"ids": int32 array, "bytes": int, "used": float}
        for d in sorted(self.root.iterdir()):
            if d.suffix == ".tmp":   # a save that never finished (crash, disk full)
                shutil.rmtree(d, ignore_errors=True)
                continue
            try:
                meta = json.loads((d / "meta.json").read_text())
                if meta["fp"] != self.fp or meta["format"] != FORMAT:
                    continue   # another build's entry: left alone
                self.entries[d.name] = {"ids": np.load(d / "ids.npy"), "bytes": meta["bytes"],
                                        "used": (d / "meta.json").stat().st_mtime}
            except Exception:   # unreadable: removed if this cache wrote it, otherwise left alone
                if ENTRY.match(d.name):
                    shutil.rmtree(d, ignore_errors=True)
                continue
        log(f"[disk] prompt cache {self.root}: {len(self.entries)} entries, {self.total() / 1e9:.1f} of "
            f"{budget_gb:g} GB")

    def total(self):
        return sum(e["bytes"] for e in self.entries.values())

    # ---- what the model exposes ----
    @staticmethod
    def _kv(m, pos):
        """Rows [0, pos) of each KV tensor (keys are stored transposed: token is the last axis)."""
        for i, d in enumerate(m.kv):
            for name, (_, a) in d.items():
                t = name.startswith("k") and not name.startswith("ks") and m.keys_t
                yield a[:, :, :pos] if t else a[:, :pos]

    @staticmethod
    def _ring(d):
        return [a for pair in d.ring for a in pair]

    # ---- lookup ----
    def lookup(self, ids, beyond):
        """Longest entry that is a prefix of `ids` and reaches past `beyond` tokens (what memory already offers)."""
        arr, best = np.asarray(ids, np.int32), None
        with self.lock:
            for name, e in self.entries.items():
                n = len(e["ids"])
                if n > beyond + self.min_gain - 1 and n < len(arr) and (best is None or n > len(self.entries[best]["ids"])) \
                        and np.array_equal(arr[:n], e["ids"]):
                    best = name
        return best

    # ---- load ----
    def load(self, eng, name):
        """Reset the model, read the entry in, and return a snapshot dict for Engine._restore."""
        t0 = time.perf_counter()
        d = self.root / name
        m, dr = eng.model, eng.drafter
        meta = json.loads((d / "meta.json").read_text())
        pos = meta["pos"]
        m.reset()
        if dr is not None:
            dr.reset()
        m.fit(pos)
        n = _read_into(d / "kv.bin", list(self._kv(m, pos)))
        m.hi = pos
        states = []
        with open(d / "deltanet.bin", "rb") as f:
            for ch in m.chunks:
                saved = {}
                for key in sorted(ch["curw"]):
                    a = ch["curw"][key]
                    saved[key] = np.frombuffer(f.read(a.nbytes), a.dtype).reshape(a.shape).copy()
                states.append(saved)
        snap = {"ids": self.entries[name]["ids"].tolist(), "pos": pos, "pending": meta["pending"], "ctx": m.ctx,
                "states": states}
        if dr is not None:
            n += _read_into(d / "ring.bin", self._ring(dr))
            with np.load(d / "draft.npz") as z:
                slot_pos = z["slot_pos"]
                pending = list(zip(z["pend_rows"], z["pend_pos"].tolist()))
            dr.slot_pos = slot_pos.copy()       # the ring on disk is as it was at the cut: every slot is valid
            snap["draft"] = (slot_pos, pending)
        os.utime(d / "meta.json")
        with self.lock:
            self.entries[name]["used"] = time.time()
        self.log(f"[disk] loaded {pos} tok ({n / 1e9:.2f} GB) in {time.perf_counter() - t0:.2f}s")
        return snap

    # ---- save ----
    def maybe_save(self, eng, ids, snap):
        """After the turn snapshot is taken: write the prompt state unless disk already holds this prefix closely."""
        ids = np.asarray(ids, np.int32)
        have = max((len(e["ids"]) for e in self.entries.values()
                    if len(e["ids"]) <= len(ids) and np.array_equal(ids[:len(e["ids"])], e["ids"])), default=0)
        if len(ids) - have < self.save_gap:
            return
        size = sum(a.nbytes for a in self._kv(eng.model, snap["pos"])) + \
            sum(v.nbytes for st in snap["states"] for v in st.values()) + \
            (sum(a.nbytes for a in self._ring(eng.drafter)) if eng.drafter is not None else 0)
        if size > self.budget:   # it would evict everything and still not fit
            self.log(f"[disk] not saved: {size / 1e9:.2f} GB is over PROMPT_DISK_GB")
            return
        try:
            self._save(eng, ids, snap)
        except Exception as ex:   # the disk tier never fails a request
            self.log(f"[disk] save failed: {ex!r}")
            for t in self.root.glob("*.tmp"):
                shutil.rmtree(t, ignore_errors=True)

    def _save(self, eng, ids, snap):
        t0 = time.perf_counter()
        m, dr, pos = eng.model, eng.drafter, snap["pos"]
        name = f"{int(time.time())}-{len(ids)}-{hashlib.sha1(ids.tobytes()).hexdigest()[:8]}"
        tmp = self.root / (name + ".tmp")
        shutil.rmtree(tmp, ignore_errors=True)
        tmp.mkdir()
        os.chmod(tmp, 0o700)
        n = _write(tmp / "kv.bin", self._kv(m, pos))
        n += _write(tmp / "deltanet.bin", [st[k] for st in snap["states"] for k in sorted(st)])
        if dr is not None:
            n += _write(tmp / "ring.bin", self._ring(dr))
            slot_pos, pending = snap["draft"]
            rows = np.stack([r for r, _ in pending]) if pending else np.zeros((0, 0), np.float16)
            with open(tmp / "draft.npz", "wb") as f:
                np.savez(f, slot_pos=slot_pos, pend_rows=rows, pend_pos=np.array([p for _, p in pending], np.int64))
                _sync(f)
        with open(tmp / "ids.npy", "wb") as f:
            np.save(f, ids)
            _sync(f)
        _write_bytes(tmp / "meta.json", json.dumps({"format": FORMAT, "fp": self.fp, "pos": pos,
                                                    "pending": snap["pending"], "bytes": n}).encode())
        _sync_dir(tmp)
        os.rename(tmp, self.root / name)
        _sync_dir(self.root)
        with self.lock:
            self.entries[name] = {"ids": ids, "bytes": n, "used": time.time()}
            for old in [k for k, e in self.entries.items() if k != name and len(e["ids"]) < len(ids)
                        and np.array_equal(ids[:len(e["ids"])], e["ids"])]:   # superseded: a prefix of the new entry
                self._drop(old)
            while self.total() > self.budget and len(self.entries) > 1:
                self._drop(min((k for k in self.entries if k != name), key=lambda k: self.entries[k]["used"]))
        self.log(f"[disk] saved {pos} tok ({n / 1e9:.2f} GB) in {time.perf_counter() - t0:.2f}s | "
                 f"{len(self.entries)} entries, {self.total() / 1e9:.1f} GB")

    def _drop(self, name):
        self.entries.pop(name, None)
        shutil.rmtree(self.root / name, ignore_errors=True)
