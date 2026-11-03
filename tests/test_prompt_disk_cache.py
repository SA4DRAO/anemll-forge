"""Disk tier for prompt snapshots (prompt_disk_cache.PromptDiskCache). Host-only: a fake model whose KV / DeltaNet
state are numpy arrays and a fake drafter with a ring and pending rows. Checks the save -> load round trip, that an
entry with another fingerprint is ignored, that a longer entry supersedes its prefix, the budget eviction, and that
unfinished saves are removed."""
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from prompt_disk_cache import PromptDiskCache  # noqa: E402

FP = {"build": "b", "kv": "int8", "keys_t": True, "drafter": "d", "ctx": 64}
QUIET = lambda *a, **k: None  # noqa: E731


class FakeModel:
    def __init__(self, ctx=64):
        self.ctx, self.keys_t, self.hi = ctx, True, 0
        # one attention layer: keys transposed (token is the last axis), values and scales token-major
        self.kv = [{"k": (None, np.zeros((2, 4, ctx), np.int8)), "v": (None, np.zeros((2, ctx, 4), np.int8)),
                    "ks": (None, np.zeros((2, ctx), np.float16))}]
        self.chunks = [{"curw": {"rec0": np.zeros((3, 3), np.float16), "conv0": np.zeros((2, 2), np.float16)}}]

    def reset(self):
        for d in self.kv:
            for _, a in d.values():
                a[...] = 0
        self.hi = 0

    def fit(self, pos):
        assert pos <= self.ctx


class FakeDrafter:
    def __init__(self, W=8):
        self.ring = [(np.zeros((W, 4), np.float16), np.zeros((W, 4), np.float16))]
        self.slot_pos, self.pending = np.full(W, -1, np.int64), []

    def reset(self):
        for k, v in self.ring:
            k[...] = 0
            v[...] = 0
        self.slot_pos[...] = -1
        self.pending = []


class Eng:
    def __init__(self):
        self.model, self.drafter = FakeModel(), FakeDrafter()

    def fill(self, seed, pos):
        rng = np.random.default_rng(seed)
        m, dr = self.model, self.drafter
        m.reset()
        dr.reset()
        for d in m.kv:
            for name, (_, a) in d.items():
                if name == "k":
                    a[:, :, :pos] = rng.integers(-100, 100, a[:, :, :pos].shape)
                else:
                    a[:, :pos] = rng.integers(-100, 100, a[:, :pos].shape)
        dr.ring[0][0][...] = rng.random(dr.ring[0][0].shape)
        dr.slot_pos[...] = np.arange(8) + pos - 8
        pend = [(rng.random(4).astype(np.float16), pos - 2 + i) for i in range(2)]
        dr.pending = list(pend)
        states = [{k: rng.random(v.shape).astype(np.float16) for k, v in m.chunks[0]["curw"].items()}]
        return {"pos": pos, "pending": 2, "states": states, "draft": (dr.slot_pos.copy(), list(pend))}


def kv_state(eng, pos):
    return [(n, (a[:, :, :pos] if n == "k" else a[:, :pos]).copy()) for d in eng.model.kv for n, (_, a) in d.items()]


class PromptDiskCacheTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name) / "cache"
        self.addCleanup(self.tmp.cleanup)

    def cache(self, fp=FP, gb=1.0, **kw):
        return PromptDiskCache(self.root, gb, fp, min_gain=4, save_gap=8, log=QUIET, **kw)

    def test_round_trip_restores_the_saved_state(self):
        eng, ids = Eng(), np.arange(40, dtype=np.int32)
        snap = eng.fill(1, 40)
        before = kv_state(eng, 40)
        ring, pend = eng.drafter.ring[0][0].copy(), list(eng.drafter.pending)
        c = self.cache()
        c.maybe_save(eng, ids, snap)
        self.assertEqual(len(c.entries), 1)

        new = Eng()                                   # a fresh process: nothing in memory
        c2 = self.cache()
        name = c2.lookup(np.concatenate([ids, np.arange(100, 110, dtype=np.int32)]), 0)
        self.assertIsNotNone(name)
        got = c2.load(new, name)
        self.assertEqual(got["ids"], ids.tolist())
        self.assertEqual(new.model.hi, 40)
        for (n, a), (_, b) in zip(before, kv_state(new, 40)):
            self.assertTrue(np.array_equal(a, b), n)
        self.assertTrue(np.array_equal(ring, new.drafter.ring[0][0]))
        for (r1, p1), (r2, p2) in zip(pend, got["draft"][1]):
            self.assertTrue(np.array_equal(r1, r2))
            self.assertEqual(p1, p2)
        for k, v in snap["states"][0].items():
            self.assertTrue(np.array_equal(v, got["states"][0][k]), k)

    def test_lookup_needs_a_prefix_with_enough_gain(self):
        eng, ids = Eng(), np.arange(40, dtype=np.int32)
        c = self.cache()
        c.maybe_save(eng, ids, eng.fill(1, 40))
        longer = np.concatenate([ids, np.arange(50, 60, dtype=np.int32)])
        self.assertIsNotNone(c.lookup(longer, 0))
        self.assertIsNone(c.lookup(longer, 38))       # memory already offers all but 2 tokens: less than min_gain
        other = longer.copy()
        other[3] = 999                                # diverges inside the saved prefix
        self.assertIsNone(c.lookup(other, 0))
        self.assertIsNone(c.lookup(ids, 0))           # an entry must be shorter than the prompt

    def test_another_fingerprint_is_ignored(self):
        eng, ids = Eng(), np.arange(40, dtype=np.int32)
        self.cache().maybe_save(eng, ids, eng.fill(1, 40))
        c = self.cache(fp={**FP, "ctx": 128})
        self.assertEqual(len(c.entries), 0)

    def test_longer_entry_supersedes_its_prefix(self):
        eng = Eng()
        c = self.cache()
        c.maybe_save(eng, np.arange(20, dtype=np.int32), eng.fill(1, 20))
        c.maybe_save(eng, np.arange(40, dtype=np.int32), eng.fill(2, 40))
        self.assertEqual([len(e["ids"]) for e in c.entries.values()], [40])
        self.assertEqual(len([d for d in self.root.iterdir()]), 1)

    def test_save_gap_skips_a_small_extension(self):
        eng = Eng()
        c = self.cache()
        c.maybe_save(eng, np.arange(30, dtype=np.int32), eng.fill(1, 30))
        c.maybe_save(eng, np.arange(34, dtype=np.int32), eng.fill(1, 34))   # 4 new tokens < save_gap 8
        self.assertEqual([len(e["ids"]) for e in c.entries.values()], [30])

    def test_budget_evicts_the_least_recently_used(self):
        eng = Eng()
        one = self.cache()
        one.maybe_save(eng, np.arange(40, dtype=np.int32), eng.fill(1, 40))
        size = next(iter(one.entries.values()))["bytes"]
        c = self.cache(gb=size * 1.5 / 1e9)           # room for one entry and a half
        a = np.arange(40, dtype=np.int32)
        b = np.arange(1000, 1040, dtype=np.int32)     # not a prefix relation: eviction has to come from the budget
        c.maybe_save(eng, b, eng.fill(2, 40))
        self.assertEqual(len(c.entries), 1)
        self.assertEqual(c.entries[next(iter(c.entries))]["ids"].tolist(), b.tolist())
        self.assertNotIn(a.tolist(), [e["ids"].tolist() for e in c.entries.values()])

    def test_unfinished_saves_are_removed_at_startup(self):
        eng = Eng()
        self.cache().maybe_save(eng, np.arange(40, dtype=np.int32), eng.fill(1, 40))
        stale = self.root / "123-40-deadbeef.tmp"
        stale.mkdir()
        (stale / "kv.bin").write_bytes(b"x" * 100)
        c = self.cache()
        self.assertFalse(stale.exists())
        self.assertEqual(len(c.entries), 1)

    def test_a_failed_save_leaves_no_partial_folder(self):
        eng, c = Eng(), self.cache()
        snap = eng.fill(1, 40)
        snap["draft"] = None                          # the save raises half way through
        c.maybe_save(eng, np.arange(40, dtype=np.int32), snap)
        self.assertEqual(len(c.entries), 0)
        self.assertEqual(list(self.root.glob("*.tmp")), [])

    def test_an_existing_folder_keeps_its_mode(self):
        self.root.mkdir(mode=0o755)
        self.cache()
        self.assertEqual(self.root.stat().st_mode & 0o777, 0o755)


if __name__ == "__main__":
    unittest.main()
