"""Reference harness for the Inkling 2 Mini architecture (a looped MoE), for planning an ANE port.

Subcommands (all run on CPU in fp32 with a *tiny* randomly initialised model that has the real structure:
24-layer pattern scaled to 18 layers, 6-layer attention-residual blocks, layers 6-11 looped 3x, sliding + full
attention, short convolutions, sigmoid-routed MoE, per-pass message gates):

  trace         per-layer-visit hidden states, router top-k per visit, and the visit schedule -> npz
  consistency   full forward vs prefill+decode vs prefill+verify chunk (what a chunked ANE decode must reproduce)
  fp16          emulate fp16 storage of groups of modules the model code keeps in fp32, report the damage

The tiny model has random weights: results validate the harness and expose structural problems (cache handling,
overflow). They do NOT predict how the real checkpoint behaves in fp16. Run the same commands with --checkpoint on a
machine that can load the real weights (untested path, see load_real below).

    python ref_harness.py trace --out trace.npz
    python ref_harness.py consistency
    python ref_harness.py fp16
"""
import argparse
import copy
import json
import sys

import numpy as np
import torch
import torch.nn.functional as F

from transformers import Inkling2LoopTextConfig
from transformers.cache_utils import DynamicCache
from transformers.models.inkling2_loop import modeling_inkling2_loop as M


def tiny_config(**over):
    kw = dict(hidden_size=128, num_hidden_layers=18, num_attention_heads=4, swa_num_attention_heads=4,
              num_key_value_heads=2, swa_num_key_value_heads=2, head_dim=32, swa_head_dim=32,
              n_routed_experts=16, num_experts_per_tok=4, moe_intermediate_size=64, vocab_size=512,
              unpadded_vocab_size=500, eos_token_id=1, rel_extent=48, sliding_window_size=32,
              log_scaling_n_floor=40, loop_layer_index_start=6, loop_layer_index_end=12, loop_num_passes=3)
    kw.update(over)
    return Inkling2LoopTextConfig(**kw)


def build_model(cfg=None, seed=0):
    cfg = cfg or tiny_config()
    torch.manual_seed(seed)
    model = M.Inkling2LoopForCausalLM(cfg).float().eval()
    g = torch.Generator().manual_seed(seed + 1)
    with torch.no_grad():
        for name, p in model.named_parameters():
            if name.endswith(("query", "queries")):
                p.copy_(torch.randn(p.shape, generator=g))                 # real checkpoints have trained queries
            elif p.dim() == 1 and "norm" in name and "weight" in name:
                p.copy_(1.0 + 0.1 * torch.randn(p.shape, generator=g))
            elif name.endswith(("scale",)):
                p.copy_(1.0 + 0.1 * torch.randn(p.shape, generator=g))
            elif "e_score_correction_bias" in name:
                p.zero_()
            else:
                p.copy_(0.08 * torch.randn(p.shape, generator=g))
    return model


def load_real(path):  # pragma: no cover - needs the real checkpoint and a machine that can hold it
    """Untested: load the real checkpoint with transformers (NVFP4/MXFP8 needs a loader that supports modelopt)."""
    from transformers import AutoModelForCausalLM
    return AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16).eval()


def logits_of(model, ids, cache=None, position_ids=None):
    out = model(input_ids=ids, cache=cache, use_cache=cache is not None, position_ids=position_ids)
    return out.logits.float()


# ------------------------------------------------------------------------------------------------ trace
class Tracer:
    """Hooks every decoder layer call (in execution order = the visit schedule) and every router call."""

    def __init__(self, model):
        self.visits, self.router, self.handles = [], [], []
        text = model.model if hasattr(model, "model") else model
        for i, layer in enumerate(text.layers):
            self.handles.append(layer.register_forward_hook(self._layer_hook(i)))
            self.handles.append(layer.mlp.gate.register_forward_hook(self._router_hook(i)))

    def _layer_hook(self, i):
        def hook(mod, args, kwargs, out):
            has_msg = (len(args) > 2 and args[2] is not None) or kwargs.get("messages") is not None
            self.visits.append(dict(layer=i, messages=bool(has_msg), out=out.detach().float().cpu().numpy()))
        return hook

    def _router_hook(self, i):
        def hook(mod, args, out):
            self.router.append(dict(layer=i, topk=out[2].detach().cpu().numpy(), weights=out[1].detach().float().cpu().numpy()))
        return hook

    def close(self):
        for h in self.handles:
            h.remove()


# the layer hooks need kwargs; re-register with_kwargs
def _register(layer, fn):
    return layer.register_forward_hook(fn, with_kwargs=True)


def cmd_trace(a):
    model = build_model()
    ids = torch.randint(0, 500, (1, a.tokens), generator=torch.Generator().manual_seed(7))
    t = Tracer.__new__(Tracer)
    t.visits, t.router, t.handles = [], [], []
    text = model.model
    for i, layer in enumerate(text.layers):
        t.handles.append(_register(layer, t._layer_hook(i)))
        t.handles.append(layer.mlp.gate.register_forward_hook(t._router_hook(i)))
    with torch.no_grad():
        logits = logits_of(model, ids)
    t.close()
    cfg = model.config.get_text_config() if hasattr(model.config, "get_text_config") else model.config
    sched = [(v["layer"], v["messages"]) for v in t.visits]
    print(f"visits={len(t.visits)} (expected {cfg.total_num_hidden_layers})")
    print("schedule (layer, has_messages):", sched)
    out = {"logits": logits.numpy(), "ids": ids.numpy()}
    for n, v in enumerate(t.visits):
        out[f"visit{n:02d}_layer{v['layer']:02d}_out"] = v["out"]
    for n, r in enumerate(t.router):
        out[f"visit{n:02d}_layer{r['layer']:02d}_topk"] = r["topk"]
        out[f"visit{n:02d}_layer{r['layer']:02d}_wts"] = r["weights"]
    np.savez(a.out, **out)
    # how much do the router choices of a looped layer change between passes?
    by_layer = {}
    for r in t.router:
        by_layer.setdefault(r["layer"], []).append(r["topk"].reshape(-1, r["topk"].shape[-1]))
    print("looped-layer expert overlap between passes (random weights, indicative of nothing but the plumbing):")
    for layer in range(cfg.loop_layer_index_start, cfg.loop_layer_index_end):
        p = by_layer[layer]
        ov = []
        for x, y in zip(p[:-1], p[1:]):
            ov.append(np.mean([len(set(a_) & set(b_)) / len(a_) for a_, b_ in zip(x, y)]))
        print(f"  layer {layer}: overlap pass1->2 {ov[0]:.2f}, pass2->3 {ov[1]:.2f}")
    print("wrote", a.out)


# ------------------------------------------------------------------------------------------- consistency
def rel(a, b):
    return float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-12))


def cmd_consistency(a):
    model = build_model()
    cfg = model.config
    L, P = a.tokens, a.prefill
    ids = torch.randint(0, 500, (1, L), generator=torch.Generator().manual_seed(11))
    with torch.no_grad():
        ref = logits_of(model, ids)

        cache = DynamicCache(config=cfg)
        parts = [logits_of(model, ids[:, :P], cache)]
        for t in range(P, L):
            parts.append(logits_of(model, ids[:, t:t + 1], cache))
        step = torch.cat(parts, 1)

        cache = DynamicCache(config=cfg)
        parts = [logits_of(model, ids[:, :P], cache)]
        for t in range(P, L, a.verify):
            parts.append(logits_of(model, ids[:, t:t + a.verify], cache))
        chunk = torch.cat(parts, 1)

    def top1(x):
        return float((x.argmax(-1) == ref.argmax(-1)).float().mean())
    print(f"tokens={L} prefill={P} verify_chunk={a.verify}")
    print(f"prefill+single-token decode vs full forward : rel L2 {rel(step.numpy(), ref.numpy()):.2e}, top-1 agree {top1(step):.3f}")
    print(f"prefill+{a.verify}-token verify chunks vs full : rel L2 {rel(chunk.numpy(), ref.numpy()):.2e}, top-1 agree {top1(chunk):.3f}")
    n_layers = len(cache.layers)
    print(f"cache layers allocated: {n_layers} (24-layer model pattern: {cfg.num_hidden_layers} + "
          f"{(cfg.loop_num_passes - 1) * (cfg.loop_layer_index_end - cfg.loop_layer_index_start)} per-pass copies)")


# ------------------------------------------------------------------------------------------------- fp16
GROUPS = {
    "attn_residual_mix": ("attention_input", "mlp_input", "final_input"),
    "message_path": ("message_input", "q_message_gate", "k_message_gate", "v_message_gate", "mlp_message_gate"),
    "short_convs": ("post_attention_convolution", "post_mlp_convolution", "sconv"),
    "router": ("mlp.gate",),
    "experts": ("mlp.experts",),
    "attn_extras": ("relative_position_bias", "sink_proj", "q_proj.norm"),
}


def _round16(t):
    return t.to(torch.float16).to(t.dtype)


def apply_fp16_emulation(model, group):
    """Round the module's weights to fp16 and its floating inputs/outputs to fp16 (storage precision only)."""
    keys = GROUPS[group]
    hit = 0
    for name, mod in model.named_modules():
        if any(k in name for k in keys) and not any(k in n for n, _ in mod.named_children() for k in ()):
            if not list(mod.children()) or isinstance(mod, (M.Inkling2LoopMessageGate,)):
                hit += 1
                with torch.no_grad():
                    for p in mod.parameters(recurse=False):
                        p.copy_(_round16(p))

                def pre(m, args, kwargs):
                    f = lambda x: _round16(x) if torch.is_tensor(x) and x.is_floating_point() else x
                    return tuple(f(x) for x in args), {k: f(v) for k, v in kwargs.items()}

                def post(m, args, kwargs, out):
                    if torch.is_tensor(out):
                        return _round16(out)
                    if isinstance(out, tuple):
                        return tuple(_round16(o) if torch.is_tensor(o) and o.is_floating_point() else o for o in out)
                    return out
                mod.register_forward_pre_hook(pre, with_kwargs=True)
                mod.register_forward_hook(post, with_kwargs=True)
    return hit


def kl(p_logits, q_logits):
    lp, lq = F.log_softmax(p_logits, -1), F.log_softmax(q_logits, -1)
    return float((lp.exp() * (lp - lq)).sum(-1).mean())


def cmd_fp16(a):
    base = build_model()
    ids = torch.randint(0, 500, (1, a.tokens), generator=torch.Generator().manual_seed(5))
    with torch.no_grad():
        ref = logits_of(base, ids)
    print(f"{'group':20s} {'modules':>7s} {'KL(ref||fp16)':>14s} {'top1 agree':>11s} {'nonfinite':>10s}")
    for g in GROUPS:
        m = copy.deepcopy(base)
        n = apply_fp16_emulation(m, g)
        with torch.no_grad():
            out = logits_of(m, ids)
        bad = int((~torch.isfinite(out)).sum())
        k = kl(ref, out) if bad == 0 else float("nan")
        print(f"{g:20s} {n:7d} {k:14.3e} {float((out.argmax(-1) == ref.argmax(-1)).float().mean()):11.3f} {bad:10d}")
    print("\nStorage-precision emulation only: fp16 accumulation and internal intermediates are NOT modelled, and the\n"
          "weights are random. Treat as a harness smoke test. The real answer needs the real weights on the target.")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("trace"); p.add_argument("--tokens", type=int, default=24); p.add_argument("--out", default="trace.npz")
    p = sub.add_parser("consistency")
    p.add_argument("--tokens", type=int, default=72); p.add_argument("--prefill", type=int, default=40); p.add_argument("--verify", type=int, default=8)
    p = sub.add_parser("fp16"); p.add_argument("--tokens", type=int, default=48)
    a = ap.parse_args()
    {"trace": cmd_trace, "consistency": cmd_consistency, "fp16": cmd_fp16}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
