# Qwen3.8-27B on the ANE: architecture cheat sheet

A working reference for reading and extending this model in Forge. Every formula is taken from the reference
implementation (`transformers/models/qwen3_5/modeling_qwen3_5.py`) or from Forge's graph builder
(`coreai/qwen38_coreai_build.py`). Line numbers are for the checkout this was written against; re-check them after
pulling. Things marked **(unverified)** are my reading, not something I ran.

## 0. Map of the code

| What | Where |
| --- | --- |
| Reference math (PyTorch, BF16, easy to read) | `.venv/lib/python3.11/site-packages/transformers/models/qwen3_5/modeling_qwen3_5.py` |
| Reference DeltaNet recurrence | `torch_recurrent_gated_delta_rule` (line 438), layer class `Qwen3_5GatedDeltaNet` (504) |
| Reference attention | `Qwen3_5Attention` (749), RoPE (666), MLP (824), RMSNorm (841), decoder layer (861) |
| Forge ANE graph: DeltaNet | `coreai/qwen38_coreai_build.py` `GDNW` (406), `verify` (464), `prefill` (484) |
| Forge ANE graph: attention | `AttnW` (527) |
| Forge ANE graph: one layer, one entry point | `LayerW` (781), `Entry` (819), `Head` (902) |
| Quantized linear layer | `QConv` (218), rotations `Hadamard` (264) |
| Runtime that drives the chunks | `scripts/qwen38_coreai_model.py`, KV helpers `scripts/qwen38_kv_cache.py` |
| Server, drafting and acceptance | `scripts/qwen38_server.py` (`draft_cycle`), `docs/SPECULATIVE_DECODING.md` |
| Quantization (GPTQ, QAT, lookup tables) | `docs/QUANTIZATION.md`, `scripts/qwen38_gptq_27b.py` |

## 1. The model in numbers

From `config.json` (text part):

| Quantity | Value | Symbol used below |
| --- | --- | --- |
| Layers | 64 (pattern: 3 DeltaNet, 1 attention, repeated 16 times) | |
| Hidden size | 5,120 | `H` |
| MLP inner size | 17,408 | `F` |
| Vocabulary | 248,320 | `V` |
| Attention: query heads / KV heads / head size | 24 / 4 / 256 | `nh`, `nkv`, `hd` |
| Attention: query-head groups per KV head | 24 / 4 = 6 | `grp` |
| Attention: rotary dims | `0.25 * 256 = 64` | `rot` |
| DeltaNet: key heads / value heads | 16 / 48 | `nk`, `nv` |
| DeltaNet: key head size / value head size | 128 / 128 | `dk`, `dv` |
| DeltaNet: total key width / value width | 2,048 / 6,144 | `kd`, `vd` |
| DeltaNet: conv width over q,k,v channels | `2*2048 + 6144 = 10,240`, kernel 4 | `cdim` |
| RMSNorm epsilon | 1e-6 | `eps` |
| RoPE | theta 10,000,000, interleaved mRoPE sections [11, 11, 10] | |
| Native max positions | 262,144 | |
| Embeddings tied to the output head? | no (`tie_word_embeddings: false`) | |
| Activation | SiLU | |

The config also lists `mtp_num_hidden_layers: 1`. Forge speeds up decoding with its own DFlash2 drafter instead;
I did not see Forge use the MTP weights **(unverified)**.

## 2. Notation

- `x` is the residual stream: one vector of `H = 5120` numbers per token.
- `T` is the number of tokens processed in one call (8 for verification, 64 or 32 for prefill).
- `S` is a DeltaNet state matrix. `K, V` are attention caches.
- `*` is matrix multiply, `⊙` is element-wise multiply, `σ` is sigmoid, `silu(u) = u * σ(u)`.

## 3. The whole forward pass

```
h = Embed[token_ids]                                   # (T, 5120), table is 248,320 x 5,120
for layer i in 0..63:
    r = h
    h = RMSNorm_1(h)                                   # input_layernorm
    h = DeltaNet(h)  if layer_types[i] == linear_attention  (3 of every 4)
        Attention(h) if layer_types[i] == full_attention    (every 4th: layers 3, 7, 11, ...)
    h = r + h                                          # residual 1
    r = h
    h = RMSNorm_2(h)                                   # post_attention_layernorm
    h = MLP(h)
    h = r + h                                          # residual 2
h = RMSNorm_final(h)
logits = h[last row] * W_head                          # (248,320,)  one score per word
```

Source: `Qwen3_5DecoderLayer.forward` (reference line 874), `LayerW` and `Entry._forward` in Forge (819 onward).
The token mixer (DeltaNet or attention) is the only place where tokens affect each other.

## 4. Building blocks

### 4.1 RMSNorm (zero-centered)

```
rms_norm(x) = x * rsqrt(mean(x^2) + eps) * (1 + w)         # note the (1 + w), w starts at 0
```

Computed in float32 in the reference. Forge builds `1 + w` into the stored weight (`ln1`, `ln2`, `qn`, `kn`).
Forge's `rms_hidden` (line 315) is a scale-free variant for FP16: it divides by `max|x|` first so squares stay in
[0, 1], because massive activations overflow and tiny layer-0 embeddings underflow.

The DeltaNet output uses a different, **gated** RMSNorm over each head's `dv = 128` values:

```
gated_norm(o, z) = ( o * rsqrt(mean(o^2) + eps) * w ) * silu(z)       # plain w here, not 1 + w
```

### 4.2 Full attention (16 of 64 layers)

Learned matrices per layer: `q_proj: H -> nh*hd*2` (query and an output gate), `k_proj: H -> nkv*hd`,
`v_proj: H -> nkv*hd`, `o_proj: nh*hd -> H`. No biases.

```
[q, gate] = split(x_n * W_q)              # each (T, 24, 256); gate has the same shape as q
q = rms_norm_head(q)                      # q_norm over the 256-dim head, with its own (1 + w)
k = rms_norm_head(x_n * W_k)              # (T, 4, 256), k_norm likewise
v = x_n * W_v                             # (T, 4, 256)
q, k = rope(q, k)                         # only the first 64 of 256 dims are rotated (partial RoPE)
K_cache, V_cache <- append(k, v)          # (4, positions, 256) each
scores = (q * K_cache^T) / sqrt(256)      # divide by 16; each KV head serves 6 query heads
scores = scores + causal_mask
p = softmax(scores)                       # float32 in the reference
o = p * V_cache                           # (T, 24, 256)
o = o ⊙ σ(gate)                           # output gate, applied before the output projection
out = o_flat * W_o                        # (T, 6144) -> (T, 5120)
```

Rotary embedding on the first `rot = 64` dims (reference `apply_rotary_pos_emb`, line 674):

```
rotate_half(a) = concat(-a[32:64], a[0:32])
q_rot = q[:64] ⊙ cos + rotate_half(q[:64]) ⊙ sin ; the other 192 dims pass through unchanged
```

`cos` and `sin` are computed on the host from the position (theta = 10^7, interleaved mRoPE) and passed in as
inputs `cos`, `sin` of shape `(T, 64)`.

Cache cost per token, all 16 attention layers: `16 * 4 * 256 * 2 = 32,768` numbers. At 8 bits plus a 16-bit scale per
token and head that is 32.25 KiB (`docs/RELEASE_NOTES_v0.2.1.md`; `docs/KV_CACHE_V8.md` describes the older
FP16-key format).

### 4.3 Gated DeltaNet (48 of 64 layers)

Learned per layer: `in_proj_qkv: H -> 10,240`, `in_proj_z: H -> 6,144`, `in_proj_b: H -> 48`, `in_proj_a: H -> 48`,
a depthwise causal `conv1d` (kernel 4, over the 10,240 channels), `A_log (48)`, `dt_bias (48)`,
`norm.weight (128)`, `out_proj: 6,144 -> H`.

**Per token, step by step** (reference `Qwen3_5GatedDeltaNet.forward`, line 550):

```
mixed = silu( causal_depthwise_conv4( x_n * W_qkv ) )     # (T, 10240); needs the previous 3 rows as state
q, k, v = split(mixed, [2048, 2048, 6144])                # q, k: 16 heads of 128; v: 48 heads of 128
q, k   = repeat each key head 3 times  ->  48 heads       # 16 key heads feed 48 value heads
z      = x_n * W_z                                        # (T, 48, 128), only used for the output gate
beta   = σ( x_n * W_b )                                   # (T, 48)   write strength, in (0, 1)
g      = -exp(A_log) * softplus( x_n * W_a + dt_bias )    # (T, 48)   log of the decay, always <= 0
q = l2norm(q) ;  k = l2norm(k)                            # unit length per head
q = q / sqrt(128)
```

**The recurrence** (reference `torch_recurrent_gated_delta_rule`, line 438). One state matrix `S` per value head,
shape `(dk, dv) = (128, 128)`:

```
S  = exp(g_t) * S                          # decay: forget a little (g <= 0, so the factor is in (0, 1])
m  = S^T * k_t                             # what memory currently returns for key k_t   (dv numbers)
d  = beta_t * (v_t - m)                    # the correction: the difference between wanted and stored
S  = S + k_t ⊗ d                           # rank-1 write: outer product (dk x dv)
o_t = S^T * q_t                            # read with the query
```

This is the "delta rule": a write replaces what the same key returned before, instead of piling on.
`exp(g)` is a per-head forgetting factor, which is the "gated" part. `S` never grows with sequence length.

**Output:**

```
o = gated_norm(o, z)                       # per head, 128 values; see 4.1
out = o_flat * W_out                       # (T, 6144) -> (T, 5120)
```

State per layer: `S` of `48 * 128 * 128 = 786,432` numbers plus the last 3 conv rows (`3 * 10,240`).
Across 48 layers about 38 million numbers, the same at any context length.

#### 4.3.1 Chunked form (what Forge actually computes)

The loop above is serial in time. The ANE graph processes blocks of 8 rows at once. Let `C = 8` rows in a block,
`g` and `beta` for the block, and `S` the state before it. All shapes are per value head.

```
cum    = cumsum(g)                                         # (C,)  running decay inside the block
pair_ij = exp(cum_i - cum_j)   for j <= i, else 0          # decay between row j and row i
kb = k ⊙ beta ;  vb = v ⊙ beta
N  = (kb * k^T) ⊙ pair ⊙ strict_lower_mask                 # (C, C) interaction between rows of the block
solve (I + N) X = [ vb , kb ⊙ exp(cum) ]                   # (C, dv + dk)
u  = X[:, :dv] ;  wk = X[:, dv:]
v_new = u - wk * S                                         # the part of v the old state does not explain
o   = (q ⊙ exp(cum)) * S  +  ((q * k^T) ⊙ pair) * v_new    # old memory + the block's own writes
S'  = S * exp(cum_last) + (k ⊙ exp(cum_last - cum))^T * v_new
```

This is the WY / UT form of the delta rule (Songlin Yang, "DeltaNet Explained, Part II", linked in the builder).
Forge code: `GDNW.verify` lines 464-482, `GDNW.prefill` lines 484-518.

Checked: `python docs/ARCHITECTURE_CHEATSHEET_check.py` runs both forms on random data. Outputs and final state
agree with the token-by-token recurrence to about 1e-16, and the Neumann-product inverse equals `inv(I + N)`.
This verifies the equations as written here; it does not run the ANE graph.

`(I + N)` is unit lower-triangular, so `(I + N)^-1` is a finite sum. Forge computes it with the Neumann product
`(I - N)(I + N^2)(I + N^4)...` (`inv_unit_lower`, line 380) because the ANE compiler rejects row-by-row substitution
and matmul chains between computed tensors. With `GDN_FAST=0` it falls back to `fwd_sub` (line 369).

### 4.4 MLP (all 64 layers, about 17 billion weights)

```
a   = silu( x_n * W_gate ) ⊙ ( x_n * W_up )     # (T, 5120) -> (T, 17408), two paths
out = a * W_down                                 # (T, 17408) -> (T, 5120)
```

Three matrices of `5120 x 17408 = 89,128,960` weights each, so 267,386,880 per layer. Source: reference line 835;
Forge `LayerW.mlp` (line 794). In Forge the MLP input and the middle activation are optionally rotated by an online
block-Hadamard matrix (`Hadamard`, block size 1024) before the quantized matmuls; seeds come from the export metadata.

### 4.5 Output head

`logits = RMSNorm_final(x_last) * W_head`, `W_head` is `5120 x 248,320`. Forge compiles it as `head_T8.aimodel` in
8 row parts with a 4-bit lookup table (`Head`, line 902). Sampling happens on the CPU in `qwen38_server.py`.

## 5. What number type each piece uses in Forge

| Piece | Stored as | Reconstructed as |
| --- | --- | --- |
| All MLP matrices (gate, up, down), release 0.2 | 6-bit index per pair of weights (64 entries of 2) = 3 bits per weight | `W[2g+t, j] = s[2g+t] * L[idx[g, j], t]`, FP16 codebook `L (64, 2)`, FP16 scale `s` per output row (`docs/QUANTIZATION.md`, `vector 2x64`) |
| Token-mixer projections (DeltaNet `in_proj_qkv`, `in_proj_z`, `out_proj`; attention `q_proj`, `o_proj`) | Lookup tables of 2 to 4 bits. In the first release's plan: vector 2x16 for layers 0-23, scalar 16-entry (LUT4) for layers 24-63. Release 0.2's exact mixer plan: **(unverified)** | same `QConv` LUT path, with per-row scale |
| Attention `k_proj`, `v_proj` | INT8 with per-channel scales (compile-time constant) | `QConv` path with `w8` (line 232) |
| DeltaNet `in_proj_a`, `in_proj_b` | dense FP16 | as is (line 411-412) |
| Mixer error correction | FP16 rank-64 pair `a @ (b @ x)` on selected mixer matrices | added to the conv output (`lr_a`, `lr_b`) |
| Output head | 4-bit lookup table (LUT4) in 8 row parts | `Head` (line 902) |
| Norm weights, conv taps, `A_log`, `dt_bias` | FP16 | as is |
| Activations between layers | FP16 | |
| Attention K and V cache | INT8 + FP16 scale per token and head (`kv8`) | `code * scale` |
| Attention scores / probabilities | INT8 scores (step 1/4), FP8 softmax values (scale 1/64) on M6 | form list `s8,s8b,sm8,pvf8` |
| DeltaNet state `S` | fixed-size tensor `(nv, dk, dv)` | reference uses float32; the ANE build's exact precision is **(unverified)** |

M5 chips reject the FP8 forms, which is why release 0.2.1 is M6 only.

## 6. Why Forge's graph looks different from the reference

| Difference | Reason | Where |
| --- | --- | --- |
| Tensors are channel-major `(1, C, 1, T)` and linears are 1x1 convs | The ANE prefers this layout | `QConv`, `Entry` |
| Fixed-shape entry points per context size (`v8_8k`, `p64_16k`, ...) | The ANE compiles fixed shapes; the runtime picks the smallest entry that fits and copies the cache | `Entry`, `docs/SPECULATIVE_DECODING.md` section "How ANE context expansion is implemented" |
| Verify entries use `T = 8` rows, prefill 64 rows (32 above 96K) | One verify call checks the anchor plus 7 draft tokens | `P, C_SUB = 8, 8` (line 44), `TPS`, `TPS_LONG` |
| DeltaNet "lazy commit": `pend`, `commit`, `commit_last` | DeltaNet state cannot be rewound, so a verify call stores its candidate rows as pending and the next call commits only the accepted prefix | `GDNW.commit_pending` (448) |
| History attention is tiled (`ATT_BLOCK` 2048 / 4096) with one global max | The ANE compiler fails on a single softmax wider than 65,536 | `AttnW.forward`, `kv_len` (807) |
| `GDN_SQ = 16`, `GDN_SV = 64` scaling around the DeltaNet core | Keeps `q * S` out of FP16 subnormals; undone in `finish` | lines 342, 445, 457 |
| Custom `softplus`, `silu` via tanh | The ANE's FP16 softplus returns 0 above about 11; native silu has about 1e-3 absolute error near 0 | lines 327, 357 |
| Online Hadamard rotations on quantized matmul inputs | Spreads outliers so 3-bit weights lose less accuracy | `Hadamard` (264), `mixer_rotation` (297) |
| Drafter reuses the target's output head | Saves 0.86 GB of Neural Engine memory | release notes 0.2.1 |

## 7. Interface of one compiled entry (one chunk of 4 layers)

From `Entry.input_names` / `output_names` (around line 838):

Inputs (in order):

```
x            (1, 5120, 1, T)        hidden states entering the chunk, FP16
cos, sin     (T, 64)                RoPE tables for these positions
mask         (1, ctx)               0 for visible history rows, large negative otherwise
conv_sel     (3, P+3)               picks the 3 conv rows to carry in
commit       (1, P, 1)              which pending rows to commit this call
commit_last  (1, P, 1)              the last committed row's selector
conv_sel_out, valid   prefill only  row selector and 0/1 validity of padded rows
per DeltaNet layer j:  conv_j (P+3, 10240), rec_j (48, 128, 128), pend_j (48, 3P+1, 128)
per attention layer j: k_j, v_j (+ ks_j, vs_j scales for INT8)   (4, ctx, 256)
```

Outputs:

```
y                       hidden states leaving the chunk
tap<l>                  features for the drafter, layers in TAPS only
per DeltaNet layer:     conv_j_out, rec_j_out, pend_j_out
per attention layer:    k_j_new, v_j_new (the runtime writes accepted rows into the cache)
```

A forward pass is 16 such chunk calls in order, then the head.

## 8. Speculative decoding in one paragraph

After prefill the target picks an anchor token. DFlash2 proposes 7 following tokens. One target call evaluates
`[anchor, d1 .. d7]` as `T = 8` rows. The server accepts drafts in order while each matches what the target would
pick (under sampling: accepted with the target's probability of that token) and takes the target's own token at the
first miss. If all 7 pass, the 8th row yields a bonus token. Only the accepted prefix is committed (`accept(k)`),
which is why DeltaNet needs the lazy-commit machinery. The drafter has its own 2,048-row ring and is unaffected by
the target's context size. Source: `docs/SPECULATIVE_DECODING.md`, `scripts/qwen38_server.py`.

## 9. Extension recipes

These are my reading of the code, not tested changes. For each, the validation you need is in section 10.

### 9.1 Longer context (beyond the 192K ladder)

- The model's own limit is 262,144 positions, so 256K is the real ceiling without extending RoPE.
- Context entries come from the `--ctx` list of the builder (`main`, `--ctx 8192,...`) and `forge.py compile --ctx`.
  Add the new size to the ladder and to `TPS_LONG`, `PREFILL_LONG_FROM`, `ATT_BLOCK_LONG`, `ATT_BLOCK_PREFILL_LONG`
  (lines 45-63). `kv_len` (807) already lets entries above 65,536 use their full length through the tiled path.
- Cost: INT8 cache is 32.25 KiB per position, so 256K is about 8.1 GiB. With 16.5 GB of weights on the Neural Engine
  and about 2 GB of process memory, that is about 26.6 GB on a 32 GB Mac. Tight; swap would likely start **(unverified)**.
- The runtime also needs the entry in its manifest and in the cache-resize logic (`qwen38_coreai_model.py`).
- Prefill time rises quickly: the notes list 62 tok/s at 160K and 192K.

### 9.2 Smaller KV cache (for more context in the same memory)

- Today K and V are INT8 with one FP16 scale per token and KV head. 4-bit keys or values would roughly halve the
  per-token cost.
- Touch points: `KV_INPUTS` and `INT8_LAYOUTS` (`scripts/qwen38_kv_cache.py`), the `quant8` / `dequant8` pairs and
  the scale handling in `AttnW.forward`, and the runtime code that writes accepted rows.
- Risk: attention scores depend on keys more than on values; measure each separately.

### 9.3 Change what a layer does

- `CFG["layer_types"]` decides which layers are DeltaNet or attention. `Entry` derives `gdn_j` and `att_j` from it,
  so the interface in section 7 follows automatically.
- Chunk boundaries come from `CHUNK` and `plan_chunks` (`scripts/qwen38_ane_model.py`) and `--plan` in the builder.
  Four layers per chunk gives 3 DeltaNet + 1 attention in every chunk.

### 9.4 A different quantization

- The weight path is `QConv`. A new format needs an export in `W[key/...]` (idx, lut, scale, or int8) and a branch in
  `QConv.__init__`. The ANE only accepts vector lookup tables up to 256 values (`docs/QUANTIZATION.md`).
- Rotations must match what the weights were fitted for (`mixer_rotation`, `mlp_rotation`); mismatches mark a build
  `TIMING_ONLY` and the runtime refuses to serve it.

### 9.5 New drafter or sampling rule

- Drafter code: `scripts/dflash2_coreai_drafter.py`, `coreai/dflash2_coreai_build.py`. Taps from the target are chosen
  by `TAPS` and exported by `Entry`.
- Acceptance and sampling: `draft_cycle` in `scripts/qwen38_server.py`; see `docs/research/DFLASH2_SAMPLING_PLAN.md`.

## 10. How to check a change

- Compare against the BF16 reference with the project's KL and top-1 scripts: `scripts/m6_kl512_eval.py`
  (release 0.2 numbers: KL-512 0.052, top-1 agreement 91.9%).
- Long context: `scripts/m6_long_ctx_eval.py` (recall of planted codes; the 0.2.1 notes report all recalled to 195K).
- Per-layer numerical checks: `scripts/m6_attn_core_check.py`, `scripts/m6_gdn_bench.py`, `DBG_O=1` debug outputs.
- Compare the ANE build against a plain-PyTorch version of the same layer with the equations in sections 4.2 and 4.3.
- Watch memory: your dashboard's "Where the 32 GB of memory is" card, and the server log's `[ctx]` lines.

## 11. Gotchas worth knowing

- The ANE compiler fails on tensors wider than 65,536 along one axis (why history is tiled).
- FP16 subnormals and overflow are the usual numeric failures: massive activations in the residual stream,
  tiny layer-0 embeddings, `q * S` products, `softplus`. Check the helpers before adding ops.
- Every macOS update, including minor ones, recompiles all packages (about 55 minutes for a 128K build).
- The first start after a cache purge can take over 30 minutes; the server resumes partial compiles.
- `GDN_FAST=0 ATT_BLOCK=16384` rebuilds the older release graph, useful as a control when something regresses.

## 12. Glossary

- **Token mixer**: the layer part where tokens exchange information (attention or DeltaNet).
- **KV cache**: stored keys and values of earlier tokens, read by every attention layer.
- **Delta rule**: write `beta * (v - what_memory_returns_for_k)` along `k`, so a repeated key replaces its old value.
- **Gated**: the decay `exp(g)` can shrink the whole memory each step; `g` is predicted from the input.
- **GQA**: grouped-query attention, 24 query heads sharing 4 key/value heads.
- **LUT / vector lookup table**: weights stored as small indices into a shared codebook.
- **Prefill / verify**: reading a prompt in large blocks versus checking 8 candidate tokens at a time.
- **Lazy commit**: keeping candidate state as pending until the accepted tokens are known.
