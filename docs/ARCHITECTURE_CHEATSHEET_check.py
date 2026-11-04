"""Checks section 4.3.1 of ARCHITECTURE_CHEATSHEET.md: the chunked gated delta rule equals the token-by-token recurrence.

Run:  python docs/ARCHITECTURE_CHEATSHEET_check.py
Needs only numpy. Random data; all three checks should print True (errors near 1e-16).
"""
import numpy as np

rng = np.random.default_rng(0)
C, dk, dv = 8, 5, 6
q = rng.normal(size=(C, dk)); k = rng.normal(size=(C, dk))
q /= np.linalg.norm(q, axis=1, keepdims=True); k /= np.linalg.norm(k, axis=1, keepdims=True)
v = rng.normal(size=(C, dv))
beta = 1 / (1 + np.exp(-rng.normal(size=C)))      # sigmoid, in (0, 1)
g = -np.abs(rng.normal(size=C)) * 0.5             # log decay, <= 0
S0 = rng.normal(size=(dk, dv)) * 0.3

# token by token (transformers modeling_qwen3_5.py, torch_recurrent_gated_delta_rule)
S, o_ref = S0.copy(), []
for t in range(C):
    S = S * np.exp(g[t])
    d = beta[t] * (v[t] - S.T @ k[t])
    S = S + np.outer(k[t], d)
    o_ref.append(S.T @ q[t])
o_ref = np.array(o_ref)

# chunked (coreai/qwen38_coreai_build.py, GDNW.verify)
lower, strict = np.tril(np.ones((C, C))), np.tril(np.ones((C, C)), -1)
cum = np.cumsum(g)
pair = np.exp(cum[:, None] - cum[None, :]) * lower
kb, vb = k * beta[:, None], v * beta[:, None]
N = (kb @ k.T) * pair * strict
X = np.linalg.solve(np.eye(C) + N, np.concatenate([vb, kb * np.exp(cum)[:, None]], 1))
u, wk = X[:, :dv], X[:, dv:]
vn = u - wk @ S0
o = (q * np.exp(cum)[:, None]) @ S0 + ((q @ k.T) * pair) @ vn
S_new = S0 * np.exp(cum[-1]) + (k * np.exp(cum[-1] - cum)[:, None]).T @ vn


def inv_unit_lower(n, rows):  # (I - N)(I + N^2)(I + N^4)...  (Forge: inv_unit_lower)
    eye, t, p, step = np.eye(rows), np.eye(rows) - n, n, 2
    while step < rows:
        p = p @ p
        t = t @ (eye + p)
        step *= 2
    return t


print("outputs match:        ", np.allclose(o, o_ref), np.abs(o - o_ref).max())
print("final state matches:  ", np.allclose(S_new, S), np.abs(S_new - S).max())
print("Neumann inverse works:", np.allclose(inv_unit_lower(N, C), np.linalg.inv(np.eye(C) + N)))
