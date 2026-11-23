import numpy as np
import pytest


@pytest.fixture(scope="session")
def stats_file(tmp_path_factory):
    """A synthetic cache/stats_*.npz in the format written by src/infer.py (no model, no dataset)."""
    rng = np.random.default_rng(0)
    edges = np.linspace(0.0, 1.0 + 1e-9, 101)
    n_obj = 3000
    out = {k: [] for k in ["img", "cls", "n_gt", "n_boundary", "h_int", "h_bnd", "h_neg", "argmax_hit"]}
    for i in range(n_obj):
        a = rng.uniform(0.5, 6.0)                       # object difficulty: small a -> low class probabilities
        p = rng.beta(a, 1.0, rng.integers(50, 400))
        bnd = np.zeros(len(p), bool)
        bnd[: len(p) // 4] = True
        out["img"].append(i // 2); out["cls"].append(int(rng.integers(1, 21)))
        out["n_gt"].append(len(p)); out["n_boundary"].append(int(bnd.sum()))
        out["h_int"].append(np.histogram(p[~bnd], edges)[0]); out["h_bnd"].append(np.histogram(p[bnd], edges)[0])
        out["h_neg"].append(np.histogram(rng.beta(1.0, 8.0, 500), edges)[0])
        out["argmax_hit"].append(int((p > 0.5).sum()))
    res = {k: np.array(v) for k, v in out.items()}
    res["confusion"] = np.diag(np.full(21, 10)).astype(np.int64)
    res["edges"] = edges
    f = tmp_path_factory.mktemp("cache") / "stats_synthetic.npz"
    np.savez(f, **res)
    return str(f)
