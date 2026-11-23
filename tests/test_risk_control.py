import numpy as np
import pytest

from analyze import Stats, apply, crc_k, evaluate, fit, per_object_k


def test_stats_fnr_is_monotone_curve(stats_file):
    S = Stats(stats_file)
    assert S.fnr.shape == (len(S.n), S.K)
    assert np.all(S.fnr[:, 0] == 0) and np.allclose(S.fnr[:, -1], 1)
    assert np.all(np.diff(S.fnr, axis=1) >= 0)
    assert S.miou() == pytest.approx(1.0)


def test_crc_k_finite_sample_rule():
    assert crc_k(np.zeros((0, 5)), 0.1) == 0
    # 9 objects with identical FNR curves: risk(k) = 0.9 * fnr(k) + 0.1 must stay <= alpha
    fnr = np.tile(np.array([0.0, 0.05, 0.1, 0.2, 0.5, 1.0]), (9, 1))
    assert crc_k(fnr, 0.2) == 2           # 0.9*0.1+0.1 = 0.19 <= 0.2 < 0.9*0.2+0.1 = 0.28
    assert crc_k(fnr, 0.1) == 0           # the +1/(n+1) term alone already uses up alpha = 0.1


def test_per_object_k_rank():
    assert per_object_k(np.zeros((5, 4)), 0.1, 0.1) == 0       # floor(6 * 0.1) = 0 < 1 -> most conservative
    t = np.arange(19)
    fnr = (np.arange(30)[None, :] > t[:, None]).astype(float)  # object i has FNR 0 up to k = t_i, then 1
    assert per_object_k(fnr, 0.1, 0.1) == 1                    # floor(20 * 0.1) = 2nd smallest t_i


@pytest.mark.parametrize("method", ["crc", "crc-class"])
def test_crc_controls_mean_fnr(stats_file, method):
    S = Stats(stats_file)
    res = evaluate(S, S, method, alpha=0.1, repeats=20)
    assert res["mean_fnr"]["mean"] <= 0.1 + 0.01
    assert res["mean_fnr"]["mean"] >= 0.1 - 0.03            # not trivially conservative


def test_per_object_guarantee(stats_file):
    S = Stats(stats_file)
    covs = []
    for seed in range(20):
        rng = np.random.default_rng(seed)
        idx = rng.permutation(len(S.n))
        ci, ti = idx[: len(idx) // 2], idx[len(idx) // 2:]
        k = fit(S, ci, "per-object", alpha=0.1, beta=0.1)
        fnr, area, _ = apply(S, ti, "per-object", k)
        covs.append((fnr <= 0.1).mean())
        assert np.all(area >= 1 - fnr - 1e-9)                 # mask area >= object pixels it covers
    assert np.mean(covs) >= 0.9 - 0.01
