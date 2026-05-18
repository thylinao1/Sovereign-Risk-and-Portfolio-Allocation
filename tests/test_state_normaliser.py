"""Tests for StateNormaliser (the Welford-running z-scorer used by PPOAgentNorm)."""
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.rl.normaliser import StateNormaliser


def test_mean_var_converge_to_population_values():
    rng = np.random.default_rng(0)
    true_mean = np.array([5.0, -3.0, 100.0])
    true_std = np.array([1.0, 2.0, 50.0])
    sn = StateNormaliser(3)
    for _ in range(5000):
        x = rng.normal(true_mean, true_std)
        sn.update(x)
    assert np.allclose(sn.mean, true_mean, atol=1.0), f"learned mean {sn.mean} vs expected {true_mean}"
    np.testing.assert_allclose(np.sqrt(sn.var), true_std, atol=2.0)


def test_normalised_output_is_zero_mean_unit_variance():
    rng = np.random.default_rng(1)
    sn = StateNormaliser(2)
    samples = []
    for _ in range(2000):
        x = rng.normal(loc=[10, -5], scale=[3, 7])
        samples.append(sn.normalise(x, update=True))
    samples = np.array(samples)
    # After 2000 updates, the streamed normalised samples should be roughly
    # zero-mean unit-variance.
    assert np.allclose(samples.mean(axis=0), 0.0, atol=0.2)
    assert np.allclose(samples.std(axis=0), 1.0, atol=0.2)


def test_freeze_stops_updates():
    sn = StateNormaliser(1)
    for _ in range(100):
        sn.update(np.array([0.0]))
    before = (sn.mean.copy(), sn.M2.copy(), sn.count)
    sn.frozen = True
    for _ in range(100):
        sn.update(np.array([100.0]))
    after = (sn.mean.copy(), sn.M2.copy(), sn.count)
    assert before[2] == after[2] == 100
    assert np.allclose(before[0], after[0])
    assert np.allclose(before[1], after[1])


def test_eps_prevents_div_by_zero_on_constant_input():
    sn = StateNormaliser(1, eps=1e-3)
    # Feed identical inputs; variance stays 0
    for _ in range(50):
        out = sn.normalise(np.array([7.0]), update=True)
    # Output should be finite, not nan / inf
    assert np.isfinite(out).all()
