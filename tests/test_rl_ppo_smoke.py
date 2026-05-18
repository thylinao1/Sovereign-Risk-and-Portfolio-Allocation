"""Smoke tests for PPOAgent. Requires TensorFlow, so they are skipped in CI
unless TF is on the runner. End-to-end notebook execution is the
load-bearing verification for the PPO code; this file exists so the coverage
gap on src/rl/ppo.py is explicit rather than silent.
"""
import pytest

tf = pytest.importorskip("tensorflow")

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.rl.ppo import PPOAgent, PPOAgentNorm


def test_ppo_agent_actor_critic_shapes():
    agent = PPOAgent(state_dim=64, action_dim=10)
    assert agent.actor.input_shape == (None, 64)
    assert agent.critic.input_shape == (None, 64)
    assert agent.actor.output_shape == [(None, 10), (None, 10)]
    assert agent.critic.output_shape == (None, 1)


def test_ppo_agent_norm_uses_smaller_actor():
    agent_big = PPOAgent(state_dim=64, action_dim=10)
    agent_norm = PPOAgentNorm(state_dim=64, action_dim=10)
    assert agent_norm.actor.count_params() < agent_big.actor.count_params(), (
        "PPOAgentNorm should use the smaller 128-64 actor"
    )


def test_log_prob_uses_clipped_log_std_consistently():
    """Regression guard for the log_std-clipping consistency: clip once and
    reuse for both the variance term and the normalising constant so the
    log-probability is internally consistent."""
    agent = PPOAgent(state_dim=4, action_dim=2)
    # extreme log_std way outside [-5, 2]
    actions = tf.constant([[0.5, -0.5]])
    mean = tf.constant([[0.0, 0.0]])
    log_std_big = tf.constant([[10.0, 10.0]])     # would normally clip to 2
    log_std_at_cap = tf.constant([[2.0, 2.0]])   # already at the cap

    lp_big = agent._log_prob_gaussian(actions, mean, log_std_big)
    lp_at_cap = agent._log_prob_gaussian(actions, mean, log_std_at_cap)

    # After the fix, these must be equal because both clip to log_std=2.0.
    # Before the fix they differed because the normalising constant used the
    # raw log_std.
    assert tf.reduce_max(tf.abs(lp_big - lp_at_cap)) < 1e-5
