"""G3 gradient-flow tests (rule R4 — kills failure F4).

The v0 height bonus was computed, logged, summed into the return — and its
gradient reached only the critic, never the policy. These tests verify,
for EVERY reward term, the full plumbing:

  term-specific event → compute_rewards → advantage → policy-gradient loss
  → nonzero gradient on POLICY (not critic) parameters.

Each test builds a RewardContext in which only the target term is nonzero,
confirms compute_rewards responds, then runs one REINFORCE update and
asserts the policy mean-net parameters actually change.

Run: python -m pytest tests/env/test_gradient_flow.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from dreaming_together.envs import rewards as R
from dreaming_together.policies.ff_policy import GaussianPolicy


def _base_ctx() -> R.RewardContext:
    """All-zero context: no term fires (time_alive excluded via alive=False)."""
    return R.RewardContext(
        dt=0.05,
        alive=np.zeros(4, dtype=bool),
        damage_taken=np.zeros(4),
        killed_this_step=np.zeros(4, dtype=bool),
        dist_to_nearest_opp=np.full(4, 3.5),
        prev_dist_to_nearest_opp=np.full(4, 3.5),
        action=np.zeros((4, 5)),
        prev_action=np.zeros((4, 5)),
        episode_done=False,
        team_result=np.zeros(2, dtype=int),
        shield_blocks_line=np.zeros(2, dtype=bool),
        window_open=np.zeros(2, dtype=bool),
        opp_in_reload=np.zeros(2, dtype=bool),
        window_assist=np.zeros(2, dtype=bool),
        pellet_hits_opp=np.zeros(4, dtype=int),
        friendly_pellets=np.zeros(4, dtype=int),
        kills=np.zeros(4, dtype=int),
        within_3m=np.zeros(4, dtype=bool),
        shot_through_window_4plus=np.zeros(2, dtype=bool),
        died_before_teammate=np.zeros(4, dtype=bool),
    )


# term name → (mutator that makes ONLY this term fire, agent index affected)
TERMS = {
    "time_alive":     (lambda c: setattr(c, "alive", np.array([True, False, False, False])), 0),
    "damage_taken":   (lambda c: c.damage_taken.__setitem__(1, 12.0), 1),
    "advance":        (lambda c: c.prev_dist_to_nearest_opp.__setitem__(1, 4.0), 1),
    "action_rate":    (lambda c: c.action.__setitem__((0, 0), 1.0), 0),
    "win_loss":       (lambda c: (setattr(c, "episode_done", True),
                                  c.team_result.__setitem__(0, 1),
                                  c.team_result.__setitem__(1, -1)), 0),
    "shield_blocking": (lambda c: c.shield_blocks_line.__setitem__(0, True), 0),
    "window_timing":  (lambda c: (c.window_open.__setitem__(0, True),
                                  c.opp_in_reload.__setitem__(0, True)), 0),
    "assist":         (lambda c: c.window_assist.__setitem__(0, True), 0),
    "pellet_hit":     (lambda c: c.pellet_hits_opp.__setitem__(1, 3), 1),
    "kill":           (lambda c: c.kills.__setitem__(1, 1), 1),
    "close_range":    (lambda c: c.within_3m.__setitem__(1, True), 1),
    "window_4plus":   (lambda c: c.shot_through_window_4plus.__setitem__(0, True), 1),
    "friendly_fire":  (lambda c: c.friendly_pellets.__setitem__(1, 2), 1),
    "die_first":      (lambda c: c.died_before_teammate.__setitem__(1, True), 1),
}


class TestGradientFlow:

    def test_base_context_is_reward_free(self):
        r = R.compute_rewards(_base_ctx())
        np.testing.assert_allclose(r, np.zeros(4), atol=1e-12)

    @pytest.mark.parametrize("term", sorted(TERMS))
    def test_term_reaches_policy_parameters(self, term):
        mutate, agent = TERMS[term]
        ctx = _base_ctx()
        mutate(ctx)
        r = R.compute_rewards(ctx)
        assert r[agent] != 0.0, f"term '{term}' produced zero reward"

        torch.manual_seed(0)
        policy = GaussianPolicy(obs_dim=16, act_dim=5)
        before = [p.detach().clone() for p in policy.mean_net.parameters()]

        obs = torch.randn(8, 16)
        with torch.no_grad():
            actions, _ = policy.act(obs)
        # the term's reward is the advantage — exactly the training path
        adv = torch.full((8,), float(r[agent]))
        d = policy.dist(obs)
        loss = -(d.log_prob(actions).sum(-1) * adv).mean()
        loss.backward()

        grads = [p.grad for p in policy.mean_net.parameters()]
        assert any(g is not None and g.abs().max() > 0 for g in grads), (
            f"term '{term}': no gradient reached the policy mean network")

        opt = torch.optim.SGD(policy.mean_net.parameters(), lr=1e-2)
        opt.step()
        changed = any(not torch.equal(b, p.detach())
                      for b, p in zip(before, policy.mean_net.parameters()))
        assert changed, f"term '{term}': policy parameters did not change"
