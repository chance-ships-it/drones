"""Sanity tests for the simulator. Run them after you change anything.

Two ways to run:
  .venv/bin/python tests/test_physics.py     # no extra install needed
  .venv/bin/python -m pytest tests/          # if you `pip install pytest`

Each test states one thing that must stay true. If you edit physics.py or
env.py and a test goes red, you changed behavior — on purpose or by accident.
That's the whole point: it tells you *before* you waste an hour training a
policy on broken physics.
"""

import os
import sys

# Make `drone_sim` importable when this file is run directly (python tests/...).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from drone_sim.physics import QuadrotorPhysics, QuadrotorParams
from drone_sim.env import QuadrotorEnv


def test_hover_thrust_holds_altitude():
    """Commanding exactly hover thrust should keep the drone (almost) still."""
    p = QuadrotorParams(randomize=False)
    sim = QuadrotorPhysics(1, p)
    sim.reset(torch.arange(1), torch.tensor([[0.0, 0.0, 2.0]]))
    hover_frac = (p.mass * p.gravity / 4.0) / sim.max_thrust_per_rotor
    action = torch.full((1, 4), 2 * hover_frac - 1.0)  # map to [-1, 1]
    for _ in range(250):  # 5 seconds
        sim.step(action)
    assert abs(sim.pos[0, 2].item() - 2.0) < 0.05, "drifted off hover altitude"


def test_zero_thrust_falls():
    """With motors off, the drone must fall under gravity."""
    sim = QuadrotorPhysics(1, QuadrotorParams(randomize=False))
    sim.reset(torch.arange(1), torch.tensor([[0.0, 0.0, 10.0]]))
    for _ in range(50):  # 1 second
        sim.step(torch.full((1, 4), -1.0))
    assert sim.pos[0, 2].item() < 9.0, "drone did not fall with motors off"


def test_quaternion_stays_normalized():
    """Attitude quaternion must stay unit-length through random flailing."""
    sim = QuadrotorPhysics(16, QuadrotorParams())
    sim.reset(torch.arange(16), torch.zeros(16, 3) + torch.tensor([0.0, 0.0, 3.0]))
    for _ in range(200):
        sim.step(torch.rand(16, 4) * 2 - 1)
    norms = sim.quat.norm(dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-3)


def test_env_rewards_are_finite():
    """No NaNs or infs should ever come out of a step."""
    for task in ("hover", "waypoint"):
        env = QuadrotorEnv(num_envs=32, task=task, seed=0)
        obs = env.reset()
        for _ in range(100):
            obs, reward, term, trunc, info = env.step(torch.rand(32, 4) * 2 - 1)
            assert torch.isfinite(obs).all(), f"{task}: non-finite obs"
            assert torch.isfinite(reward).all(), f"{task}: non-finite reward"


def test_waypoint_capture_advances_target():
    """Reaching a waypoint should count a hit and move the target."""
    env = QuadrotorEnv(num_envs=1, task="waypoint", seed=0)
    env.reset()
    old_target = env.target[0].clone()
    env.sim.pos[0] = env.target[0].clone()   # teleport onto the target
    env.prev_dist[0] = 0.0
    env.step(torch.zeros(1, 4))
    assert int(env.waypoints_hit[0]) == 1, "capture was not counted"
    assert not torch.equal(env.target[0], old_target), "target did not advance"


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"  FAIL  {t.__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    raise SystemExit(1 if failed else 0)
